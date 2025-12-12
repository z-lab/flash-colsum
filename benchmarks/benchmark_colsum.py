import argparse
import csv
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import torch

from flash_colsum import flash_colsum
from benchmarks.baselines import naive_colsum, triton_attention
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Modern color scheme
COLOR_OURS = "#4CAF50"   # Medium green for flash-colsum
COLOR_NAIVE = "#E91E63"  # Moderate magenta for naive/torch
COLOR_TRITON_ATTN = "#2196F3"  # Blue for triton flash attention
COLOR_OOM = "#9E9E9E"    # Neutral gray for OOM annotations

# Improved, readable global style
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['axes.facecolor'] = "#FAFAFA"
matplotlib.rcParams['figure.facecolor'] = "white"
matplotlib.rcParams['grid.color'] = "#E0E0E0"
matplotlib.rcParams['grid.linestyle'] = "--"
matplotlib.rcParams['grid.linewidth'] = 0.9


def _sync():
	if torch.cuda.is_available():
		torch.cuda.synchronize()


def measure_latency(fn, warmup: int = 10, iters: int = 50) -> float:
	"""
	Measure average latency of a pure inference function `fn`.
	Gradients are disabled so we don't allocate autograd graphs between runs.
	"""
	with torch.no_grad():
		for _ in range(warmup):
			fn()
			_sync()
		start = time.perf_counter()
		for _ in range(iters):
			fn()
			_sync()
		end = time.perf_counter()
	return (end - start) / iters


def peak_memory(fn) -> int:
	if not torch.cuda.is_available():
		return 0
	# Clear any unused cached blocks from previous runs so this measurement
	# reflects the memory footprint of the current point as much as possible.
	torch.cuda.empty_cache()
	# Reset stats so max_memory_allocated tracks this invocation only,
	# starting from the *current* allocated baseline (Q, K, etc.).
	torch.cuda.reset_peak_memory_stats()
	with torch.no_grad():
		fn()
	_sync()
	return torch.cuda.max_memory_allocated()


def run_chunked_baseline(strategy: str, Q: torch.Tensor, K: torch.Tensor, scale: float, warmup: int, iters: int, batch_chunk_size: int = None, seq_chunk_size: int = 256) -> Dict[str, Optional[float]]:
	"""
	Chunked baseline that processes in smaller chunks to avoid OOM.
	- For noncausal_batched: chunks over batch dimension using batch_chunk_size.
	- For causal_batched: chunks over batch dimension using batch_chunk_size.
	- For noncausal/causal: chunks over sequence dimension using seq_chunk_size.
	"""
	B, H, *_ = Q.shape
	
	# Compute chunk sizes before defining function to avoid capturing loop vars
	if strategy in ("noncausal_batched", "causal_batched"):
		if batch_chunk_size is None:
			# Default if not provided: half the batch
			batch_chunk_size = max(1, B // 2)
		# Use the provided chunk size directly (last successful B)
		import sys
		print(f"[Chunked baseline: B={B}, chunk_size={batch_chunk_size}, num_chunks={B//batch_chunk_size}]", file=sys.stderr)
	
	def chunked_colsum():
		if strategy == "causal":
			# For causal: Q is (1,H,Q_len,D), K is (1,H,K_len,D); chunk over key length
			# Chunk over key length dimension
			Q_len, K_len = Q.shape[2], K.shape[2]
			D = Q.shape[-1]
			
			# Compute in chunks of keys
			all_col_sums = []
			for k_start in range(0, K_len, seq_chunk_size):
				k_end = min(k_start + seq_chunk_size, K_len)
				K_chunk = K[:, :, k_start:k_end, :]
				
				# QK^T for this chunk: (1,H,Q_len,k_chunk_size)
				scores = torch.matmul(Q, K_chunk.transpose(-2, -1)) * scale
				
				# Apply causal mask
				q_indices = torch.arange(Q_len, device=Q.device).unsqueeze(1)
				k_indices = torch.arange(k_start, k_end, device=K.device).unsqueeze(0)
				mask = q_indices < k_indices
				scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
				
				# Softmax over all keys (need to defer until we have all chunks)
				all_col_sums.append(scores)
			
			# Concatenate and do full softmax
			full_scores = torch.cat(all_col_sums, dim=-1)  # (1,H,Q_len,K_len)
			attn = torch.softmax(full_scores, dim=-1)
			col_mean = attn.mean(dim=(1, 2))  # (1, K_len)
			return col_mean
		elif strategy == "causal_batched":
			# For batched causal: Q,K are (B,H,Q_len/K_len,D) - chunk over BATCH dimension
			Q_len, K_len = Q.shape[2], K.shape[2]
			all_col_means = []
			num_chunks = (B + batch_chunk_size - 1) // batch_chunk_size
			for chunk_idx in range(num_chunks):
				b_start = chunk_idx * batch_chunk_size
				b_end = min(b_start + batch_chunk_size, B)
				Q_chunk = Q[b_start:b_end]  # (chunk_B, H, Q_len, D)
				K_chunk = K[b_start:b_end]  # (chunk_B, H, K_len, D)
				
				# QK^T: (chunk_B, H, Q_len, K_len)
				scores = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) * scale
				
				# Apply right-aligned causal mask
				q_indices = torch.arange(Q_len, device=Q.device).unsqueeze(1)
				k_indices = torch.arange(K_len, device=K.device).unsqueeze(0)
				mask = q_indices < k_indices
				scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
				
				attn = torch.softmax(scores, dim=-1)
				# Mean over heads and queries -> (chunk_B, K_len)
				col_mean_chunk = attn.mean(dim=(1, 2))
				all_col_means.append(col_mean_chunk)
			# Concatenate results across batch chunks -> (B, K_len)
			return torch.cat(all_col_means, dim=0)
		elif strategy == "noncausal":
			# For non-causal unbatched: Q,K are (1,H,S,D) - chunk over sequence dimension
			S = Q.shape[2]
			D = Q.shape[-1]
			
			# Accumulate column sums across query chunks
			col_sum_accumulator = torch.zeros(1, S, device=Q.device, dtype=Q.dtype)
			num_query_chunks = 0
			
			for q_start in range(0, S, seq_chunk_size):
				q_end = min(q_start + seq_chunk_size, S)
				Q_chunk = Q[:, :, q_start:q_end, :]  # (1, H, chunk_size, D)
				
				# QK^T: (1, H, chunk_size, S)
				scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale
				attn_chunk = torch.softmax(scores, dim=-1)  # (1, H, chunk_size, S)

				# Early reduction over heads (mean) followed by sum over sequence,
				# mirroring the mean-head/sequence pattern in mem_efficient_compute_attn_weights.
				attn_chunk = attn_chunk.mean(dim=1)          # (1, chunk_size, S)
				col_sum_accumulator += attn_chunk.sum(dim=1)  # (1, S)
				num_query_chunks += (q_end - q_start)
			
			# Average over all query positions (we already averaged over heads above)
			col_mean = col_sum_accumulator / num_query_chunks  # (1, S)
			return col_mean
		else:  # "noncausal_batched"
			# For batched non-causal: Q,K are (B,H,S,D) - chunk over BATCH dimension
			S = Q.shape[2]
			D = Q.shape[-1]
			
			# Process batches in chunks and accumulate results
			all_col_means = []
			num_chunks = (B + batch_chunk_size - 1) // batch_chunk_size
			
			for chunk_idx in range(num_chunks):
				b_start = chunk_idx * batch_chunk_size
				b_end = min(b_start + batch_chunk_size, B)
				Q_chunk = Q[b_start:b_end]  # (chunk_B, H, S, D)
				K_chunk = K[b_start:b_end]  # (chunk_B, H, S, D)
				
				# Compute attention for this batch chunk
				# QK^T: (chunk_B, H, S, S)
				scores = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) * scale
				attn = torch.softmax(scores, dim=-1)  # (chunk_B, H, S, S)
				
				# Column mean for this batch chunk
				col_mean_chunk = attn.mean(dim=(1, 2))  # (chunk_B, S)
				all_col_means.append(col_mean_chunk)
			
			# Concatenate results across batches
			col_mean = torch.cat(all_col_means, dim=0)  # (B, S)
			return col_mean
	
	res: Dict[str, Optional[float]] = {}
	try:
		# Test run first to verify correctness
		result = chunked_colsum()
		
		# Now measure
		t_chunked = measure_latency(chunked_colsum, warmup, iters)
		m_chunked = peak_memory(chunked_colsum)
		res["t_chunked"] = t_chunked
		res["m_chunked"] = float(m_chunked) if m_chunked is not None else None
		
		# Verify result shape
		if strategy == "noncausal_batched":
			expected_shape = (Q.shape[0], Q.shape[2])
			assert result.shape == expected_shape, f"Chunked result shape {result.shape} != expected {expected_shape}"
	except Exception as e:
		# Print error for debugging
		import sys
		print(f"[Chunked baseline failed: {type(e).__name__}: {str(e)}]", file=sys.stderr)
		res["t_chunked"] = None
		res["m_chunked"] = None
	return res


def run_point(
	is_causal: bool,
	Q: torch.Tensor,
	K: torch.Tensor,
	scale: float,
	warmup: int,
	iters: int,
	measure_chunked: bool = False,
	last_successful_size: int = None,
	last_successful_latency: Optional[float] = None,
	last_successful_memory: Optional[float] = None,
	methods: Optional[List[str]] = None,  # None = all, or subset of ["naive", "flash_colsum", "triton_fa2"]
) -> Dict[str, Optional[float]]:
	"""
	Run benchmark for specified methods.
	
	Args:
		methods: List of methods to run. Options: "naive", "flash_colsum", "triton_fa2"
		         If None, runs all methods.
	"""
	import sys
	
	# Determine strategy name for chunked baseline compatibility
	if is_causal and Q.shape[0] > 1:
		strategy = "causal_batched"
	elif is_causal:
		strategy = "causal"
	elif Q.shape[0] == 1:
		strategy = "noncausal"
	else:
		strategy = "noncausal_batched"
	
	# Default to all methods if not specified
	if methods is None:
		methods = ["naive", "flash_colsum", "triton_fa2"]
	
	num_methods = len(methods)
	method_idx = 0
	res: Dict[str, Optional[float]] = {}
	t_ref, m_ref = None, None

	# ========== PyTorch Naive ==========
	if "naive" in methods:
		method_idx += 1
		print(f"  [{method_idx}/{num_methods}] PyTorch Naive...", end=" ", file=sys.stderr, flush=True)
		def run_ref():
			return naive_colsum(Q, K, scale=scale, is_causal=is_causal)
		try:
			t_ref = measure_latency(run_ref, warmup, iters)
			m_ref = peak_memory(run_ref)
			print(f"done ({t_ref*1000:.2f} ms)", file=sys.stderr)
		except RuntimeError as e:
			if "out of memory" in str(e).lower() or "CUDA" in str(e):
				t_ref, m_ref = None, None
				print("OOM", file=sys.stderr)
			else:
				raise
		finally:
			if torch.cuda.is_available():
				torch.cuda.synchronize()
				torch.cuda.empty_cache()
	res["t_ref"] = t_ref
	res["m_ref"] = float(m_ref) if m_ref is not None else None

	# chunked baseline (only if naive requested and OOM'd)
	if "naive" in methods and measure_chunked and t_ref is None and last_successful_size is not None:
		if strategy in ("noncausal_batched", "causal_batched") and last_successful_latency is not None:
			total_size = Q.shape[0]
			num_chunks = math.ceil(total_size / last_successful_size)
			t_chunked_est = last_successful_latency * num_chunks
			m_chunked_est = float(last_successful_memory) if last_successful_memory is not None else None
			res["t_chunked"] = t_chunked_est
			res["m_chunked"] = m_chunked_est
		else:
			if strategy == "noncausal":
				chunk_S = last_successful_size
				chunked_res = None
				while chunk_S >= 128:
					try_res = run_chunked_baseline(
						strategy, Q, K, scale, warmup, iters, seq_chunk_size=chunk_S
					)
					if try_res.get("t_chunked") is not None or try_res.get("m_chunked") is not None:
						chunked_res = try_res
						break
					chunk_S //= 2
				if chunked_res is not None:
					res.update(chunked_res)
			else:
				chunk_S = last_successful_size
				chunked_res = run_chunked_baseline(
					strategy, Q, K, scale, warmup, iters, seq_chunk_size=chunk_S
				)
				res.update(chunked_res)
	else:
		res["t_chunked"] = None
		res["m_chunked"] = None

	# ========== Flash-ColSum (Ours) ==========
	t_fast, m_fast = None, None
	if "flash_colsum" in methods:
		method_idx += 1
		print(f"  [{method_idx}/{num_methods}] Flash-ColSum...", end=" ", file=sys.stderr, flush=True)
		try:
			def run_fast():
				return flash_colsum(Q, K, scale=scale, is_causal=is_causal)
			t_fast = measure_latency(run_fast, warmup, iters)
			m_fast = peak_memory(run_fast)
			print(f"done ({t_fast*1000:.2f} ms)", file=sys.stderr)
		except RuntimeError as e:
			if "out of memory" in str(e).lower() or "CUDA" in str(e):
				t_fast, m_fast = None, None
				print("OOM", file=sys.stderr)
			else:
				t_fast, m_fast = None, None
				print(f"failed ({type(e).__name__})", file=sys.stderr)
		except Exception as e:
			t_fast, m_fast = None, None
			print(f"failed ({type(e).__name__})", file=sys.stderr)
		finally:
			if torch.cuda.is_available():
				torch.cuda.synchronize()
				torch.cuda.empty_cache()
	res["t_fast"] = t_fast
	res["m_fast"] = float(m_fast) if m_fast is not None else None

	# ========== Triton FA2 ==========
	t_triton_attn, m_triton_attn = None, None
	if "triton_fa2" in methods:
		method_idx += 1
		print(f"  [{method_idx}/{num_methods}] Triton FA2...", end=" ", file=sys.stderr, flush=True)
		V = None
		try:
			V = torch.randn_like(K)
			def run_triton_attn():
				return triton_attention(Q, K, V, is_causal=is_causal, scale=scale)
			t_triton_attn = measure_latency(run_triton_attn, warmup, iters)
			m_triton_attn = peak_memory(run_triton_attn)
			print(f"done ({t_triton_attn*1000:.2f} ms)", file=sys.stderr)
		except RuntimeError as e:
			# Handle OOM like naive baseline
			if "out of memory" in str(e).lower() or "CUDA" in str(e):
				t_triton_attn, m_triton_attn = None, None
				print("OOM", file=sys.stderr)
			else:
				t_triton_attn, m_triton_attn = None, None
				print(f"failed ({type(e).__name__})", file=sys.stderr)
		except Exception as e:
			t_triton_attn, m_triton_attn = None, None
			print(f"failed ({type(e).__name__})", file=sys.stderr)
		finally:
			# Clean up V tensor and CUDA cache
			if V is not None:
				del V
			if torch.cuda.is_available():
				torch.cuda.synchronize()
				torch.cuda.empty_cache()
	res["t_triton_attn"] = t_triton_attn
	res["m_triton_attn"] = float(m_triton_attn) if m_triton_attn is not None else None
	
	# Clear CUDA cache to prevent memory issues between benchmark points
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	
	return res


def sweep_noncausal_batched(device: torch.device, H: int, S: int, D: int, warmup: int, iters: int, out_dir: str, generate_plots: bool = False, measure_chunked: bool = True, methods: Optional[List[str]] = None):
	# Fix S=1024 and vary B up to 1024
	B_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
	results: List[Dict[str, Optional[float]]] = []
	console = Console()
	console.rule(f"[bold]Non-Causal Batched[/] S={S}, H={H}, D={D}")
	# Also print fixed sequence length for terminal/pytest visibility
	import sys
	print(f"[size] S={S}", file=sys.stderr, flush=True)
	last_successful_B = None
	last_successful_t_ref = None
	last_successful_m_ref = None
	with Progress("[progress.description]{task.description}", BarColumn(), "{task.completed}/{task.total}", TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as progress:
		task = progress.add_task("Benchmarking", total=len(B_values))
		for B in B_values:
			point = {"B": B, "H": H, "S": S, "D": D}
			Q, K = None, None
			try:
				if torch.cuda.is_available():
					torch.cuda.synchronize()
					torch.cuda.empty_cache()
				dtype = torch.float16 if device.type == "cuda" else torch.float32
				Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
				K = torch.randn(B, H, S, D, device=device, dtype=dtype)
				scale = 1.0 / math.sqrt(D)
				point.update(
					run_point(
						is_causal=False,
						Q=Q,
						K=K,
						scale=scale,
						warmup=warmup,
						iters=iters,
						measure_chunked=measure_chunked,
						last_successful_size=last_successful_B,
						last_successful_latency=last_successful_t_ref,
						last_successful_memory=last_successful_m_ref,
						methods=methods,
					)
				)
				# Track last successful B for chunking
				if point.get("t_ref") is not None:
					last_successful_B = B
					last_successful_t_ref = point.get("t_ref")
					last_successful_m_ref = point.get("m_ref")
			except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
				if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
					console.print(f"[yellow]B={B}: OOM during tensor allocation[/yellow]")
					point.update({
						"t_ref": None, "m_ref": None,
						"t_chunked": None, "m_chunked": None,
						"t_fast": None, "m_fast": None,
						"t_triton_attn": None, "m_triton_attn": None,
					})
				else:
					raise
			finally:
				if Q is not None:
					del Q
				if K is not None:
					del K
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			results.append(point)
			progress.advance(task)
	_print_table(console, "Non-Causal Batched (S=1024)", "B", results)
	if generate_plots:
		_save_and_plot(results, x_key="B", title="Non-Causal Batched (S=1024) latency", ylabel="ms", out_dir=out_dir, filename="noncausal_batched_latency.png", value_keys=("t_ref", "t_fast"), scale_ms=True)
		_save_and_plot(results, x_key="B", title="Non-Causal Batched (S=1024) memory", ylabel="GiB", out_dir=out_dir, filename="noncausal_batched_memory.png", value_keys=("m_ref", "m_fast"), scale_gib=True)
	_save_csv(results, os.path.join(out_dir, "noncausal_batched.csv"), primary_key="B")
	return results


def sweep_noncausal(device: torch.device, H: int, D: int, warmup: int, iters: int, out_dir: str, generate_plots: bool = False, measure_chunked: bool = True, methods: Optional[List[str]] = None):
	# B=1, S up to 256K
	S_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
	results: List[Dict[str, Optional[float]]] = []
	console = Console()
	console.rule(f"[bold]Non-Causal ColSum (B=1)[/] H={H}, D={D}")
	# Print the swept sequence length for terminal/pytest
	import sys
	last_successful_S = None
	last_successful_t_ref = None
	last_successful_m_ref = None
	with Progress("[progress.description]{task.description}", BarColumn(), "{task.completed}/{task.total}", TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as progress:
		task = progress.add_task("Benchmarking", total=len(S_values))
		for S in S_values:
			print(f"[size] S={S}", file=sys.stderr, flush=True)
			point = {"B": 1, "H": H, "S": S, "D": D}
			Q, K = None, None
			try:
				if torch.cuda.is_available():
					torch.cuda.synchronize()
					torch.cuda.empty_cache()
				dtype = torch.float16 if device.type == "cuda" else torch.float32
				Q = torch.randn(1, H, S, D, device=device, dtype=dtype)
				K = torch.randn(1, H, S, D, device=device, dtype=dtype)
				scale = 1.0 / math.sqrt(D)
				point.update(
					run_point(
						is_causal=False,
						Q=Q,
						K=K,
						scale=scale,
						warmup=warmup,
						iters=iters,
						measure_chunked=measure_chunked,
						last_successful_size=last_successful_S,
						last_successful_latency=last_successful_t_ref,
						last_successful_memory=last_successful_m_ref,
						methods=methods,
					)
				)
				# Track last successful S for chunking
				if point.get("t_ref") is not None:
					last_successful_S = S
					last_successful_t_ref = point.get("t_ref")
					last_successful_m_ref = point.get("m_ref")
			except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
				if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
					console.print(f"[yellow]S={S}: OOM during tensor allocation[/yellow]")
					point.update({
						"t_ref": None, "m_ref": None,
						"t_chunked": None, "m_chunked": None,
						"t_fast": None, "m_fast": None,
						"t_triton_attn": None, "m_triton_attn": None,
					})
				else:
					raise
			finally:
				if Q is not None:
					del Q
				if K is not None:
					del K
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			results.append(point)
			progress.advance(task)
	_print_table(console, "Non-Causal ColSum (B=1)", "S", results)
	if generate_plots:
		_save_and_plot(results, x_key="S", title="Non-Causal Long Sequence latency", ylabel="ms", out_dir=out_dir, filename="noncausal_latency.png", value_keys=("t_ref", "t_fast"), scale_ms=True, x_log=True)
		_save_and_plot(results, x_key="S", title="Non-Causal Long Sequence memory", ylabel="GiB", out_dir=out_dir, filename="noncausal_memory.png", value_keys=("m_ref", "m_fast"), scale_gib=True, x_log=True)
	_save_csv(results, os.path.join(out_dir, "noncausal.csv"), primary_key="S")
	return results


def sweep_causal(device: torch.device, H: int, Q_len: int, D: int, warmup: int, iters: int, out_dir: str, generate_plots: bool = False, measure_chunked: bool = True, methods: Optional[List[str]] = None):
	# B=1, Q_len=128 fixed, K_len up to 128K
	K_values = [16384, 32768, 65536, 131072, 262144]
	results: List[Dict[str, Optional[float]]] = []
	console = Console()
	console.rule(f"[bold]Causal[/] B=1, Q_len={Q_len}, H={H}, D={D}")
	# Print the swept key length (sequence length) for terminal/pytest
	import sys
	last_successful_K = None
	last_successful_t_ref = None
	last_successful_m_ref = None
	with Progress("[progress.description]{task.description}", BarColumn(), "{task.completed}/{task.total}", TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as progress:
		task = progress.add_task("Benchmarking", total=len(K_values))
		for K_len in K_values:
			print(f"[size] K_len={K_len}, Q_len={Q_len}", file=sys.stderr, flush=True)
			point = {"B": 1, "H": H, "Q_len": Q_len, "K_len": K_len, "D": D}
			Q, K = None, None
			try:
				if torch.cuda.is_available():
					torch.cuda.synchronize()
					torch.cuda.empty_cache()
				dtype = torch.float16 if device.type == "cuda" else torch.float32
				Q = torch.randn(1, H, Q_len, D, device=device, dtype=dtype)
				K = torch.randn(1, H, K_len, D, device=device, dtype=dtype)
				scale = 1.0 / math.sqrt(D)
				point.update(
					run_point(
						is_causal=True,
						Q=Q,
						K=K,
						scale=scale,
						warmup=warmup,
						iters=iters,
						measure_chunked=measure_chunked,
						last_successful_size=last_successful_K,
						last_successful_latency=last_successful_t_ref,
						last_successful_memory=last_successful_m_ref,
						methods=methods,
					)
				)
				# Track last successful K_len for chunking
				if point.get("t_ref") is not None:
					last_successful_K = K_len
					last_successful_t_ref = point.get("t_ref")
					last_successful_m_ref = point.get("m_ref")
			except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
				if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
					console.print(f"[yellow]K_len={K_len}: OOM during tensor allocation[/yellow]")
					point.update({
						"t_ref": None, "m_ref": None,
						"t_chunked": None, "m_chunked": None,
						"t_fast": None, "m_fast": None,
						"t_triton_attn": None, "m_triton_attn": None,
					})
				else:
					raise
			finally:
				if Q is not None:
					del Q
				if K is not None:
					del K
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			results.append(point)
			progress.advance(task)
	_print_table(console, "Causal (Q=128)", "K_len", results)
	if generate_plots:
		_save_and_plot(results, x_key="K_len", title="Causal (Q=128) latency", ylabel="ms", out_dir=out_dir, filename="causal_latency.png", value_keys=("t_ref", "t_fast"), scale_ms=True, x_log=True)
		_save_and_plot(results, x_key="K_len", title="Causal (Q=128) memory", ylabel="GiB", out_dir=out_dir, filename="causal_memory.png", value_keys=("m_ref", "m_fast"), scale_gib=True, x_log=True)
	_save_csv(results, os.path.join(out_dir, "causal.csv"), primary_key="K_len")
	return results


def sweep_causal_batched(device: torch.device, H: int, Q_len: int, K_len: int, D: int, warmup: int, iters: int, out_dir: str, generate_plots: bool = False, measure_chunked: bool = True, methods: Optional[List[str]] = None):
	"""Causal attention with varying batch size (fixed Q_len and K_len)."""
	# Smaller batch range due to large K_len (65536) causing high memory usage
	B_values = [1, 2, 4, 8, 16, 32]
	results: List[Dict[str, Optional[float]]] = []
	console = Console()
	console.rule(f"[bold]Causal Batched[/] Q_len={Q_len}, K_len={K_len}, H={H}, D={D}")
	# Print fixed lengths for terminal/pytest visibility
	import sys
	print(f"[size] K_len={K_len}, Q_len={Q_len}", file=sys.stderr, flush=True)
	last_successful_B = None
	last_successful_t_ref = None
	last_successful_m_ref = None
	with Progress("[progress.description]{task.description}", BarColumn(), "{task.completed}/{task.total}", TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as progress:
		task = progress.add_task("Benchmarking", total=len(B_values))
		for B in B_values:
			point = {"B": B, "H": H, "Q_len": Q_len, "K_len": K_len, "D": D}
			Q, K = None, None
			try:
				if torch.cuda.is_available():
					torch.cuda.synchronize()
					torch.cuda.empty_cache()
				dtype = torch.float16 if device.type == "cuda" else torch.float32
				Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
				K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
				scale = 1.0 / math.sqrt(D)
				point.update(
					run_point(
						is_causal=True,
						Q=Q,
						K=K,
						scale=scale,
						warmup=warmup,
						iters=iters,
						measure_chunked=measure_chunked,
						last_successful_size=last_successful_B,
						last_successful_latency=last_successful_t_ref,
						last_successful_memory=last_successful_m_ref,
						methods=methods,
					)
				)
				if point.get("t_ref") is not None:
					last_successful_B = B
					last_successful_t_ref = point.get("t_ref")
					last_successful_m_ref = point.get("m_ref")
			except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
				if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
					console.print(f"[yellow]B={B}: OOM during tensor allocation[/yellow]")
					point.update({
						"t_ref": None, "m_ref": None,
						"t_chunked": None, "m_chunked": None,
						"t_fast": None, "m_fast": None,
						"t_triton_attn": None, "m_triton_attn": None,
					})
				else:
					raise
			finally:
				if Q is not None:
					del Q
				if K is not None:
					del K
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			results.append(point)
			progress.advance(task)
	_print_table(console, f"Causal Batched (Q={Q_len}, K={K_len})", "B", results)
	if generate_plots:
		_save_and_plot(results, x_key="B", title=f"Causal Batched (Q={Q_len}, K={K_len}) latency", ylabel="ms", out_dir=out_dir, filename="causal_batched_latency.png", value_keys=("t_ref", "t_fast"), scale_ms=True)
		_save_and_plot(results, x_key="B", title=f"Causal Batched (Q={Q_len}, K={K_len}) memory", ylabel="GiB", out_dir=out_dir, filename="causal_batched_memory.png", value_keys=("m_ref", "m_fast"), scale_gib=True)
	_save_csv(results, os.path.join(out_dir, "causal_batched.csv"), primary_key="B")
	return results


def _save_csv(rows: List[Dict[str, Optional[float]]], path: str, primary_key: str = None):
	"""Save results to CSV, merging with existing data if present.
	
	Args:
		rows: List of result dictionaries
		path: Path to save the CSV
		primary_key: The key to use as the unique identifier for merging rows.
		             If None, uses the first key in the first row.
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	if not rows:
		return
	
	# Determine the primary key
	new_keys = list(rows[0].keys())
	if primary_key is None:
		primary_key = new_keys[0]  # First column is typically the x-axis key
	
	# Try to load existing CSV and merge
	existing_data = {}
	all_keys = set(new_keys)
	if os.path.exists(path):
		try:
			with open(path, "r", newline="") as f:
				reader = csv.DictReader(f)
				for row in reader:
					pk_val = row[primary_key]
					# Convert numeric strings back to appropriate types
					parsed_row = {}
					for k, v in row.items():
						if v == '' or v == 'None':
							parsed_row[k] = None
						else:
							try:
								parsed_row[k] = float(v) if '.' in v else int(v)
							except ValueError:
								parsed_row[k] = v
					existing_data[pk_val] = parsed_row
					all_keys.update(row.keys())
		except Exception:
			pass  # If reading fails, just overwrite
	
	# Merge new rows into existing data
	for row in rows:
		pk_val = str(row[primary_key])
		if pk_val in existing_data:
			# Update only non-None values from new row
			for k, v in row.items():
				if v is not None:
					existing_data[pk_val][k] = v
		else:
			existing_data[pk_val] = row.copy()
		all_keys.update(row.keys())
	
	# Sort rows by primary key (numeric sort if possible)
	try:
		sorted_keys = sorted(existing_data.keys(), key=lambda x: float(x))
	except ValueError:
		sorted_keys = sorted(existing_data.keys())
	
	# Write merged data
	fieldnames = [primary_key] + [k for k in sorted(all_keys) if k != primary_key]
	with open(path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for pk_val in sorted_keys:
			w.writerow(existing_data[pk_val])


def _save_and_plot(rows: List[Dict[str, Optional[float]]], x_key: str, title: str, ylabel: str, out_dir: str, filename: str, value_keys: Tuple[str, str], scale_ms: bool = False, scale_mib: bool = False, scale_gib: bool = False, x_log: bool = False):
	os.makedirs(out_dir, exist_ok=True)
	x_vals = [r[x_key] for r in rows]
	y1 = rows_to_series(rows, value_keys[0])
	y2 = rows_to_series(rows, value_keys[1])
	plt.figure(figsize=(10, 5))
	width = 0.35
	indices = range(len(x_vals))
	# convert scales
	if scale_ms:
		y1_plot = [v * 1000 if v is not None else None for v in y1]
		y2_plot = [v * 1000 if v is not None else None for v in y2]
	elif scale_mib:
		y1_plot = [v / (1024**2) if v is not None else None for v in y1]
		y2_plot = [v / (1024**2) if v is not None else None for v in y2]
	elif scale_gib:
		y1_plot = [v / (1024**3) if v is not None else None for v in y1]
		y2_plot = [v / (1024**3) if v is not None else None for v in y2]
	else:
		y1_plot, y2_plot = y1, y2
	# bar plotting with None handling
	ind_list = list(indices)
	y1_vals = [v if v is not None else 0.0 for v in y1_plot]
	y2_vals = [v if v is not None else 0.0 for v in y2_plot]
	plt.bar([i - width/2 for i in ind_list], y1_vals, width=width, label=value_keys[0])
	plt.bar([i + width/2 for i in ind_list], y2_vals, width=width, label=value_keys[1])
	plt.xticks(ind_list, [str(x) for x in x_vals], rotation=45)
	if x_log:
		plt.xscale("linear")  # for categorical ticks, keep linear
	plt.title(title)
	plt.ylabel(ylabel)
	plt.legend()
	plt.tight_layout()
	out_path = os.path.join(out_dir, filename)
	plt.savefig(out_path)
	plt.close()


def rows_to_series(rows: List[Dict[str, Optional[float]]], key: str) -> List[Optional[float]]:
	return [r.get(key) for r in rows]


def _print_table(console: Console, title: str, x_key: str, rows: List[Dict[str, Optional[float]]]) -> None:
	table = Table(title=title)
	table.add_column(x_key, justify="right", style="bold")
	table.add_column("PyTorch Naive", justify="right")
	table.add_column("Chunked", justify="right")
	table.add_column("Flash-ColSum", justify="right")
	table.add_column("Triton FA2", justify="right")
	table.add_column("Speedup", justify="right")
	table.add_column("Naive GiB", justify="right")
	table.add_column("ColSum GiB", justify="right")
	table.add_column("Mem Savings", justify="right")
	for r in rows:
		xv = str(r.get(x_key))
		t_ref = r.get("t_ref"); t_chunked = r.get("t_chunked"); t_fast = r.get("t_fast")
		t_triton_attn = r.get("t_triton_attn")
		m_ref = r.get("m_ref"); m_chunked = r.get("m_chunked"); m_fast = r.get("m_fast")
		t_ref_s = f"{t_ref*1000:.2f}" if t_ref is not None else "OOM"
		t_chunked_s = f"{t_chunked*1000:.2f}" if t_chunked is not None else "-"
		t_fast_s = f"{t_fast*1000:.2f}" if t_fast is not None else "OOM"
		t_triton_attn_s = f"{t_triton_attn*1000:.2f}" if t_triton_attn is not None else "OOM"
		
		# Compare flash_colsum to either naive or chunked (whichever is available)
		t_baseline = t_ref if t_ref is not None else t_chunked
		speed = f"{(t_baseline/t_fast):.2f}x" if (t_baseline and t_fast and t_fast > 0) else "-"
		
		# If the naive run OOM'd in time, treat memory as OOM as well
		if t_ref is None:
			m_ref_s = "OOM"
		else:
			m_ref_s = f"{(m_ref/(1024**3)):.2f}" if m_ref is not None else "NA"
		m_fast_s = f"{(m_fast/(1024**3)):.2f}" if m_fast is not None else "NA"
		
		# Memory savings: how much LESS memory flash-colsum uses (m_ref / m_fast)
		m_savings = f"{(m_ref/m_fast):.2f}x" if (m_ref and m_fast and m_fast > 0) else "-"
		table.add_row(xv, t_ref_s, t_chunked_s, t_fast_s, t_triton_attn_s, speed, m_ref_s, m_fast_s, m_savings)
	console.print(table)


def create_unified_plot(
	noncausal_batched_results: List[Dict[str, Optional[float]]],
	noncausal_results: List[Dict[str, Optional[float]]],
	causal_results: List[Dict[str, Optional[float]]],
	out_dir: str,
	filename: str = "unified_benchmark.png",
	causal_batched_results: Optional[List[Dict[str, Optional[float]]]] = None,
):
	"""
	Create a unified plot with latency (top row) and memory (bottom row)
	for all benchmark types. Includes chunked baseline for OOM cases.
	
	Column order: batched non-causal, non-causal, batched causal, causal
	"""
	# Determine number of columns based on whether causal_batched is provided
	num_cols = 4 if causal_batched_results else 3
	fig, axes = plt.subplots(2, num_cols, figsize=(5.5 * num_cols, 6.7))
	
	# Titles for each column - order: batched non-causal, non-causal, batched causal, causal
	# Square = Q_len == K_len, Non-Square = Q_len != K_len  
	# Subtitle shows params + square/non-square in light style
	if causal_batched_results:
		titles = [
			"Batched Non-Causal",
			"Non-Causal",
			"Batched Causal",
			"Causal",
		]
		subtitles = [
			"S=1024  (square)",
			"B=1  (square)",
			"Q=128, K=65536  (non-square)",
			"Q=128  (non-square)",
		]
		configs = [
			(noncausal_batched_results, "B", "b = batch size", False),
			(noncausal_results, "S", "s = sequence length", True),
			(causal_batched_results, "B", "b = batch size", False),
			(causal_results, "K_len", "k = key length", True),
		]
	else:
		titles = [
			"Batched Non-Causal",
			"Non-Causal",
			"Causal",
		]
		subtitles = [
			"S=1024  (square)",
			"B=1  (square)",
			"Q=128  (non-square)",
		]
		configs = [
			(noncausal_batched_results, "B", "b = batch size", False),
			(noncausal_results, "S", "s = sequence length", True),
			(causal_results, "K_len", "k = key length", True),
		]
	
	for col, (results, x_key, x_label, x_log) in enumerate(configs):
		if not results:
			continue
			
		x_vals = [r[x_key] for r in results]
		
		# Extract latency data (top row)
		t_ref = [r.get("t_ref") for r in results]
		t_chunked = [r.get("t_chunked") for r in results]
		t_fast = [r.get("t_fast") for r in results]
		t_triton_attn = [r.get("t_triton_attn") for r in results]
		
		# Extract memory data (bottom row)
		m_ref = [r.get("m_ref") for r in results]
		m_chunked = [r.get("m_chunked") for r in results]
		m_fast = [r.get("m_fast") for r in results]
		m_triton_attn = [r.get("m_triton_attn") for r in results]
		
		# Convert to ms and GiB
		t_ref_ms = [v * 1000 if v is not None else None for v in t_ref]
		t_chunked_ms = [v * 1000 if v is not None else None for v in t_chunked]
		t_fast_ms = [v * 1000 if v is not None else None for v in t_fast]
		t_triton_attn_ms = [v * 1000 if v is not None else None for v in t_triton_attn]
		m_ref_gib = [v / (1024**3) if v is not None else None for v in m_ref]
		m_chunked_gib = [v / (1024**3) if v is not None else None for v in m_chunked]
		m_fast_gib = [v / (1024**3) if v is not None else None for v in m_fast]
		m_triton_attn_gib = [v / (1024**3) if v is not None else None for v in m_triton_attn]
		
		# Plot latency (top row)
		ax_lat = axes[0, col]
		# Two-line title: bold main title, light grey italic subtitle
		ax_lat.set_title(f"{titles[col]}\n{subtitles[col]}", fontweight='bold', pad=10)
		# Override the subtitle styling by using text directly
		ax_lat.title.set_fontweight('bold')
		# Create a custom title with mixed styling
		ax_lat.set_title("")  # Clear default
		ax_lat.text(0.5, 1.08, titles[col], transform=ax_lat.transAxes, 
				   fontsize=11, fontweight='bold', ha='center', va='bottom')
		ax_lat.text(0.5, 1.02, subtitles[col], transform=ax_lat.transAxes,
				   fontsize=9, fontstyle='italic', color='#666666', ha='center', va='bottom')
		
		# Find first OOM point for naive
		first_oom_idx = None
		for i, t in enumerate(t_ref_ms):
			if t is None:
				first_oom_idx = i
				break
		
		# Plot naive baseline (until OOM)
		x_ref_lat = [x for x, y in zip(x_vals, t_ref_ms) if y is not None]
		y_ref_lat = [y for y in t_ref_ms if y is not None]
		if x_ref_lat and y_ref_lat:
			ax_lat.plot(x_ref_lat, y_ref_lat, marker='o', linewidth=2, markersize=6, 
					   color=COLOR_NAIVE, label='PyTorch Naive ColSum', zorder=3)
		
		# Plot chunked baseline (from first OOM onwards)
		if first_oom_idx is not None:
			x_chunked_lat = [x_vals[i] for i in range(first_oom_idx, len(x_vals)) if t_chunked_ms[i] is not None]
			y_chunked_lat = [t_chunked_ms[i] for i in range(first_oom_idx, len(x_vals)) if t_chunked_ms[i] is not None]
			# Ensure the dashed chunked line visually connects to the last native PyTorch point
			if x_chunked_lat and y_chunked_lat and first_oom_idx > 0:
				prev_idx = first_oom_idx - 1
				if t_ref_ms[prev_idx] is not None:
					x_chunked_lat.insert(0, x_vals[prev_idx])
					y_chunked_lat.insert(0, t_ref_ms[prev_idx])
			if x_chunked_lat and y_chunked_lat:
				ax_lat.plot(
					x_chunked_lat,
					y_chunked_lat,
					marker='D',
					linewidth=2,
					markersize=5,
					color=COLOR_NAIVE,
					linestyle='--',
					alpha=0.7,
					label='PyTorch Naive (Chunked) ColSum',
					zorder=2,
				)
			# Mark the OOM boundary as a vertical line at the last non-OOM point,
			# even if the chunked baseline itself is missing or OOM.
			if first_oom_idx > 0 and t_ref_ms[first_oom_idx - 1] is not None:
				x_boundary = x_vals[first_oom_idx - 1]
				ax_lat.axvline(
					x_boundary,
					color=COLOR_OOM,
					linestyle='--',
					alpha=0.7,
					linewidth=2.0,
				)
		
		# Plot flash-colsum
		x_fast_lat = [x for x, y in zip(x_vals, t_fast_ms) if y is not None]
		y_fast_lat = [y for y in t_fast_ms if y is not None]
		if x_fast_lat and y_fast_lat:
			ax_lat.plot(x_fast_lat, y_fast_lat, marker='s', linewidth=2, markersize=6,
					   color=COLOR_OURS, label='Flash-ColSum', zorder=4)
		
		# Find first OOM point for triton attention
		first_triton_oom_idx = None
		for i, t in enumerate(t_triton_attn_ms):
			if t is None:
				first_triton_oom_idx = i
				break
		
		# Plot triton attention (standard flash attention for reference)
		x_triton_attn_lat = [x for x, y in zip(x_vals, t_triton_attn_ms) if y is not None]
		y_triton_attn_lat = [y for y in t_triton_attn_ms if y is not None]
		if x_triton_attn_lat and y_triton_attn_lat:
			ax_lat.plot(x_triton_attn_lat, y_triton_attn_lat, marker='^', linewidth=2, markersize=6,
					   color=COLOR_TRITON_ATTN, label='Triton FA2', zorder=3, linestyle='--')
		
		# Mark triton attention OOM boundary
		if first_triton_oom_idx is not None:
			if first_triton_oom_idx > 0 and t_triton_attn_ms[first_triton_oom_idx - 1] is not None:
				# OOM after some successful points - draw line at last successful point
				x_triton_boundary = x_vals[first_triton_oom_idx - 1]
			elif first_triton_oom_idx == 0:
				# OOM at first point - draw line at first x value
				x_triton_boundary = x_vals[0]
			else:
				x_triton_boundary = None
			if x_triton_boundary is not None:
				ax_lat.axvline(
					x_triton_boundary,
					color=COLOR_TRITON_ATTN,
					linestyle=':',
					alpha=0.7,
					linewidth=2.0,
				)
		
		if x_log:
			ax_lat.set_xscale('log', base=2)
			ax_lat.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
		else:
			ax_lat.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
		ax_lat.set_ylabel('Latency (ms)', fontweight='bold')
		ax_lat.grid(True, alpha=0.3, linestyle='--', linewidth=1.1)
		# Draw OOM label after all curves so axis limits are final.
		# Use a small *pixel* offset to keep spacing visually consistent across subplots.
		if first_oom_idx is not None and first_oom_idx > 0 and t_ref_ms[first_oom_idx - 1] is not None:
			x_boundary = x_vals[first_oom_idx - 1]
			ax_lat.annotate(
				"OOM",
				xy=(x_boundary, 0.5),              # data x, centered in axes-y
				xycoords=("data", "axes fraction"),
				xytext=(-8, 0),                    # 8 px to the left
				textcoords="offset points",
				color=COLOR_OOM,
				fontweight="bold",
				rotation=90,
				va="center",
				ha="center",
				fontsize=11,
				alpha=0.9,
			)
		
		# Draw triton attention OOM label
		if first_triton_oom_idx is not None:
			if first_triton_oom_idx > 0 and t_triton_attn_ms[first_triton_oom_idx - 1] is not None:
				x_triton_boundary = x_vals[first_triton_oom_idx - 1]
			elif first_triton_oom_idx == 0:
				x_triton_boundary = x_vals[0]
			else:
				x_triton_boundary = None
			# Only show label if different from naive OOM point (to avoid overlap)
			show_label = x_triton_boundary is not None
			if show_label and first_oom_idx is not None and first_oom_idx > 0:
				naive_boundary = x_vals[first_oom_idx - 1]
				if x_triton_boundary == naive_boundary:
					show_label = False
			if show_label and x_triton_boundary is not None:
				ax_lat.annotate(
					"OOM",
					xy=(x_triton_boundary, 0.3),
					xycoords=("data", "axes fraction"),
					xytext=(8, 0),                    # 8 px to the right
					textcoords="offset points",
					color=COLOR_TRITON_ATTN,
					fontweight="bold",
					rotation=90,
					va="center",
					ha="center",
					fontsize=9,
					alpha=0.9,
				)
		
		# Plot memory (bottom row)
		ax_mem = axes[1, col]
		
		# Plot naive baseline memory (until OOM)
		x_ref_mem = [x for x, y in zip(x_vals, m_ref_gib) if y is not None]
		y_ref_mem = [y for y in m_ref_gib if y is not None]
		if x_ref_mem and y_ref_mem:
			ax_mem.plot(x_ref_mem, y_ref_mem, marker='o', linewidth=2, markersize=6,
					   color=COLOR_NAIVE, label='PyTorch Naive ColSum', zorder=3)
		
		# Plot chunked baseline memory (from first OOM onwards)
		if first_oom_idx is not None:
			x_chunked_mem = [x_vals[i] for i in range(first_oom_idx, len(x_vals)) if m_chunked_gib[i] is not None]
			y_chunked_mem = [m_chunked_gib[i] for i in range(first_oom_idx, len(x_vals)) if m_chunked_gib[i] is not None]
			# Ensure the dashed chunked memory line connects to the last native PyTorch memory point
			if x_chunked_mem and y_chunked_mem and first_oom_idx > 0:
				prev_idx = first_oom_idx - 1
				if m_ref_gib[prev_idx] is not None:
					x_chunked_mem.insert(0, x_vals[prev_idx])
					y_chunked_mem.insert(0, m_ref_gib[prev_idx])
			if x_chunked_mem and y_chunked_mem:
				ax_mem.plot(
					x_chunked_mem,
					y_chunked_mem,
					marker='D',
					linewidth=2,
					markersize=5,
					color=COLOR_NAIVE,
					linestyle='--',
					alpha=0.7,
					label='PyTorch Naive (Chunked) ColSum',
					zorder=2,
				)
			# Vertical OOM boundary at last non-OOM point, even if chunked is missing/OOM
			if first_oom_idx > 0 and m_ref_gib[first_oom_idx - 1] is not None:
				x_boundary = x_vals[first_oom_idx - 1]
				ax_mem.axvline(
					x_boundary,
					color=COLOR_OOM,
					linestyle='--',
					alpha=0.7,
					linewidth=2.0,
				)
		
		# Plot flash-colsum memory
		x_fast_mem = [x for x, y in zip(x_vals, m_fast_gib) if y is not None]
		y_fast_mem = [y for y in m_fast_gib if y is not None]
		if x_fast_mem and y_fast_mem:
			ax_mem.plot(x_fast_mem, y_fast_mem, marker='s', linewidth=2, markersize=6,
					   color=COLOR_OURS, label='Flash-ColSum', zorder=4)
		
		# Plot triton attention memory
		x_triton_attn_mem = [x for x, y in zip(x_vals, m_triton_attn_gib) if y is not None]
		y_triton_attn_mem = [y for y in m_triton_attn_gib if y is not None]
		if x_triton_attn_mem and y_triton_attn_mem:
			ax_mem.plot(x_triton_attn_mem, y_triton_attn_mem, marker='^', linewidth=2, markersize=6,
					   color=COLOR_TRITON_ATTN, label='Triton FA2', zorder=3, linestyle='--')
		
		# Mark triton attention OOM boundary on memory plot
		if first_triton_oom_idx is not None:
			if first_triton_oom_idx > 0 and m_triton_attn_gib[first_triton_oom_idx - 1] is not None:
				x_triton_boundary = x_vals[first_triton_oom_idx - 1]
			elif first_triton_oom_idx == 0:
				x_triton_boundary = x_vals[0]
			else:
				x_triton_boundary = None
			if x_triton_boundary is not None:
				ax_mem.axvline(
					x_triton_boundary,
					color=COLOR_TRITON_ATTN,
					linestyle=':',
					alpha=0.7,
					linewidth=2.0,
				)
		
		if x_log:
			ax_mem.set_xscale('log', base=2)
			ax_mem.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
		else:
			ax_mem.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
		ax_mem.set_xlabel(x_label, fontweight='bold')
		ax_mem.set_ylabel('Memory (GiB)', fontweight='bold')
		ax_mem.grid(True, alpha=0.3, linestyle='--', linewidth=1.1)
		# Draw OOM label after all curves so axis limits are final
		if first_oom_idx is not None and first_oom_idx > 0 and m_ref_gib[first_oom_idx - 1] is not None:
			x_boundary = x_vals[first_oom_idx - 1]
			ax_mem.annotate(
				"OOM",
				xy=(x_boundary, 0.5),
				xycoords=("data", "axes fraction"),
				xytext=(-8, 0),                    # 8 px to the left
				textcoords="offset points",
				color=COLOR_OOM,
				fontweight="bold",
				rotation=90,
				va="center",
				ha="center",
				fontsize=11,
				alpha=0.9,
			)
	
	# Single, consolidated legend for the entire figure with specific ordering
	# Order: PyTorch Naive ColSum, PyTorch Naive (Chunked) ColSum, Triton FA2, Flash-ColSum
	desired_order = [
		'PyTorch Naive ColSum',
		'PyTorch Naive (Chunked) ColSum', 
		'Triton FA2',
		'Flash-ColSum',
	]
	handles_dict = {}
	for ax in axes.flatten():
		h, l = ax.get_legend_handles_labels()
		for handle, label in zip(h, l):
			if label not in handles_dict:
				handles_dict[label] = handle
	
	# Build ordered handles/labels
	handles_ordered = []
	labels_ordered = []
	for label in desired_order:
		if label in handles_dict:
			handles_ordered.append(handles_dict[label])
			labels_ordered.append(label)
	
	if handles_ordered:
		fig.legend(
			handles_ordered,
			labels_ordered,
			loc="upper center",
			ncol=len(labels_ordered),
			frameon=False,
			bbox_to_anchor=(0.5, 1.05),
		)
	
	plt.tight_layout()
	out_path = os.path.join(out_dir, filename)
	os.makedirs(out_dir, exist_ok=True)
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved unified plot to: {out_path}")

def sweep_all_unified(device: torch.device, warmup: int = None, iters: int = None, out_dir: str = "benchmarks/out", methods: Optional[List[str]] = None):
	"""
	Run all four benchmark sweeps and generate a unified plot.
	Uses appropriate warmup/iters for each benchmark type if not specified.
	"""
	console = Console()
	console.rule("[bold cyan]Running All Benchmarks for Unified Plot[/bold cyan]")
	
	# Use default values if not provided
	# Vision benchmarks: 25 warmup, 250 iters
	# Text causal: 100 warmup, 1000 iters
	vision_warmup = warmup if warmup is not None else 25
	vision_iters = iters if iters is not None else 250
	text_warmup = warmup if warmup is not None else 100
	text_iters = iters if iters is not None else 1000
	
	# Run non-causal batched (S=1024, vary B) - no individual plots
	console.print("\n[bold]1/4: Non-Causal Batched[/bold]")
	console.print(f"[dim]warmup={vision_warmup}, iters={vision_iters}[/dim]")
	noncausal_batched_results = sweep_noncausal_batched(
		device=device, H=16, S=1024, D=64, 
		warmup=vision_warmup, iters=vision_iters, out_dir=out_dir, generate_plots=False, methods=methods
	)
	
	# Run non-causal (B=1, vary S) - no individual plots
	console.print("\n[bold]2/4: Non-Causal ColSum (B=1)[/bold]")
	console.print(f"[dim]warmup={vision_warmup}, iters={vision_iters}[/dim]")
	noncausal_results = sweep_noncausal(
		device=device, H=16, D=64, 
		warmup=vision_warmup, iters=vision_iters, out_dir=out_dir, generate_plots=False, methods=methods
	)
	
	# Run causal (B=1, Q_len=128, vary K_len) - no individual plots
	console.print("\n[bold]3/4: Causal[/bold]")
	console.print(f"[dim]warmup={text_warmup}, iters={text_iters}[/dim]")
	causal_results = sweep_causal(
		device=device, H=32, Q_len=128, D=128, 
		warmup=text_warmup, iters=text_iters, out_dir=out_dir, generate_plots=False, methods=methods
	)
	
	# Run causal batched (Q_len=128, K_len=65536, vary B) - no individual plots
	console.print("\n[bold]4/4: Causal Batched[/bold]")
	console.print(f"[dim]warmup={text_warmup}, iters={text_iters}[/dim]")
	causal_batched_results = sweep_causal_batched(
		device=device, H=32, Q_len=128, K_len=65536, D=128, 
		warmup=text_warmup, iters=text_iters, out_dir=out_dir, generate_plots=False, methods=methods
	)
	
	# Generate unified plot
	console.print("\n[bold green]Generating unified plot...[/bold green]")
	create_unified_plot(noncausal_batched_results, noncausal_results, causal_results, 
					   out_dir, filename="unified_benchmark.png",
					   causal_batched_results=causal_batched_results)
	console.print("[bold green]✓ All benchmarks complete![/bold green]\n")


def main():
	parser = argparse.ArgumentParser(description="Flash-ColSum benchmark")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations (default: 100 for individual sweeps, auto for --sweep all)")
	parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations (default: 1000 for individual sweeps, auto for --sweep all)")
	parser.add_argument("--out", type=str, default="benchmarks/out")
	parser.add_argument("--sweep", type=str, choices=["noncausal_batched", "noncausal", "causal", "causal_batched", "all"], help="Run a predefined sweep and generate charts")
	parser.add_argument("--method", type=str, nargs="+", choices=["naive", "flash_colsum", "triton_fa2"], 
					   help="Run only specific methods (default: all). Options: naive, flash_colsum, triton_fa2")
	# fallback single-point args
	parser.add_argument("--causal", action="store_true", help="Use causal masking for single-point benchmark")
	parser.add_argument("--B", type=int, default=8)
	parser.add_argument("--H", type=int, default=16)
	parser.add_argument("--S", type=int, default=1024)
	parser.add_argument("--D", type=int, default=128)
	parser.add_argument("--Q_len", type=int, default=128)
	parser.add_argument("--K_len", type=int, default=4096)
	args = parser.parse_args()

	device = torch.device(args.device)

	if args.sweep:
		methods = args.method  # Will be None if not specified (run all)
		if args.sweep == "noncausal_batched":
			sweep_noncausal_batched(device=device, H=args.H, S=1024, D=args.D, warmup=args.warmup, iters=args.iters, out_dir=args.out, generate_plots=True, methods=methods)
		elif args.sweep == "noncausal":
			sweep_noncausal(device=device, H=args.H, D=args.D, warmup=args.warmup, iters=args.iters, out_dir=args.out, generate_plots=True, methods=methods)
		elif args.sweep == "causal":
			sweep_causal(device=device, H=args.H, Q_len=128, D=args.D, warmup=args.warmup, iters=args.iters, out_dir=args.out, generate_plots=True, methods=methods)
		elif args.sweep == "causal_batched":
			sweep_causal_batched(device=device, H=args.H, Q_len=128, K_len=65536, D=args.D, warmup=args.warmup, iters=args.iters, out_dir=args.out, generate_plots=True, methods=methods)
		elif args.sweep == "all":
			# For sweep all, use per-benchmark defaults unless explicitly overridden
			sweep_all_unified(device=device, out_dir=args.out, methods=methods)
		return

	# Single-point fallback
	dtype = torch.float16 if device.type == "cuda" else torch.float32
	if args.causal:
		if args.B != 1:
			raise ValueError("causal attention requires B==1")
		Q = torch.randn(1, args.H, args.Q_len, args.D, device=device, dtype=dtype)
		K = torch.randn(1, args.H, args.K_len, args.D, device=device, dtype=dtype)
	else:
		Q = torch.randn(args.B, args.H, args.S, args.D, device=device, dtype=dtype)
		K = torch.randn(args.B, args.H, args.S, args.D, device=device, dtype=dtype)

	D = Q.shape[-1]
	scale = 1.0 / math.sqrt(D)
	res = run_point(is_causal=args.causal, Q=Q, K=K, scale=scale, warmup=args.warmup, iters=args.iters, methods=args.method)
	print(f"latency_ref_ms={None if res['t_ref'] is None else res['t_ref']*1000:.3f}, latency_fast_ms={None if res['t_fast'] is None else res['t_fast']*1000:.3f}")
	print(f"mem_ref_bytes={res['m_ref']}, mem_fast_bytes={res['m_fast']}")


if __name__ == "__main__":
	main()


