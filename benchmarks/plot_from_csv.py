import csv
import os
from typing import Dict, List, Optional

import torch  # noqa: F401

from benchmarks import benchmark_colsum as bc


def _load_csv(path: str) -> List[Dict[str, Optional[float]]]:
	"""
	Load a benchmark CSV into a list of dictionaries, parsing numeric fields when possible.
	"""
	if not os.path.exists(path):
		print(f"[warn] CSV not found: {path}")
		return []
	rows: List[Dict[str, Optional[float]]] = []
	with open(path, newline="") as f:
		reader = csv.DictReader(f)
		for line in reader:
			entry: Dict[str, Optional[float]] = {}
			for k, v in line.items():
				if v is None or v == "" or v.lower() == "none":
					entry[k] = None
					continue
				try:
					# Heuristic parse: int if possible, else float, else string
					if any(ch in v for ch in [".", "e", "E"]):
						entry[k] = float(v)
					else:
						entry[k] = int(v)
				except ValueError:
					entry[k] = v
			rows.append(entry)
	return rows


def plot_unified_from_existing(out_dir: str = "benchmarks/out") -> None:
	"""
	Recreate the unified benchmark plot from existing CSVs, without re-running sweeps.

	Expected files under out_dir:
	- noncausal_batched.csv
	- noncausal.csv
	- causal.csv
	- causal_batched.csv (optional)
	"""
	noncausal_batched_csv = os.path.join(out_dir, "noncausal_batched.csv")
	noncausal_csv = os.path.join(out_dir, "noncausal.csv")
	causal_csv = os.path.join(out_dir, "causal.csv")
	causal_batched_csv = os.path.join(out_dir, "causal_batched.csv")

	noncausal_batched_results = _load_csv(noncausal_batched_csv)
	noncausal_results = _load_csv(noncausal_csv)
	causal_results = _load_csv(causal_csv)
	causal_batched_results = _load_csv(causal_batched_csv)  # Optional

	if not noncausal_batched_results:
		print(f"[error] Missing or empty CSV: {noncausal_batched_csv}")
		return
	if not noncausal_results:
		print(f"[error] Missing or empty CSV: {noncausal_csv}")
		return
	if not causal_results:
		print(f"[error] Missing or empty CSV: {causal_csv}")
		return
	
	# causal_batched is optional - just warn if missing
	if not causal_batched_results:
		print(f"[warn] Missing or empty CSV: {causal_batched_csv} (will generate 3-column plot)")

	bc.create_unified_plot(
		noncausal_batched_results,
		noncausal_results,
		causal_results,
		out_dir=out_dir,
		filename="unified_benchmark.png",
		causal_batched_results=causal_batched_results if causal_batched_results else None,
	)
	print(f"[ok] Wrote unified_benchmark.png to {out_dir}")


if __name__ == "__main__":
	plot_unified_from_existing()


