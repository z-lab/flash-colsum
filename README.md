# Flash-ColSum

**Efficient attention column-sum primitives with Triton kernels**

[![PyPI](https://img.shields.io/pypi/v/flash-colsum)](https://pypi.org/project/flash-colsum/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Flash-ColSum provides efficient implementations for computing the attention column sum without materializing full attention matrices.

Originally developed for [SparseVILA](https://arxiv.org/abs/2510.17777), Flash-ColSum is a general-purpose library for computing the column statistics of attention weights (token importance, attention analysis, etc).

## Installation

Install from PyPI:
```bash
pip install flash-colsum
```

From source:
```bash
git clone https://github.com/z-lab/flash-colsum.git
cd flash-colsum
pip install -e .
```

## Quick Start

```python
import torch
from flash_colsum import flash_colsum, flash_colmean

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
torch.manual_seed(0)

# 1) Non-causal (square, batched)
Q = torch.randn(8, 16, 512, 64, device=device, dtype=dtype)
K = Q.clone()
col_sum = flash_colsum(Q, K)           # (8, 512)

# 2) Non-causal (non-square, B=1)
Q = torch.randn(1, 16, 1024, 64, device=device, dtype=dtype)
K = torch.randn(1, 16, 4096, 64, device=device, dtype=dtype)
col_sum = flash_colsum(Q, K)           # (1, 4096)

# 3) Causal (right-aligned, non-square)
Q = torch.randn(1, 32, 128, 128, device=device, dtype=dtype)
K = torch.randn(1, 32, 4096, 128, device=device, dtype=dtype)
col_sum = flash_colsum(Q, K, is_causal=True)      # (1, 4096)

# 4) Causal (right-aligned, square)
Q = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
K = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
col_sum = flash_colsum(Q, K, is_causal=True)      # (1, 512)

```

## API

### `flash_colsum(query, key, scale=None, is_causal=False)`

Compute the attention column sum efficiently without materializing the full attention matrix.

**Parameters:**
- `query` (Tensor): Query tensor `(B, H, S, D)` or `(1, H, Q_len, D)` for causal
- `key` (Tensor): Key tensor (same shape as query for non-causal), or `K_len >= Q_len` for causal
- `scale` (float, optional): Attention scale. Default: `1/sqrt(D)`
- `is_causal` (bool): Apply causal masking. Default: `False`

**Returns:**
- `Tensor`:
  - Non-causal: `(B, S)` column sum per key position
  - Causal (right-aligned): `(1, K_len)` column sum per key position

Notes:
- Causal masking is right-aligned: for non-square inputs, later keys see fewer queries.
- Column means can be obtained via the function below.

### `flash_colmean(query, key, scale=None, is_causal=False)`

Syntactic sugar for computing the attention mean column sum, implemented as a thin wrapper over `flash_colsum` with the correct normalization for each key position (including non-square, right-aligned causal cases).

**Returns:**
- `Tensor` with the same shape as `flash_colsum` but normalized to produce per-key means.

## Performance

Flash-ColSum achieves significant speedups and memory savings over naïve implementations:

![A6000 Benchmark Results](assets/A6000_benchmark.png)
*Benchmarked on NVIDIA RTX A6000 with FP16 precision*

![A6000 Benchmark Results](assets/5090_benchmark.png)
*Benchmarked on NVIDIA GeForce RTX 5090 with FP16 precision*

## Development

### Package Structure

Top-level layout:

```
flash-colsum/
├── flash_colsum/          # Library code
│   ├── __init__.py
│   ├── ops.py             # Public API (flash_colsum, naive_colsum)
│   ├── baselines.py       # Naive/reference implementations
│   ├── kernel_causal.py
│   ├── kernel_noncausal.py
│   └── kernel_noncausal_batched.py
├── benchmarks/            # Benchmark script
│   ├── __init__.py
│   └── benchmark_colsum.py
├── assets/                # Benchmark figures and other assets
├── tests/                 # Pytest-based tests
│   ├── test_core.py
│   └── test_benchmarks.py
└── pyproject.toml
```

### Evaluation (Tests & Benchmarks)

#### 1. Evaluate correctness (pytest)

```bash
# Fast unit tests (correctness + error handling)
pytest -v -s
```

#### 2. Evaluate efficiency (benchmarks, via pytest)

```bash
# Run only the benchmark sweeps (plot under benchmarks/out)
FLASH_COLSUM_RUN_BENCH=1 pytest tests/test_benchmarks.py -v -s

# Or: run full test suite + benchmark sweeps together
FLASH_COLSUM_RUN_BENCH=1 pytest -v -s
```

For more fine-grained control (single-point runs, custom sweeps), you can also call
the benchmark driver directly via `python -m benchmarks.benchmark_colsum` and pass
flags such as `--sweep {noncausal_batched,noncausal,causal,all}` and `--out PATH`.

## Citation

If you use Flash-ColSum in your research, please cite our SparseVILA paper:

```bibtex
@InProceedings{Khaki_2025_ICCV,
    author    = {Khaki, Samir and Guo, Junxian and Tang, Jiaming and Yang, Shang and Chen, Yukang and Plataniotis, Konstantinos N. and Lu, Yao and Han, Song and Liu, Zhijian},
    title     = {SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {23784-23794}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Flash-ColSum builds on ideas from:
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Efficient attention kernels
- [SparseVILA](https://arxiv.org/abs/2510.17777) - Token Sparsity for vision-language models
