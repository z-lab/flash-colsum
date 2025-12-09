# Flash-ColSum

**Efficient attention column-sum primitives with Triton kernels**

[![PyPI](https://img.shields.io/pypi/v/flash-colsum)](https://pypi.org/project/flash-colsum/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Flash-ColSum provides efficient implementations for computing the mean attention column sum without materializing full attention matrices.

Originally developed for [SparseVILA](https://arxiv.org/abs/2510.17777), Flash-ColSum is a general-purpose library for computing the mean column statistics of attention weights (token importance, attention analysis, etc).

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
from flash_colsum import flash_colsum

# Non-causal (ViT, BERT, etc.)
Q = torch.randn(8, 16, 2048, 64, device="cuda", dtype=torch.float16)
K = Q.clone()
col_mean = flash_colsum(Q, K)  # (8, 2048)

# Non-causal with CLS tokens (e.g., CLIP-style, first position is CLS)
# Average how much attention each token receives from the CLS token(s)
cls_col_mean = flash_colsum(Q, K, cls_len=1)  # (8, 2048)

# Causal (GPT, retrieval, etc.)
Q = torch.randn(1, 32, 128, 128, device="cuda", dtype=torch.float16)
K = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
col_mean = flash_colsum(Q, K, is_causal=True)  # (1, 4096)
```

## API

### `flash_colsum(query, key, scale=None, is_causal=False, cls_len=None)`

Compute attention column means efficiently without materializing full attention matrix.

**Parameters:**
- `query` (Tensor): Query tensor `(B, H, S, D)` or `(1, H, Q_len, D)` for causal
- `key` (Tensor): Key tensor (same shape as query for non-causal), or `K_len >= Q_len` for causal
- `scale` (float, optional): Attention scale. Default: `1/sqrt(D)`
- `is_causal` (bool): Apply causal masking. Default: `False`
- `cls_len` (int, optional): In the non-causal case, average only over the first `cls_len`
  query positions (e.g., CLS tokens). If `None`, averages over all query positions.

**Returns:**
- `Tensor`:
  - Non-causal: `(B, S)` mean per key position
    - with `cls_len=None`: averaged over all query positions and heads
    - with `cls_len>0`: averaged over the first `cls_len` query positions and all heads
  - Causal: `(1, K_len)` mean per key position (no `cls_len` support)

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
