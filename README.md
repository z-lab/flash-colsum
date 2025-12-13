# Flash-ColSum

[![PyPI](https://img.shields.io/pypi/v/flash-colsum)](https://pypi.org/project/flash-colsum/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Flash-ColSum** provides highly optimized Triton kernels for computing the column sums (or means) of the attention matrix **without materializing the full $O(N^2)$ attention weights**.

This primitive is essential for efficient KV-cache pruning, token importance estimation, and attention analysis in Large Language Models (LLMs) and Vision-Language Models (VLMs), such as [SparseVILA](https://arxiv.org/abs/2510.17777).

## Highlights

- **🚀 Fast & Efficient**: Fused kernels compute column statistics in **$O(N)$ memory**, enabling 128k+ context lengths.
- **🧩 Flexible**: Supports **causal** and **non-causal** attention with irregular shapes ($M \neq N$).
- **✅ Exact**: Uses online softmax for numerical precision and correct causal masking.

## Prerequisites

- **Python**: 3.10+
- **PyTorch**: 2.1+ (with CUDA support)
- **Triton**: 3.0.0+
- **GPU**: NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer recommended)

## Installation

Install from PyPI:
```bash
pip install flash-colsum
```

Or build from source:
```bash
git clone https://github.com/z-lab/flash-colsum.git
cd flash-colsum
pip install -e .
```

## Usage

### 1. Non-Causal Attention (Bidirectional)

Compute the column sum of the attention matrix. This is equivalent to summing the attention weights $\text{Softmax}(QK^T)$ over the query dimension.

```python
import torch
from flash_colsum import flash_colsum

device = "cuda"
dtype = torch.float16

# Shapes: (Batch, Heads, Seq_Len, Head_Dim)
Q = torch.randn(8, 16, 512, 64, device=device, dtype=dtype)
K = torch.randn(8, 16, 512, 64, device=device, dtype=dtype)

# Returns: (Batch, Key_Len) aggregated over all queries and heads
col_sum = flash_colsum(Q, K) 
print(col_sum.shape) # (8, 512)
```

### 2. Causal Attention (KV Cache)

Handle autoregressive attention where $M \neq N$ (e.g., decoding steps). The kernel applies a **right-aligned causal mask**.

```python
import torch
from flash_colsum import flash_colsum, flash_colmean

# Example: Single sequence, 32 heads, 128 new queries, 4096 existing keys
Q = torch.randn(1, 32, 128, 128, device="cuda", dtype=torch.float16)
K = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)

# Compute column sums with causal masking
col_sum = flash_colsum(Q, K, is_causal=True) # Shape: (1, 4096)

# Compute column means (automatically handles the varying denominator due to masking)
# Keys at the start are attended to by ALL queries.
# Keys at the end are attended to by FEWER queries (due to causality).
col_mean = flash_colmean(Q, K, is_causal=True) # Shape: (1, 4096)
```

## Performance

Flash-ColSum achieves significant speedups and memory savings over naïve implementations. By fusing the softmax and reduction steps, it avoids writing the huge $B \times H \times M \times N$ matrix to GPU memory.

![A6000 Benchmark Results](assets/A6000_benchmark.png)
*Benchmarked on NVIDIA RTX A6000 with FP16 precision*

![5090 Benchmark Results](assets/5090_benchmark.png)
*Benchmarked on NVIDIA GeForce RTX 5090 with FP16 precision*

## Development

### Project Structure
```
flash-colsum/
├── flash_colsum/          # Source code
│   ├── flash.py           # Triton kernels & API
│   └── naive.py           # Reference PyTorch implementations
├── benchmarks/            # Performance scripts
└── tests/                 # Correctness tests
```

### Running Tests
```bash
# Install test dependencies
pip install -e ".[test]"

# Run correctness tests
pytest -v
```

### Running Benchmarks
```bash
# Run benchmark sweeps
FLASH_COLSUM_RUN_BENCH=1 pytest tests/test_benchmarks.py -v -s
```

## Citation

If you use Flash-ColSum in your research, please cite the SparseVILA paper:

```bibtex
@inproceedings{khaki2025sparsevila,
  title = {{SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference}},
  author = {Khaki, Samir and Guo, Junxian and Tang, Jiaming and Yang, Shang and Chen, Yukang and Plataniotis, Konstantinos N and Lu, Yao and Han, Song and Liu, Zhijian},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025}
}
```

## License

[MIT License](LICENSE)

## Acknowledgments

- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)**: The tiling and online softmax approach is heavily inspired by FlashAttention.
- **[SparseVILA](https://arxiv.org/abs/2510.17777)**: The original project that necessitated this primitive.
