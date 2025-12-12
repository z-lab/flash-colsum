"""
Flash-ColSum: Efficient attention column-sum operations with Triton kernels.
"""

from .ops import flash_colsum, flash_colmean

__all__ = ["flash_colsum", "flash_colmean"]

__version__ = "1.0.0"
