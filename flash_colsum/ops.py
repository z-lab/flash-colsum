"""
Flash-ColSum Operations
=======================

High-level API for Flash-ColSum operations.
"""

from .kernel_unified import flash_colsum as _flash_colsum_kernel
import torch


def flash_colsum(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Compute column sums of the softmax attention matrix using Flash Attention.
    
    This is memory-efficient and does not materialize the full attention matrix.
    
    Args:
        query: Query tensor of shape (B, H, Q_len, D)
        key: Key tensor of shape (B, H, K_len, D)
        scale: Attention scale factor. Default: 1/sqrt(D)
        is_causal: If True, applies right-aligned causal masking
        
    Returns:
        Column sums of shape (B, K_len)
    """
    # Validate inputs
    if not query.is_cuda or not key.is_cuda:
        raise ValueError("Query and Key tensors must be on CUDA device")
    
    B, H, Q_len, D = query.shape
    B_k, H_k, K_len, D_k = key.shape
    
    if B != B_k or H != H_k or D != D_k:
        raise ValueError(f"Query and Key must have same batch, heads, and dim. "
                        f"Got Q: {query.shape}, K: {key.shape}")
    
    return _flash_colsum_kernel(query, key, causal=is_causal, sm_scale=scale)


def flash_colmean(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Compute column means of the softmax attention matrix using Flash Attention.
    
    This is memory-efficient and does not materialize the full attention matrix.
    
    For causal attention with right-aligned masking, each key position has a 
    different number of attending queries (staircase pattern). This function
    automatically computes the correct per-key normalization factor:
    - Keys 0 to K_PAST-1: normalized by H * Q_len (all queries attend)
    - Keys K_PAST to K_len-1: normalized by H * (K_len - k) (decreasing)
    
    Args:
        query: Query tensor of shape (B, H, Q_len, D)
        key: Key tensor of shape (B, H, K_len, D)
        scale: Attention scale factor. Default: 1/sqrt(D)
        is_causal: If True, applies right-aligned causal masking
        
    Returns:
        Column means of shape (B, K_len)
    """
    B, H, Q_len, _ = query.shape
    _, _, K_len, _ = key.shape
    
    col_sum = flash_colsum(query, key, scale, is_causal)
    
    if is_causal and Q_len != K_len:
        # Right-aligned causal: per-key normalization factors
        # Keys 0 to K_PAST-1: all Q_len queries attend
        # Keys K_PAST to K_len-1: (K_len - k) queries attend (decreasing: Q_len, Q_len-1, ..., 1)
        K_PAST = K_len - Q_len
        norm_factors = torch.empty(K_len, device=query.device, dtype=col_sum.dtype)
        norm_factors[:K_PAST] = H * Q_len
        norm_factors[K_PAST:] = H * torch.arange(Q_len, 0, -1, device=query.device, dtype=col_sum.dtype)
        return col_sum / norm_factors
    else:
        # Non-causal or square causal: uniform normalization
        norm_factor = H * Q_len
        return col_sum / norm_factor


__all__ = ['flash_colsum', 'flash_colmean']


