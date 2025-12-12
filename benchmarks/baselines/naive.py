"""
Naive PyTorch Baselines
=======================

Reference implementations using standard PyTorch operations.
These materialize the full attention matrix and are used for correctness verification.
"""

import torch
from typing import Optional


def naive_colsum(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Naive column sum of attention matrix (materializes full attention).
    
    Args:
        query: (B, H, S_q, D)
        key: (B, H, S_k, D)
        scale: Attention scale (default: 1/sqrt(D))
        is_causal: Whether to apply right-aligned causal masking
    
    Returns:
        Column sums: (B, S_k)
    """
    B, H, S_q, D = query.shape
    _, _, S_k, _ = key.shape
    
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    
    # FP32 matmul to match Triton's tl.dot precision
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    
    if is_causal:
        K_PAST = max(0, S_k - S_q)
        q_idx = torch.arange(S_q, device=query.device).unsqueeze(1) + K_PAST
        k_idx = torch.arange(S_k, device=query.device).unsqueeze(0)
        mask = q_idx >= k_idx
        scores = scores.masked_fill(~mask, float('-inf'))
    
    attn = torch.softmax(scores, dim=-1)
    return attn.sum(dim=(1, 2)).to(query.dtype)


def naive_colmean(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Naive column mean of attention matrix.
    """
    norm_factor = query.shape[1] * query.shape[2]
    return naive_colsum(query, key, scale, is_causal) / norm_factor


__all__ = ['naive_colsum', 'naive_colmean']

