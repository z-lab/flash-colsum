"""
Flash-ColSum Core Tests
=======================

ColSum Tests (5 configurations):
1. Non-causal square (Q_len == K_len)
2. Non-causal non-square (Q_len != K_len)
3. Causal square (Q_len == K_len, triangular mask)
4. Causal non-square (Q_len < K_len, K-cache scenario)
5. Causal batched (B > 1, non-square)

ColMean Tests (5 configurations):
1. Non-causal square (Q_len == K_len)
2. Non-causal non-square (Q_len != K_len)
3. Causal square (Q_len == K_len, triangular mask)
4. Causal non-square (Q_len < K_len, per-key normalization)
5. Causal batched (B > 1, per-key normalization)

Run with: pytest -s -v tests/test_core.py
"""

import math
import pytest
import torch

from flash_colsum import flash_colsum, flash_colmean


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def _relative_error_pct(out_ref: torch.Tensor, out_fast: torch.Tensor) -> float:
    """Compute relative error as a percentage (using max for denominator)."""
    abs_err = (out_ref - out_fast).abs().max().item()
    max_val = out_ref.abs().max().item()
    return abs_err / max_val * 100 if max_val > 0 else 0.0


# ============================================================================
# Naive Reference Implementations
# ============================================================================

def naive_colsum(query: torch.Tensor, key: torch.Tensor, scale: float = None, is_causal: bool = False) -> torch.Tensor:
    """
    Reference implementation of attention column sum (computed in FP32).
    
    Args:
        query: (B, H, Q_len, D)
        key: (B, H, K_len, D)
        scale: Softmax scale (default: 1/sqrt(D))
        is_causal: If True, applies right-aligned causal masking
        
    Returns:
        Column sums of shape (B, K_len) summed over heads
    """
    B, H, Q_len, D = query.shape
    _, _, K_len, _ = key.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Compute everything in FP32 for reference accuracy
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale  # (B, H, Q_len, K_len)
    
    if is_causal:
        # Right-aligned causal mask: q_idx + K_PAST >= k_idx
        K_PAST = max(0, K_len - Q_len)
        q_idx = torch.arange(Q_len, device=query.device).unsqueeze(1) + K_PAST
        k_idx = torch.arange(K_len, device=query.device).unsqueeze(0)
        mask = q_idx >= k_idx  # (Q_len, K_len)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax and sum in fp32, then convert to query dtype
    attn = torch.softmax(scores, dim=-1)
    return attn.sum(dim=(1, 2)).to(query.dtype)


def naive_colmean(query: torch.Tensor, key: torch.Tensor, scale: float = None, is_causal: bool = False) -> torch.Tensor:
    """
    Reference implementation of attention column mean (computed in FP32).
    
    For causal non-square (right-aligned), each key has a different number of
    attending queries, so we use per-key normalization factors.
    
    Args:
        query: (B, H, Q_len, D)
        key: (B, H, K_len, D)
        scale: Softmax scale (default: 1/sqrt(D))
        is_causal: If True, applies right-aligned causal masking
        
    Returns:
        Column means of shape (B, K_len) averaged over heads and query positions
    """
    B, H, Q_len, _ = query.shape
    _, _, K_len, _ = key.shape
    
    col_sum = naive_colsum(query, key, scale, is_causal)
    
    if is_causal and Q_len != K_len:
        # Right-aligned causal: per-key normalization
        # Keys 0 to K_PAST-1: all Q_len queries attend
        # Keys K_PAST to K_len-1: (K_len - k) queries attend
        K_PAST = K_len - Q_len
        norm_factors = torch.empty(K_len, device=query.device, dtype=col_sum.dtype)
        norm_factors[:K_PAST] = H * Q_len
        norm_factors[K_PAST:] = H * torch.arange(Q_len, 0, -1, device=query.device, dtype=col_sum.dtype)
        return col_sum / norm_factors
    else:
        norm_factor = H * Q_len
        return col_sum / norm_factor


# ============================================================================
# ColSum Tests - Non-Causal (square and non-square)
# ============================================================================

@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colsum_noncausal_square():
    """Test: Non-causal square (Q_len == K_len)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, S, D = 4, 8, 512, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    out_ref = naive_colsum(Q, K, is_causal=False)
    out_fast = flash_colsum(Q, K, is_causal=False)
    
    assert out_ref.shape == out_fast.shape == (B, S)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[Non-Causal Square] B={B}, H={H}, Q_len={S}, K_len={S}, D={D}")
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colsum_noncausal_nonsquare():
    """Test: Non-causal non-square (Q_len != K_len)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, Q_len, K_len, D = 2, 8, 256, 1024, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
    K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
    
    out_ref = naive_colsum(Q, K, is_causal=False)
    out_fast = flash_colsum(Q, K, is_causal=False)
    
    assert out_ref.shape == out_fast.shape == (B, K_len)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[Non-Causal Non-Square] B={B}, H={H}, Q_len={Q_len}, K_len={K_len}, D={D}")
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


# ============================================================================
# ColSum Tests - Causal (square and non-square)
# ============================================================================

@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colsum_causal_square():
    """Test: Causal square (Q_len == K_len, triangular mask)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, S, D = 1, 16, 1024, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    out_ref = naive_colsum(Q, K, is_causal=True)
    out_fast = flash_colsum(Q, K, is_causal=True)
    
    assert out_ref.shape == out_fast.shape == (B, S)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[Causal Square] B={B}, H={H}, Q_len={S}, K_len={S}, D={D}")
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colsum_causal_nonsquare():
    """Test: Causal non-square (Q_len < K_len, K-cache / right-aligned mask)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, Q_len, K_len, D = 1, 16, 128, 16384, 128
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
    K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
    
    out_ref = naive_colsum(Q, K, is_causal=True)
    out_fast = flash_colsum(Q, K, is_causal=True)
    
    K_PAST = K_len - Q_len
    assert out_ref.shape == out_fast.shape == (B, K_len)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[Causal Non-Square] B={B}, H={H}, Q_len={Q_len}, K_len={K_len}, D={D}")
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


# ============================================================================
# ColMean Tests - Non-Causal (square and non-square)
# ============================================================================

@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colmean_noncausal_square():
    """Test: ColMean non-causal square (Q_len == K_len)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, S, D = 4, 8, 512, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    out_ref = naive_colmean(Q, K, is_causal=False)
    out_fast = flash_colmean(Q, K, is_causal=False)
    
    assert out_ref.shape == out_fast.shape == (B, S)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[ColMean Non-Causal Square] B={B}, H={H}, Q_len={S}, K_len={S}, D={D}")
    
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colmean_noncausal_nonsquare():
    """Test: ColMean non-causal non-square (Q_len != K_len)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, Q_len, K_len, D = 4, 8, 256, 1024, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
    K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
    
    out_ref = naive_colmean(Q, K, is_causal=False)
    out_fast = flash_colmean(Q, K, is_causal=False)
    
    assert out_ref.shape == out_fast.shape == (B, K_len)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[ColMean Non-Causal Non-Square] B={B}, H={H}, Q_len={Q_len}, K_len={K_len}, D={D}")
    
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


# ============================================================================
# ColMean Tests - Causal (square and non-square)
# ============================================================================

@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colmean_causal_square():
    """Test: ColMean causal square (Q_len == K_len, triangular mask)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, S, D = 1, 16, 1024, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    out_ref = naive_colmean(Q, K, is_causal=True)
    out_fast = flash_colmean(Q, K, is_causal=True)
    
    assert out_ref.shape == out_fast.shape == (B, S)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[ColMean Causal Square] B={B}, H={H}, Q_len={S}, K_len={S}, D={D}")
    
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colmean_causal_nonsquare():
    """Test: ColMean causal non-square (Q_len < K_len, right-aligned mask with per-key normalization)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, Q_len, K_len, D = 1, 16, 128, 2048, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
    K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
    
    out_ref = naive_colmean(Q, K, is_causal=True)
    out_fast = flash_colmean(Q, K, is_causal=True)
    
    K_PAST = K_len - Q_len
    assert out_ref.shape == out_fast.shape == (B, K_len)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[ColMean Causal Non-Square] B={B}, H={H}, Q_len={Q_len}, K_len={K_len}, D={D}")
    print(f"  Per-key normalization: keys 0-{K_PAST-1} by H*Q_len={H*Q_len}, keys {K_PAST}-{K_len-1} by H*{Q_len}..H*1")
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


# ============================================================================
# Batched Causal Tests
# ============================================================================

@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colsum_causal_batched():
    """Test: Batched causal attention (B > 1)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, Q_len, K_len, D = 4, 8, 128, 1024, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
    K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
    
    out_ref = naive_colsum(Q, K, is_causal=True)
    out_fast = flash_colsum(Q, K, is_causal=True)
    
    K_PAST = K_len - Q_len
    assert out_ref.shape == out_fast.shape == (B, K_len)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[Causal Batched] B={B}, H={H}, Q_len={Q_len}, K_len={K_len}, D={D}")
    assert torch.allclose(out_ref, out_fast, atol=1e-4, rtol=1e-4), \
        f"Relative error too high: {rel_err:.4f}%"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_colmean_causal_batched():
    """Test: Batched causal colmean (B > 1, with per-key normalization)."""
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, Q_len, K_len, D = 4, 8, 128, 1024, 64
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
    K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
    
    out_ref = naive_colmean(Q, K, is_causal=True)
    out_fast = flash_colmean(Q, K, is_causal=True)
    
    K_PAST = K_len - Q_len
    assert out_ref.shape == out_fast.shape == (B, K_len)
    rel_err = _relative_error_pct(out_ref, out_fast)
    print(f"\n[ColMean Causal Batched] B={B}, H={H}, Q_len={Q_len}, K_len={K_len}, D={D}")
    assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
        f"Relative error too high: {rel_err:.4f}%"


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_error_handling_cpu_tensor():
    """Test error handling: CPU tensors should raise."""
    Q_cpu = torch.randn(1, 8, 128, 64)
    K_cpu = torch.randn(1, 8, 128, 64)
    with pytest.raises(ValueError, match="CUDA"):
        flash_colsum(Q_cpu, K_cpu)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
