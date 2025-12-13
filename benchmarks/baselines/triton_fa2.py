"""
Triton Flash Attention v2 Baseline
==================================

A simplified Flash Attention v2 kernel for benchmarking.
Supports right-aligned causal masking for Q_len != K_len.

Adapted from: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    K_LEN: tl.constexpr, K_PAST: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M + K_PAST
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M + K_PAST, (start_m + 1) * BLOCK_M + K_PAST
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, K_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = (offs_m[:, None] + K_PAST) >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vn, stride_vk,
              stride_oz, stride_oh, stride_om, stride_ok,
              Z, H, Q_LEN, K_LEN, K_PAST,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    q = tl.load(Q_block_ptr)
    
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, K_LEN, K_PAST)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, K_LEN, K_PAST)
    
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_LEN + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(tl.float16))


def triton_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Triton Flash Attention v2 with right-aligned causal mask support.
    
    Args:
        query: (B, H, Q_LEN, D)
        key: (B, H, K_LEN, D)
        value: (B, H, K_LEN, D)
        is_causal: Apply right-aligned causal masking
        scale: Attention scale (default: 1/sqrt(D))
    
    Returns:
        Output: (B, H, Q_LEN, D)
    """
    import math
    
    B, H, Q_LEN, HEAD_DIM = query.shape
    _, _, K_LEN, _ = key.shape
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    assert key.shape == value.shape, "K and V must have same shape"
    
    if scale is None:
        scale = 1.0 / math.sqrt(HEAD_DIM)
    
    K_PAST = max(0, K_LEN - Q_LEN) if is_causal else 0
    
    o = torch.empty((B, H, Q_LEN, HEAD_DIM), device=query.device, dtype=query.dtype)
    M = torch.empty((B * H, Q_LEN), device=query.device, dtype=torch.float32)
    
    stage = 3 if is_causal else 1
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)
    
    _attn_fwd[grid](
        query, key, value, scale, M, o,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, Q_LEN, K_LEN, K_PAST,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=stage,
        num_warps=4,
        num_stages=2,
    )
    
    return o


# Alias for backwards compatibility
triton_attention = triton_flash_attention

__all__ = ['triton_flash_attention', 'triton_attention']

