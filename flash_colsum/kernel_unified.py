"""
Flash-ColSum: Efficient attention column-sum operations with Triton kernels.
==========================
We compute: column_sum(softmax(QK^T * scale))
Adapted from: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
Credits: OpenAI kernel team, Tri Dao (Flash Attention v2)
"""

import torch
import triton
import triton.language as tl

DEVICE = 'cuda'

@triton.jit
def _attn_fwd_inner(l_i, m_i, q,  #
                    K_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    K_LEN: tl.constexpr, K_PAST: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M + K_PAST
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M + K_PAST, (start_m + 1) * BLOCK_M + K_PAST
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, K_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    
    # loop over k and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        
        qk = tl.dot(q, k)
        if STAGE == 2:
            # Right-aligned causal mask: (q_idx + K_PAST) >= k_idx
            mask = (offs_m[:, None] + K_PAST) >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
            
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # update m_i
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return l_i, m_i


@triton.jit
def _attn_fwd(Q, K, sm_scale, M, ColSum,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_csz, stride_css,  #
              Z, H, Q_LEN, K_LEN, K_PAST,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qk_offset,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        l_i, m_i = _attn_fwd_inner(l_i, m_i, q, K_block_ptr,  #
                                   start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                   4 - STAGE, offs_m, offs_n, K_LEN, K_PAST)
    # stage 2: on-band
    if STAGE & 2:
        l_i, m_i = _attn_fwd_inner(l_i, m_i, q, K_block_ptr,  #
                                   start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                   2, offs_m, offs_n, K_LEN, K_PAST)
    # === COLUMN SUM: recompute normalized attention and accumulate ===
    # Compute log-sum-exp for better numerical precision
    lse = m_i + tl.math.log2(l_i)  # log2(sum(exp2(qk)))
    
    # Atomic add: all Q-blocks accumulate into same (B*H, K_LEN) buffer
    col_sum_base = ColSum + off_hz * stride_csz
    
    # Determine range based on causal/non-causal
    if STAGE == 1:
        lo, hi = 0, K_LEN
    else:
        lo, hi = 0, (start_m + 1) * BLOCK_M + K_PAST
    
    # Reset K block pointer for pass 2 (incremental advance like pass 1)
    K_block_ptr2 = tl.make_block_ptr(
        base=K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, lo),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr2)
        qk = tl.dot(q, k) * qk_scale
        if STAGE != 1:
            # Right-aligned causal mask
            mask = (offs_m[:, None] + K_PAST) >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, -1.0e6)
        # Use lse directly: softmax = exp2(qk - lse)
        p = tl.math.exp2(qk - lse[:, None])
        # Column sum - atomic add to shared buffer (memory efficient)
        col_sum_block = tl.sum(p, axis=0)
        col_ptrs = col_sum_base + (start_n + offs_n)
        tl.atomic_add(col_ptrs, col_sum_block, mask=(start_n + offs_n) < K_LEN)
        # Advance incrementally (more efficient than recomputing from scratch)
        K_block_ptr2 = tl.advance(K_block_ptr2, (0, BLOCK_N))
    
    # epilogue - store M
    m_ptrs = M + off_hz * Q_LEN + offs_m
    tl.store(m_ptrs, lse, mask=offs_m < Q_LEN)


def flash_colsum(q, k, causal=False, sm_scale=None):
    """
    Compute column sums of softmax attention matrix.
    
    Args:
        q: Query tensor (batch, heads, q_len, head_dim)
        k: Key tensor (batch, heads, k_len, head_dim)
        causal: If True, applies right-aligned causal masking
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        
    Returns:
        Column sums (batch, k_len) summed over heads
    """
    import math
    B, H, Q_LEN, HEAD_DIM = q.shape
    _, _, K_LEN, _ = k.shape
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # K_PAST for right-aligned causal mask
    K_PAST = max(0, K_LEN - Q_LEN) if causal else 0
    
    BLOCK_M = 64
    BLOCK_N = 64
    num_q_blocks = triton.cdiv(Q_LEN, BLOCK_M)
    
    # Memory-efficient: all Q-blocks atomic_add into shared (B*H, K_LEN) buffer
    col_sum = torch.zeros((B * H, K_LEN), device=q.device, dtype=torch.float32)
    M = torch.empty((B * H, Q_LEN), device=q.device, dtype=torch.float32)
    
    stage = 3 if causal else 1
    grid = lambda args: (num_q_blocks, B * H)
    
    # col_sum: (B*H, K_LEN), stride_csz=K_LEN, stride_css unused (set to 0)
    _attn_fwd[grid](
        q, k, sm_scale, M, col_sum,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        K_LEN, 0,  # stride_csz=K_LEN, stride_css unused
        B, H, Q_LEN, K_LEN, K_PAST,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=stage,
        num_warps=4,
        num_stages=3,
    )
    
    # Sum across heads only (Q-blocks already accumulated via atomic_add)
    return col_sum.view(B, H, K_LEN).sum(dim=1).to(q.dtype)


if __name__ == "__main__":
    import math
    
    B, H, S, D = 1, 32, 1024, 64
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)
    
    # Test non-causal
    col_sum = flash_colsum(q, k, causal=False, sm_scale=sm_scale)
    # Compute Ref in FP32
    ref = torch.softmax((torch.matmul(q.float(), k.float().transpose(-2,-1)) * sm_scale).float(), dim=-1).sum(dim=(1,2))
    print(f"Non-causal error percentage: {(col_sum - ref.half()).abs().max() / ref.abs().max():.6f}")
    
    # Test causal  
    col_sum_c = flash_colsum(q, k, causal=True, sm_scale=sm_scale)
    mask = torch.tril(torch.ones(S,S,device=DEVICE,dtype=torch.bool))
    scores = (torch.matmul(q.float(), k.float().transpose(-2,-1)) * sm_scale).float().masked_fill(~mask, float("-inf"))
    ref_c = torch.softmax(scores, dim=-1).sum(dim=(1,2))
    print(f"Causal error percentage: {(col_sum_c - ref_c.half()).abs().max() / ref_c.abs().max():.6f}")
    
    # Test Causal with K-cache shifted (Right-aligned causal mask)
    k = torch.randn(B, H, S+1024, D, device=DEVICE, dtype=torch.float16)
    col_sum_c_shifted = flash_colsum(q, k, causal=True, sm_scale=sm_scale)
    K_PAST = max(0, k.shape[2] - q.shape[2])
    q_idx = torch.arange(q.shape[2], device=q.device).unsqueeze(1) + K_PAST
    k_idx = torch.arange(k.shape[2], device=k.device).unsqueeze(0)
    mask = q_idx >= k_idx
    scores = (torch.matmul(q.float(), k.float().transpose(-2,-1)) * sm_scale).float()
    scores = scores.masked_fill(~mask, float("-inf"))
    ref_c_shifted = torch.softmax(scores, dim=-1).sum(dim=(1,2))
    print(f"Causal error percentage with K-cache shifted: {(col_sum_c_shifted - ref_c_shifted.half()).abs().max() / ref_c_shifted.abs().max():.6f}")
