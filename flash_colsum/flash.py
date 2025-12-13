import math

import torch
import triton
import triton.language as tl

__all__ = ["flash_colsum", "flash_colmean"]


INF = tl.constexpr(1e6)


@triton.jit
def _attn_qk_softmax_lse_update(
    l_i,
    m_i,
    q,
    k_block_ptr,
    start_m,
    scale,
    offsets_m,
    lo,
    hi,
    M,
    N,
    BLOCK_N: tl.constexpr,
    ON_BAND: tl.constexpr,
):
    C = max(0, N - M)
    k_block_ptr = tl.advance(k_block_ptr, (0, lo))

    indices_m = start_m + offsets_m

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        indices_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(k_block_ptr, boundary_check=(1,), padding_option="zero")
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

        qk = tl.dot(q, k) * scale
        if ON_BAND:
            qk = tl.where(indices_m[:, None] + C >= indices_n[None, :], qk, -INF)
        qk = tl.where(indices_n[None, :] < N, qk, -INF)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk -= m_ij[:, None]
        p = tl.exp2(qk)
        alpha = tl.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return l_i, m_i


@triton.jit
def _attn_colsum_accumulate(
    q,
    k_block_ptr,
    o_ptr,
    scale,
    lse,
    start_m,
    offsets_m,
    lo,
    hi,
    M,
    N,
    BLOCK_N: tl.constexpr,
    ON_BAND: tl.constexpr,
):
    C = max(0, N - M)
    k_block_ptr = tl.advance(k_block_ptr, (0, lo))

    indices_m = start_m + offsets_m
    offsets_n = tl.arange(0, BLOCK_N)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        indices_n = start_n + offsets_n

        k = tl.load(k_block_ptr, boundary_check=(1,), padding_option="zero")
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

        qk = tl.dot(q, k) * scale
        if ON_BAND:
            qk = tl.where(indices_m[:, None] + C >= indices_n[None, :], qk, -INF)
        qk = tl.where(indices_n[None, :] < N, qk, -INF)
        qk = tl.where(indices_m[:, None] < M, qk, -INF)

        p = tl.exp2(qk - lse[:, None])
        o = tl.sum(p, axis=0)
        tl.atomic_add(o_ptr + indices_n, o, mask=indices_n < N)


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3)],
    key=["D", "CAUSAL"],
)
@triton.jit
def _flash_colsum_kernel(
    q_ptr,
    k_ptr,
    o_ptr,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_obh,
    H,
    M,
    N,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    bh = tl.program_id(1)
    b, h = bh // H, bh % H

    # Offset pointers by batch and head indices
    q_ptr += b.to(tl.int64) * stride_qb + h.to(tl.int64) * stride_qh
    k_ptr += b.to(tl.int64) * stride_kb + h.to(tl.int64) * stride_kh
    o_ptr += bh.to(tl.int64) * stride_obh

    # Initialize query and key block pointers
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(M, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_N),
        order=(0, 1),
    )

    # Load the query block (keep in SRAM)
    q = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")

    # Initialize offsets for query and key blocks
    offsets_m = tl.arange(0, BLOCK_M)

    # Running softmax stats
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    if CAUSAL:
        C = max(0, N - M)
        band_start = ((start_m + C) // BLOCK_N) * BLOCK_N
        band_end = tl.minimum(tl.cdiv(start_m + C + BLOCK_M, BLOCK_N) * BLOCK_N, N)

    l_i, m_i = _attn_qk_softmax_lse_update(
        l_i,
        m_i,
        q,
        k_block_ptr,
        start_m,
        scale,
        offsets_m,
        lo=0,
        hi=band_start if CAUSAL else N,
        M=M,
        N=N,
        BLOCK_N=BLOCK_N,
        ON_BAND=False,
    )
    if CAUSAL:
        l_i, m_i = _attn_qk_softmax_lse_update(
            l_i,
            m_i,
            q,
            k_block_ptr,
            start_m,
            scale,
            offsets_m,
            lo=band_start,
            hi=band_end,
            M=M,
            N=N,
            BLOCK_N=BLOCK_N,
            ON_BAND=True,
        )

    lse = m_i + tl.log2(l_i)

    _attn_colsum_accumulate(
        q,
        k_block_ptr,
        o_ptr,
        scale,
        lse,
        start_m,
        offsets_m,
        lo=0,
        hi=band_start if CAUSAL else N,
        M=M,
        N=N,
        BLOCK_N=BLOCK_N,
        ON_BAND=False,
    )
    if CAUSAL:
        _attn_colsum_accumulate(
            q,
            k_block_ptr,
            o_ptr,
            scale,
            lse,
            start_m,
            offsets_m,
            lo=band_start,
            hi=band_end,
            M=M,
            N=N,
            BLOCK_N=BLOCK_N,
            ON_BAND=True,
        )


def _flash_colsum(
    q: torch.Tensor,
    k: torch.Tensor,
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    b, h, m, d = q.shape
    _, _, n, _ = k.shape
    o = torch.zeros((b * h, n), device=q.device, dtype=torch.float32)
    _flash_colsum_kernel[lambda config: (triton.cdiv(m, config["BLOCK_M"]), b * h)](
        q,
        k,
        o,
        scale / math.log(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        o.stride(0),
        h,
        m,
        n,
        d,
        is_causal,
    )
    return o.view(b, h, n).sum(dim=1).to(q.dtype)


def flash_colsum(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Compute per-key column sums of the softmax attention matrix using a
    FlashAttention-style fused kernel.

    Given queries Q ∈ R[B, H, M, D] and keys K ∈ R[B, H, N, D], this function
    computes, without materializing the full attention matrix:

        S[b, k] = ∑_{h=1..H} ∑_{m=1..M} softmax(QKᵀ · scale)[b, h, m, k]

    The computation uses an online softmax formulation and is memory-efficient,
    avoiding explicit construction of the QKᵀ matrix.

    If `is_causal=True`, a **right-aligned causal mask** is applied: when M ≠ N,
    query index `m` attends to keys up to index `m + (N - M)`, matching the
    behavior of KV-cached autoregressive attention.

    Args:
        query: Query tensor of shape (B, H, M, D).
        key: Key tensor of shape (B, H, N, D).
        is_causal: If True, apply right-aligned causal masking.
        scale: Optional attention scale factor. Defaults to 1 / sqrt(D).

    Returns:
        A tensor of shape (B, N) containing the summed attention weights for each
        key position, aggregated over all queries and heads.
    """
    if not query.is_cuda or not key.is_cuda:
        raise ValueError("Query and key tensors must be on CUDA device.")
    if not all(query.shape[k] == key.shape[k] for k in [0, 1, 3]):
        raise ValueError(
            f"Query and key tensors must have same batch size, number of heads, and head dimension. "
            f"Got query shape: {query.shape}, key shape: {key.shape}."
        )
    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])
    return _flash_colsum(query, key, is_causal=is_causal, scale=scale)


def flash_colmean(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Compute per-key column means of the softmax attention matrix using a
    FlashAttention-style fused kernel.

    This function returns the average attention weight assigned to each key
    position, aggregated over all queries and heads, without materializing
    the full attention matrix.

    Internally, this is computed as the column sum of the attention matrix
    (see `flash_colsum`) followed by a per-key normalization.

    If `is_causal=True`, a **right-aligned causal mask** is assumed. When the
    query and key lengths differ (M ≠ N), different key positions are attended
    by different numbers of queries:
        - Keys [0, N - M): attended by all M queries
        - Keys [N - M, N): attended by N - k queries (decreasing from M to 1)

    The function automatically applies the correct per-key normalization in
    this case.

    Args:
        query: Query tensor of shape (B, H, M, D).
        key: Key tensor of shape (B, H, N, D).
        is_causal: If True, apply right-aligned causal masking.
        scale: Optional attention scale factor. Defaults to 1 / sqrt(D).

    Returns:
        A tensor of shape (B, N) containing the mean attention weight for each
        key position, averaged over all queries and heads.
    """
    m, n = query.shape[2], key.shape[2]
    weight = flash_colsum(query, key, is_causal=is_causal, scale=scale)
    if is_causal:
        c = max(n - m, 0)
        weight[:, :c] /= m
        weight[:, c:] /= torch.arange(n - c, 0, -1, device=query.device)
    else:
        weight /= m
    return weight / query.shape[1]
