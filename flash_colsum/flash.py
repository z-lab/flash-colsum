import math

import torch
import triton
import triton.language as tl

__all__ = ["flash_colsum", "flash_colmean"]


@triton.jit
def _mask_scores(
    scores,
    indices_m,
    indices_n,
    M,
    N,
    APPLY_CAUSAL: tl.constexpr,
    APPLY_ROW_BOUND: tl.constexpr,
    APPLY_COL_BOUND: tl.constexpr,
):
    if APPLY_CAUSAL:
        scores = tl.where(
            indices_m[:, None] + max(N - M, 0) >= indices_n[None, :],
            scores,
            -float("inf"),
        )
    if APPLY_ROW_BOUND:
        scores = tl.where(indices_m[:, None] < M, scores, -float("inf"))
    if APPLY_COL_BOUND:
        scores = tl.where(indices_n[None, :] < N, scores, -float("inf"))
    return scores


@triton.jit
def _online_softmax_lse_update(
    l_running,
    m_running,
    q_block,
    k_block_ptr,
    scale,
    indices_m,
    n_min,
    n_max,
    M,
    N,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL: tl.constexpr,
):
    k_block_ptr = tl.advance(k_block_ptr, (0, n_min))

    for start_n in range(n_min, n_max, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        indices_n = start_n + tl.arange(0, BLOCK_N)

        k_block = tl.load(k_block_ptr, boundary_check=(1,), padding_option="zero")
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

        scores = tl.dot(q_block, k_block) * scale
        scores = _mask_scores(
            scores,
            indices_m,
            indices_n,
            M,
            N,
            APPLY_CAUSAL=APPLY_CAUSAL,
            APPLY_ROW_BOUND=False,
            APPLY_COL_BOUND=True,
        )

        m_updated = tl.maximum(m_running, tl.max(scores, axis=1))
        scores -= m_updated[:, None]

        p = tl.exp2(scores)
        alpha = tl.exp2(m_running - m_updated)

        l_running = l_running * alpha + tl.sum(p, axis=1)
        m_running = m_updated

    return l_running, m_running


@triton.jit
def _accumulate_softmax_colsum(
    q_block,
    k_block_ptr,
    o_ptr,
    scale,
    lse,
    indices_m,
    n_min,
    n_max,
    M,
    N,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL: tl.constexpr,
):
    k_block_ptr = tl.advance(k_block_ptr, (0, n_min))

    for start_n in range(n_min, n_max, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        indices_n = start_n + tl.arange(0, BLOCK_N)

        k_block = tl.load(k_block_ptr, boundary_check=(1,), padding_option="zero")
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

        scores = tl.dot(q_block, k_block) * scale
        scores = _mask_scores(
            scores,
            indices_m,
            indices_n,
            M,
            N,
            APPLY_CAUSAL=APPLY_CAUSAL,
            APPLY_ROW_BOUND=True,
            APPLY_COL_BOUND=True,
        )

        probs = tl.exp2(scores - lse[:, None])
        tl.atomic_add(o_ptr + indices_n, tl.sum(probs, axis=0), mask=indices_n < N)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M in [64]  # [64, 128]
        for BLOCK_N in [64]  # [32, 64, 128]
        for num_stages in [3]  # [2, 3, 4]
        for num_warps in [4]  # [4, 8]
    ],
    key=["D", "CAUSAL"],
    reset_to_zero=["o_ptr"],
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

    # Offset pointers to (b, h)
    q_ptr += b.to(tl.int64) * stride_qb + h.to(tl.int64) * stride_qh
    k_ptr += b.to(tl.int64) * stride_kb + h.to(tl.int64) * stride_kh
    o_ptr += bh.to(tl.int64) * stride_obh

    # View Q as (M, D) and K as (D, N)
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

    q_block = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")
    indices_m = start_m + tl.arange(0, BLOCK_M)

    # Split [0, N) into:
    #   [0, mid): always allowed (no causal mask needed)
    #   [mid, end): causal band (mask needed)
    if CAUSAL:
        offset = max(N - M, 0)
        mid = ((start_m + offset) // BLOCK_N) * BLOCK_N
        end = tl.cdiv(start_m + offset + BLOCK_M, BLOCK_N) * BLOCK_N
    else:
        mid = N
        end = N

    # Online softmax stats per query row
    m_running = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_running = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 1) Compute log-sum-exp for each query row
    l_running, m_running = _online_softmax_lse_update(
        l_running,
        m_running,
        q_block,
        k_block_ptr,
        scale,
        indices_m,
        0,
        mid,
        M,
        N,
        BLOCK_N,
        APPLY_CAUSAL=False,
    )
    l_running, m_running = _online_softmax_lse_update(
        l_running,
        m_running,
        q_block,
        k_block_ptr,
        scale,
        indices_m,
        mid,
        end,
        M,
        N,
        BLOCK_N,
        APPLY_CAUSAL=True,
    )
    lse = m_running + tl.log2(l_running)

    # 2) Accumulate column sums (atomic across M-tiles)
    _accumulate_softmax_colsum(
        q_block,
        k_block_ptr,
        o_ptr,
        scale,
        lse,
        indices_m,
        0,
        mid,
        M,
        N,
        BLOCK_N,
        APPLY_CAUSAL=False,
    )
    _accumulate_softmax_colsum(
        q_block,
        k_block_ptr,
        o_ptr,
        scale,
        lse,
        indices_m,
        mid,
        end,
        M,
        N,
        BLOCK_N,
        APPLY_CAUSAL=True,
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
