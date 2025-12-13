import math

import torch

__all__ = ["naive_colsum", "naive_colmean"]


def naive_colsum(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    m, n = query.shape[2], key.shape[2]
    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])
    weight = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if is_causal:
        q = torch.arange(m, device=query.device).view(1, 1, -1, 1)
        k = torch.arange(n, device=query.device).view(1, 1, 1, -1)
        weight = weight.masked_fill(q + max(n - m, 0) < k, -float("inf"))
    weight = torch.softmax(weight, dim=-1)
    return weight.sum(dim=(1, 2)).to(query.dtype)


def naive_colmean(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    m, n = query.shape[2], key.shape[2]
    weight = naive_colsum(query, key, is_causal=is_causal, scale=scale)
    if is_causal:
        c = max(n - m, 0)
        weight[:, :c] /= m
        weight[:, c:] /= torch.arange(n - c, 0, -1, device=query.device)
    else:
        weight /= m
    return weight / query.shape[1]
