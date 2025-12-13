import pytest
import torch
from typing import Callable

from flash_colsum import flash_colsum, flash_colmean, naive_colsum, naive_colmean


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for these tests")
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("d", [16, 32, 64])
@pytest.mark.parametrize("m,n", [(128, 128), (131, 131), (128, 1024), (131, 1027)])
@pytest.mark.parametrize("h", [8, 13])
@pytest.mark.parametrize("b", [1, 4, 7])
@pytest.mark.parametrize("flash_impl,naive_impl", [(flash_colsum, naive_colsum), (flash_colmean, naive_colmean)])
def test_correctness(
    flash_impl: Callable,
    naive_impl: Callable,
    b: int,
    h: int,
    m: int,
    n: int,
    d: int,
    is_causal: bool,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    query = torch.randn(b, h, m, d, device="cuda", dtype=torch.float16)
    key = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    output = flash_impl(query, key, is_causal=is_causal)
    target = naive_impl(query, key, is_causal=is_causal)

    if target.shape != output.shape:
        raise AssertionError(
            "Shape mismatch:\n"
            + f"  output.shape = {tuple(output.shape)}\n"
            + f"  target.shape = {tuple(target.shape)}"
        )

    if not torch.allclose(target, output, atol=1e-3, rtol=1e-3):
        error = (output - target).abs()
        raise AssertionError(
            "Value mismatch:\n"
            + f"  max_abs_error = {error.max().item():.5g}\n"
            + f"  max_rel_error = {(error / target.abs().clip(min=1e-6)).max().item():.5g}"
        )
