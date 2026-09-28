"""The scalar view-dependent raydrop SH kernel must match the gsplat reference.

``_eval_view_dependent_raydrop`` has two implementations: a dedicated CUDA
kernel (fast path) and a fallback that packs the scalar coefficients into
gsplat's colour-shaped ``(N, K, 3)`` layout. They must agree.
"""

from __future__ import annotations

import pytest
import torch

from splatsim import _lidar_cull_ext
from splatsim._conversions import SH_C0
from splatsim.lidar_renderer import _raydrop_sh_degree_from_coefs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _gsplat_reference(
    means: torch.Tensor,
    view_pos: torch.Tensor,
    dc_logit: torch.Tensor,
    raydrop_sh: torch.Tensor,
) -> torch.Tensor:
    """The (N, K, 3)-packing path, verbatim from _eval_view_dependent_raydrop."""
    import gsplat

    n = means.shape[0]
    degree = _raydrop_sh_degree_from_coefs(raydrop_sh.shape[1])
    k = (degree + 1) ** 2
    coeffs = torch.zeros((n, k, 3), dtype=torch.float32, device=means.device)
    coeffs[:, 0, 0] = dc_logit.float() / SH_C0
    coeffs[:, 1:k, 0] = raydrop_sh.float()
    dirs = means.float() - view_pos.reshape(3)
    return gsplat.spherical_harmonics(degree, dirs, coeffs)[:, 0]


@cuda
@pytest.mark.parametrize("c_high", [3, 8, 15])  # SH degree 1, 2, 3
def test_kernel_matches_gsplat_packing(c_high: int) -> None:
    ext = _lidar_cull_ext._try_load()
    if ext is None or not hasattr(ext, "raydrop_sh_eval"):
        pytest.skip("raydrop_sh_eval extension unavailable")

    g = torch.Generator(device="cuda").manual_seed(c_high)
    n = 50_000
    means = (torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 200.0
    view_pos = torch.tensor([1.5, -2.0, 0.7], device="cuda")
    dc = torch.randn(n, device="cuda", generator=g)
    sh = torch.randn(n, c_high, device="cuda", generator=g)

    ref = _gsplat_reference(means, view_pos, dc, sh)
    got = ext.raydrop_sh_eval(means, view_pos, dc, sh)

    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


@cuda
def test_zero_higher_bands_reduce_to_the_dc_logit() -> None:
    """With no higher-order energy the result is the scalar logit itself."""
    ext = _lidar_cull_ext._try_load()
    if ext is None or not hasattr(ext, "raydrop_sh_eval"):
        pytest.skip("raydrop_sh_eval extension unavailable")

    n = 1000
    means = torch.randn(n, 3, device="cuda")
    dc = torch.randn(n, device="cuda")
    sh = torch.zeros(n, 8, device="cuda")
    got = ext.raydrop_sh_eval(means, torch.zeros(3, device="cuda"), dc, sh)
    torch.testing.assert_close(got, dc, rtol=1e-5, atol=1e-6)


@cuda
def test_empty_input() -> None:
    ext = _lidar_cull_ext._try_load()
    if ext is None or not hasattr(ext, "raydrop_sh_eval"):
        pytest.skip("raydrop_sh_eval extension unavailable")

    got = ext.raydrop_sh_eval(
        torch.zeros(0, 3, device="cuda"),
        torch.zeros(3, device="cuda"),
        torch.zeros(0, device="cuda"),
        torch.zeros(0, 8, device="cuda"),
    )
    assert got.shape == (0,)
