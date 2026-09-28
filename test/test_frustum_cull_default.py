"""The pre-rasterizer frustum cull must stay off by default, and stay a subset.

The projection kernel already rejects Gaussians on their exact projected
extent, so this Python-side pass only exists to shrink the arrays. Since the
LOD gather started dropping whole out-of-range octree cells it removes little,
its gathers cost more than they save, and its linearized elevation band can
discard Gaussians the rasterizer would have kept -- so it defaults to off.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import splatsim.lidar_renderer as lr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _spec() -> lr.LidarSensorSpec:
    return lr.LidarSensorSpec(
        name="t", sensor_type="XT32", s2b=np.eye(4), n_columns=512
    )


def _render(*, frustum_cull: bool, seed: int, n: int = 30_000):
    g = torch.Generator(device="cuda").manual_seed(seed)
    means = (torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 160.0
    quats = torch.zeros(n, 4, device="cuda")
    quats[:, 0] = 1.0
    with torch.no_grad():
        return lr.render_lidar_panorama(
            means=means,
            quats=quats,
            scales=torch.full((n, 3), 0.3, device="cuda"),
            opacities=torch.full((n,), 0.9, device="cuda"),
            intensity_sig=torch.rand(n, device="cuda", generator=g),
            raydrop_logit=torch.full((n,), -6.0, device="cuda"),
            sensor_to_world=torch.eye(4, device="cuda"),
            lidar_spec=_spec(),
            min_range_m=0.3,
            max_range_m=120.0,
            frustum_cull=frustum_cull,
        )


def test_cull_is_off_by_default() -> None:
    r = lr.LidarRenderer(_spec(), device="cpu")
    assert r.frustum_cull is False


@cuda
def test_cull_only_ever_removes_returns_never_adds() -> None:
    """Culling approximates the projection's exact test.

    It may only LOSE returns relative to not culling; a cell that culling turns
    INTO a return would mean it lets through something the rasterizer would
    otherwise never see.
    """
    full = _render(frustum_cull=False, seed=0)
    cull = _render(frustum_cull=True, seed=0)
    h_full = full["alpha"] > 0.1
    h_cull = cull["alpha"] > 0.1
    invented = int((h_cull & ~h_full).sum())
    assert invented == 0, (
        f"culling produced {invented} returns the unculled render does not have"
    )


@cuda
def test_cull_keeps_the_bulk_of_returns() -> None:
    """Sanity bound: the cull is an approximation, not a different renderer."""
    full = _render(frustum_cull=False, seed=1)
    cull = _render(frustum_cull=True, seed=1)
    hf = full["alpha"] > 0.1
    hc = cull["alpha"] > 0.1
    iou = ((hf & hc).sum() / (hf | hc).sum().clamp(min=1)).item()
    assert iou > 0.99, f"cull changed the render too much: IoU {iou:.6f}"
