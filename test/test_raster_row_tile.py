"""The rasterizer's one-beam-row fast path must match the general path.

With ``_SPLATAD_TILE_HEIGHT == 1`` every thread in a block shares the tile's
elevation, so the kernel folds the elevation half of the conic into a quadratic
in azimuth once per Gaussian (ROW_TILE) instead of recomputing it per pixel.
That is an algebraic rearrangement, so it must agree with the general path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import splatsim.lidar_renderer as lr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _scene(n: int = 4000, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    means = (torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 60.0
    means[:, 0] = torch.rand(n, device="cuda", generator=g) * 40.0 + 5.0
    quats = torch.zeros(n, 4, device="cuda")
    quats[:, 0] = 1.0
    return dict(
        means=means,
        quats=quats,
        scales=torch.full((n, 3), 0.2, device="cuda"),
        opacities=torch.full((n,), 0.9, device="cuda"),
        intensity_sig=torch.rand(n, device="cuda", generator=g),
        raydrop_logit=torch.full((n,), -6.0, device="cuda"),
    )


def _render(tile_h: int, tile_w: int, spec, s2w, scene):
    lr._SPLATAD_TILE_HEIGHT, lr._SPLATAD_TILE_WIDTH = tile_h, tile_w
    lr._PANO_GEOM_CACHE.clear()
    with torch.no_grad():
        return lr.render_lidar_panorama(
            sensor_to_world=s2w,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            **scene,
        )


@cuda
@pytest.mark.parametrize("tile_w", [8, 16, 32])
def test_row_tile_matches_multi_row_tiling(tile_w: int) -> None:
    """1xW (ROW_TILE) vs 2xW: same Gaussians per tile column, folded differently.

    Tiling regroups Gaussians so a handful of boundary cells can differ (the
    3-sigma bbox binning vs the ~3.7-sigma alpha cutoff); the fold itself must
    not move anything beyond that.
    """
    spec = lr.LidarSensorSpec(
        name="t", sensor_type="XT32", s2b=np.eye(4), n_columns=512
    )
    s2w = torch.eye(4, device="cuda")
    sc = _scene()
    saved = (lr._SPLATAD_TILE_HEIGHT, lr._SPLATAD_TILE_WIDTH)
    try:
        row = _render(1, tile_w, spec, s2w, sc)
        multi = _render(2, tile_w, spec, s2w, sc)
    finally:
        lr._SPLATAD_TILE_HEIGHT, lr._SPLATAD_TILE_WIDTH = saved
        lr._PANO_GEOM_CACHE.clear()

    hr, hm = row["alpha"] > 0.1, multi["alpha"] > 0.1
    iou = ((hr & hm).sum() / (hr | hm).sum().clamp(min=1)).item()
    assert iou > 0.999, f"hit-mask IoU {iou:.6f}"
    both = hr & hm
    d = (row["distance"][both] - multi["distance"][both]).abs()
    assert d.quantile(0.99).item() < 1e-3, f"distance p99 {d.quantile(0.99):.3e}"
    di = (row["intensity"][both] - multi["intensity"][both]).abs()
    assert di.quantile(0.99).item() < 1e-3, f"intensity p99 {di.quantile(0.99):.3e}"


@cuda
def test_row_tile_is_the_default_tiling() -> None:
    """The shipped tiling must actually take the ROW_TILE path."""
    assert lr._SPLATAD_TILE_HEIGHT == 1, (
        "the rasterizer's folded fast path only runs for one-beam-row tiles; "
        f"default tile height is {lr._SPLATAD_TILE_HEIGHT}"
    )
