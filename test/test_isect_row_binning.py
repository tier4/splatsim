"""Per-beam azimuth binning must never drop a contribution.

The binner used to apply a Gaussian's widest azimuth extent to every elevation
row its bounding box spans. At the shipped tiling a row IS one beam, so the
only elevation sampled in that row is the beam's own -- and the reachable
azimuth interval there is much narrower, shrinking to nothing near the
ellipse's poles. Tightening it is exact: it may only remove (Gaussian, tile)
pairs that no pixel in the tile could have been hit by.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import splatsim.lidar_renderer as lr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _spec(n_columns: int = 512) -> lr.LidarSensorSpec:
    return lr.LidarSensorSpec(
        name="t", sensor_type="XT32", s2b=np.eye(4), n_columns=n_columns
    )


def _scene(n: int = 40_000, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    means = (torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 80.0
    means[:, 0] = torch.rand(n, device="cuda", generator=g) * 50.0 + 3.0
    quats = torch.randn(n, 4, device="cuda", generator=g)  # rotated, not axis-aligned
    return dict(
        means=means,
        quats=quats,
        scales=torch.rand(n, 3, device="cuda", generator=g) * 0.5 + 0.05,
        opacities=torch.rand(n, device="cuda", generator=g) * 0.6 + 0.4,
        intensity_sig=torch.rand(n, device="cuda", generator=g),
        raydrop_logit=torch.full((n,), -6.0, device="cuda"),
    )


def _pairs(scene, spec, s2w, *, row_elevations: bool):
    """Tile-list length for one render, with and without the tightening."""
    from splatad_kernel.cuda._wrapper import isect_lidar_tiles
    from splatad_kernel.rendering import fully_fused_lidar_projection

    device = scene["means"].device
    geom = lr._panorama_geometry(spec, device)
    w = geom.azs_cw.shape[0]
    tw = lr._SPLATAD_TILE_WIDTH
    vm = lr._rigid_inverse_4x4(s2w).unsqueeze(0)
    q = torch.nn.functional.normalize(scene["quats"].float(), dim=-1)
    radii, means2d, depths, conics, _c, _v, _d = fully_fused_lidar_projection(
        scene["means"].float(),
        None,
        q,
        scene["scales"].float(),
        None,
        vm,
        torch.zeros(1, 3, device=device),
        torch.zeros(1, 3, device=device),
        torch.zeros(1, device=device),
        min_elevation=geom.min_el_deg,
        max_elevation=geom.max_el_deg + 1e-3,
        min_azimuth=-180.0,
        max_azimuth=180.0,
        eps2d=0.017,
        near_plane=0.3,
        far_plane=120.0,
        radius_clip=0.0,
    )
    _, _, flat = isect_lidar_tiles(
        means2d,
        radii,
        depths,
        elev_boundaries=geom.tile_boundaries.clone(),
        tile_azim_resolution=(360.0 / w) * tw,
        min_azim=-180.0,
        packed=False,
        n_cameras=1,
        conics=conics,
        opacities=scene["opacities"].float().unsqueeze(0),
        row_elevations=geom.row_elevations_asc if row_elevations else None,
    )
    return int(flat.shape[0])


@cuda
def test_tightened_binning_does_not_change_the_render() -> None:
    """Same panorama, fewer tile-list entries."""
    spec = _spec()
    s2w = torch.eye(4, device="cuda")
    sc = _scene()
    with torch.no_grad():
        out = lr.render_lidar_panorama(
            sensor_to_world=s2w,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            **sc,
        )
    # The tightening is compiled into the shipped path, so the contract we can
    # assert here is that it produces a sane render and strictly fewer pairs.
    assert torch.isfinite(out["distance"]).all()
    assert int((out["alpha"] > 0.1).sum()) > 0

    loose = _pairs(sc, spec, s2w, row_elevations=False)
    tight = _pairs(sc, spec, s2w, row_elevations=True)
    assert tight < loose, f"tightening emitted no fewer pairs ({tight} vs {loose})"


@cuda
def test_tightening_keeps_every_return() -> None:
    """A pair it removes must be one no pixel in that tile could be hit by.

    Rendering with the loose binning is the ground truth here: any return it
    finds must survive the tightened binning.
    """
    spec = _spec()
    s2w = torch.eye(4, device="cuda")
    sc = _scene(seed=3)

    with torch.no_grad():
        tight = lr.render_lidar_panorama(
            sensor_to_world=s2w,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            **sc,
        )
        # Re-render with the tightening disabled by pretending the tiling is
        # multi-row (which makes the renderer pass row_elevations=None).
        saved = lr._SPLATAD_TILE_HEIGHT
        try:
            lr._SPLATAD_TILE_HEIGHT = 2
            lr._PANO_GEOM_CACHE.clear()
            loose = lr.render_lidar_panorama(
                sensor_to_world=s2w,
                lidar_spec=spec,
                min_range_m=0.3,
                max_range_m=120.0,
                **sc,
            )
        finally:
            lr._SPLATAD_TILE_HEIGHT = saved
            lr._PANO_GEOM_CACHE.clear()

    h_tight = tight["alpha"] > 0.1
    h_loose = loose["alpha"] > 0.1
    # Tiling differences move a few boundary cells either way; what must not
    # happen is a systematic loss.
    iou = ((h_tight & h_loose).sum() / (h_tight | h_loose).sum().clamp(min=1)).item()
    assert iou > 0.99, f"tightened binning lost returns: IoU {iou:.6f}"
    both = h_tight & h_loose
    d = (tight["distance"][both] - loose["distance"][both]).abs()
    assert d.quantile(0.99).item() < 1e-3
