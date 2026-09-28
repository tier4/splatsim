"""Motion-during-sweep (``sensor_to_world_end``) LiDAR rendering.

Exercises the ``sensor_to_world_end`` path of :func:`render_lidar_panorama`.
This is real rolling shutter: the panorama is rendered from the MID-sweep pose
and each column is displaced by its own scan time times the sensor's motion.
The correctness of that displacement -- sign, frame and units -- is checked in
test_rolling_shutter.py against a swept reference; these tests cover the shape
of the interface around it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from splatsim.lidar_renderer import LidarSensorSpec, render_lidar_panorama

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _synthetic_scene(n: int = 4000, seed: int = 0):
    """A ring of opaque unit-quat Gaussians in front of the sensor."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    means = (torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 60.0
    means[:, 0] = torch.rand(n, device="cuda", generator=g) * 40.0 + 5.0  # in front
    quats = torch.zeros(n, 4, device="cuda")
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.2, device="cuda")
    opacities = torch.full((n,), 0.9, device="cuda")
    intensity = torch.full((n,), 0.5, device="cuda")
    raydrop = torch.full((n,), -6.0, device="cuda")
    return means, quats, scales, opacities, intensity, raydrop


def _spec() -> LidarSensorSpec:
    return LidarSensorSpec(name="t", sensor_type="XT32", s2b=np.eye(4), n_columns=512)


def _render(sensor_to_world, sensor_to_world_end=None, **kw):
    means, quats, scales, opac, inten, rdrop = _synthetic_scene()
    return render_lidar_panorama(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opac,
        intensity_sig=inten,
        raydrop_logit=rdrop,
        sensor_to_world=sensor_to_world,
        lidar_spec=_spec(),
        sensor_to_world_end=sensor_to_world_end,
        **kw,
    )


def test_static_end_pose_matches_global() -> None:
    """End pose == start pose must reproduce the single-pose (GLOBAL) render.

    The pose-interpolating path and the GLOBAL path are distinct gsplat kernels,
    so they match on returns but not on empty cells (whose distance is undefined
    where alpha ~ 0); compare only cells both renders mark as hits.
    """
    s2w = torch.eye(4, device="cuda")
    base = _render(s2w)
    rs = _render(s2w, sensor_to_world_end=s2w.clone())
    base_hit = base["alpha"] > 0.1
    rs_hit = rs["alpha"] > 0.1
    # Near-identical hit masks (a handful of near-tangent cells may resolve
    # differently between the GLOBAL and UT kernels).
    iou = (base_hit & rs_hit).sum().item() / max((base_hit | rs_hit).sum().item(), 1)
    assert iou > 0.99, f"hit-mask IoU {iou:.4f} too low"
    both = base_hit & rs_hit
    d = (base["distance"][both] - rs["distance"][both]).abs()
    assert (d < 0.1).float().mean().item() > 0.99
    assert d.mean().item() < 0.05


def test_motion_changes_the_scan() -> None:
    """A moving end pose must shift the scan away from the static render.

    With the midpoint approximation this measures the average displacement over
    the sweep, not intra-sweep skew (see the module docstring).
    """
    s2w = torch.eye(4, device="cuda")
    s2w_end = torch.eye(4, device="cuda")
    s2w_end[0, 3] = 2.0  # +2 m forward across the sweep
    static = _render(s2w)
    moved = _render(s2w, sensor_to_world_end=s2w_end)
    valid = (static["alpha"] > 0.1) & (moved["alpha"] > 0.1)
    assert valid.sum() > 0
    diff = (static["distance"] - moved["distance"])[valid].abs()
    assert diff.mean().item() > 0.01, "rolling shutter had no effect under motion"


def test_end_pose_is_not_just_the_midpoint_pose() -> None:
    """Motion must actually skew the sweep, not merely shift it.

    The renderer used to approximate a moving sweep by rendering everything
    from the midpoint pose; this asserts that is no longer what happens. That
    the displacement is also correct in sign and magnitude is checked in
    test_rolling_shutter.py, against a reference swept over many poses -- this
    scene does not cover the full azimuth, so its sweep extremes are too sparse
    to say anything about the error profile.
    """
    start = torch.eye(4, device="cuda")
    end = torch.eye(4, device="cuda")
    end[0, 3] = 4.0
    mid = torch.eye(4, device="cuda")
    mid[0, 3] = 2.0

    swept = _render(start, sensor_to_world_end=end)
    at_mid = _render(mid)

    both = (swept["alpha"] > 0.1) & (at_mid["alpha"] > 0.1)
    assert int(both.sum()) > 0
    diff = (swept["distance"] - at_mid["distance"]).abs()
    assert float(diff[both].mean()) > 1e-3, (
        "the swept render is indistinguishable from the midpoint pose"
    )


def test_inert_kernel_flags_are_accepted() -> None:
    """``with_ut``/``with_eval3d``/``packed`` no longer reach a kernel argument.

    They are gsplat-era knobs the SplatAD path ignores; they stay in the
    signature for call-site compatibility, so passing them must be a no-op
    rather than an error or a behaviour change.
    """
    s2w = torch.eye(4, device="cuda")
    base = _render(s2w)
    off = _render(s2w, with_ut=False, with_eval3d=False, packed=True)
    for key in ("alpha", "distance", "intensity", "raydrop_logit"):
        assert torch.equal(base[key], off[key]), f"{key} changed by an inert flag"
