"""Rolling shutter must reproduce what a spinning sensor actually samples.

Each azimuth column is scanned at a different instant, so from a different
sensor pose. The rasterizer approximates that by displacing each Gaussian's
projection by its angular rate times the column's sweep time. These tests check
that approximation against a reference built without it: render the panorama
from many poses across the sweep and keep from each only the columns scanned at
that instant. That reference makes no assumption about the kernel's velocity
conventions -- sign, frame or units -- which is exactly what it is guarding.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import splatsim.lidar_renderer as lr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

SWEEP_HZ = 10.0
SWEEP_S = 1.0 / SWEEP_HZ


def _spec(n_columns: int = 1024) -> lr.LidarSensorSpec:
    return lr.LidarSensorSpec(
        name="t",
        sensor_type="XT32",
        s2b=np.eye(4),
        n_columns=n_columns,
        spinning_frequency_hz=SWEEP_HZ,
    )


def _ring_scene(n: int = 60_000, seed: int = 0):
    """Surfaces all around the sensor, so every azimuth has a return."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ang = torch.rand(n, device="cuda", generator=g) * 2 * np.pi
    rad = 12.0 + torch.rand(n, device="cuda", generator=g) * 3.0
    means = torch.stack(
        [
            rad * torch.cos(ang),
            rad * torch.sin(ang),
            (torch.rand(n, device="cuda", generator=g) - 0.5) * 6.0,
        ],
        dim=1,
    )
    quats = torch.zeros(n, 4, device="cuda")
    quats[:, 0] = 1.0
    return dict(
        means=means,
        quats=quats,
        scales=torch.full((n, 3), 0.12, device="cuda"),
        opacities=torch.full((n,), 0.95, device="cuda"),
        intensity_sig=torch.rand(n, device="cuda", generator=g),
        raydrop_logit=torch.full((n,), -6.0, device="cuda"),
    )


def _render(scene, spec, s2w, s2w_end=None):
    with torch.no_grad():
        return lr.render_lidar_panorama(
            sensor_to_world=s2w,
            sensor_to_world_end=s2w_end,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            **scene,
        )


def _pose_x(x: float) -> torch.Tensor:
    m = torch.eye(4, device="cuda")
    m[0, 3] = x
    return m


def _swept_reference(scene, spec, travel_m: float, slices: int = 32):
    """Ground truth:每 column rendered from the pose it was actually scanned at.

    No rolling-shutter machinery involved -- just static renders and pose
    interpolation, stitched by sweep time.
    """
    geom = lr._panorama_geometry(spec, torch.device("cuda"))
    phase = geom.roll_time_s / SWEEP_S  # (W,) ascending columns, [-0.5, +0.5]
    ref = None
    for k in range(slices):
        centre = (k + 0.5) / slices - 0.5
        out = _render(scene, spec, _pose_x(0.5 * travel_m + centre * travel_m))
        if ref is None:
            ref = {key: torch.zeros_like(v) for key, v in out.items()}
        lo, hi = k / slices - 0.5, (k + 1) / slices - 0.5
        # Output grids are descending elevation / CW azimuth; the phase is on
        # the ascending grid, so flip it to line up.
        sel = ((phase >= lo) & (phase < hi)).flip(0)
        for key in ref:
            if ref[key].dim() == 2:
                ref[key][:, sel] = out[key][:, sel]
            else:
                ref[key][:, sel, :] = out[key][:, sel, :]
    return ref


def _mean_range_error(a, b) -> float:
    both = (a["alpha"] > 0.1) & (b["alpha"] > 0.1)
    if int(both.sum()) == 0:
        return float("inf")
    return float((a["distance"][both] - b["distance"][both]).abs().mean())


@cuda
def test_beats_the_midpoint_approximation() -> None:
    """The whole point: it must be closer to the truth than doing nothing."""
    spec = _spec()
    scene = _ring_scene()
    travel = 20.0 * SWEEP_S  # 20 m/s for one sweep

    ref = _swept_reference(scene, spec, travel)
    midpoint = _render(scene, spec, _pose_x(0.5 * travel))
    swept = _render(scene, spec, _pose_x(0.0), _pose_x(travel))

    err_mid = _mean_range_error(ref, midpoint)
    err_rs = _mean_range_error(ref, swept)
    assert err_rs < err_mid / 2.0, (
        f"rolling shutter ({err_rs:.4f} m) is not clearly better than the "
        f"midpoint pose ({err_mid:.4f} m) -- check the sign, frame and units of "
        "the sweep velocities"
    )


@cuda
def test_error_shrinks_most_where_the_sweep_is_furthest_from_its_middle() -> None:
    """A sign error would show up as the ends getting WORSE, not better."""
    spec = _spec()
    scene = _ring_scene(seed=1)
    travel = 20.0 * SWEEP_S

    ref = _swept_reference(scene, spec, travel)
    midpoint = _render(scene, spec, _pose_x(0.5 * travel))
    swept = _render(scene, spec, _pose_x(0.0), _pose_x(travel))

    geom = lr._panorama_geometry(spec, torch.device("cuda"))
    phase = (geom.roll_time_s / SWEEP_S).flip(0)
    ends = (phase.abs() > 0.3).unsqueeze(0).expand_as(ref["distance"])

    def err_on(mask, other):
        both = (ref["alpha"] > 0.1) & (other["alpha"] > 0.1) & mask
        return float((ref["distance"][both] - other["distance"][both]).abs().mean())

    assert err_on(ends, swept) < err_on(ends, midpoint), (
        "the sweep extremes did not improve -- the displacement is applied with "
        "the wrong sign or magnitude"
    )


@cuda
def test_a_static_sweep_changes_nothing() -> None:
    """End pose == start pose must reproduce the plain single-pose render."""
    spec = _spec(n_columns=512)
    scene = _ring_scene(n=20_000, seed=2)
    s2w = _pose_x(0.0)
    plain = _render(scene, spec, s2w)
    swept = _render(scene, spec, s2w, s2w.clone())
    for key in ("distance", "alpha", "intensity"):
        torch.testing.assert_close(swept[key], plain[key], rtol=1e-4, atol=1e-4)


@cuda
def test_sweep_time_is_seconds_not_a_phase() -> None:
    """The column times must scale with the spin rate.

    They multiply an angular rate in deg/s inside the kernel, so a dimensionless
    phase would be wrong by exactly the sweep duration -- the bug this caught.
    """
    device = torch.device("cuda")
    fast = lr._panorama_geometry(_spec(), device)
    slow_spec = lr.LidarSensorSpec(
        name="t",
        sensor_type="XT32",
        s2b=np.eye(4),
        n_columns=1024,
        spinning_frequency_hz=SWEEP_HZ / 2.0,
    )
    slow = lr._panorama_geometry(slow_spec, device)

    assert abs(float(fast.roll_time_s.abs().max()) - SWEEP_S / 2) < 1e-6
    torch.testing.assert_close(slow.roll_time_s, fast.roll_time_s * 2.0)
    # First column scanned is the LAST in the ascending grid (the sweep runs
    # +180 -> -180 while the grid ascends).
    assert float(fast.roll_time_s[-1]) < 0.0 < float(fast.roll_time_s[0])


@cuda
def test_depth_compensation_off_is_not_paid_for() -> None:
    """The velocity path must not stage depth compensation it never uses.

    The renderer runs with depth compensation off, so those coefficients are
    all zero. Staging them anyway cost 8 of 48 bytes per Gaussian in shared
    memory -- and shared memory is what bounds blocks per SM here, which is why
    the rolling-shutter render was slower than the static one almost exactly in
    proportion to the record size. This pins that the zero case still renders
    correctly through the narrower record.
    """
    spec = _spec(n_columns=512)
    scene = _ring_scene(n=20_000, seed=4)
    travel = 20.0 * SWEEP_S
    swept = _render(scene, spec, _pose_x(0.0), _pose_x(travel))
    assert torch.isfinite(swept["distance"]).all()
    assert int((swept["alpha"] > 0.1).sum()) > 0
    # And it must still beat the midpoint against the swept reference.
    ref = _swept_reference(scene, spec, travel, slices=16)
    midpoint = _render(scene, spec, _pose_x(0.5 * travel))
    assert _mean_range_error(ref, swept) < _mean_range_error(ref, midpoint)
