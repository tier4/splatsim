"""Sector streaming: render a LiDAR revolution as azimuth wedges.

Three contracts are pinned here:

* **Azimuth-wedge cull parity** — the fused CUDA cull's wedge test matches the
  PyTorch fallback bit-for-bit, including wedges touching the ±180° seam.
* **Sector equivalence** — concatenating the S sector renders reproduces the
  full-frame panorama EXACTLY when the pose is static (the tile grid stays the
  full ring; a sector only rasterizes its own tile columns), and closely under
  translation (rolling shutter per sector vs one whole-sweep pose pair are
  different — per-sector is the better — approximations, so only near-equality
  can be asserted).
* **The gRPC sector loop** — sectors are scheduled off the pose timeline
  (rendered as soon as their sweep slice is covered), revolutions publish with
  the sweep-end stamp, and the loop resyncs to fresh poses after a gap.
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ── azimuth-wedge cull parity ────────────────────────────────────────────


@cuda
@pytest.mark.parametrize("az_center", [0.0, 2.0, math.pi - 0.05, -math.pi + 0.05])
@pytest.mark.parametrize("use_elev", [False, True])
def test_azimuth_wedge_cull_cuda_matches_pytorch(
    az_center: float, use_elev: bool
) -> None:
    from splatsim import lidar_renderer as lr
    from splatsim._lidar_cull_ext import is_available

    if not is_available():
        pytest.skip("CUDA cull extension unavailable")

    g = torch.Generator(device="cuda").manual_seed(0)
    n = 200_000
    means = (
        (torch.rand(n, 3, device="cuda", generator=g) * 2 - 1) * 300.0
    ).contiguous()
    scales = (0.05 + torch.rand(n, 3, device="cuda", generator=g) * 0.5).contiguous()

    s2w = torch.eye(4, device="cuda")
    s2w[:3, 3] = torch.tensor([1.0, -2.0, 1.5], device="cuda")

    kwargs: dict[str, Any] = dict(
        means=means,
        scales=scales,
        sensor_to_world=s2w,
        min_range_m=0.5,
        max_range_m=150.0,
        cull_scale_sigmas=3.0,
        elev_fov_cull=use_elev,
        sin_min=math.sin(-0.4),
        cos_min=math.cos(-0.4),
        sin_max=math.sin(0.3),
        cos_max=math.cos(0.3),
        azim_cull=True,
        az_center_rad=az_center,
        az_halfwidth_rad=math.pi / 8,
        az_pad_rad=0.01,
    )
    keep_cuda = lr._lidar_cull_keep(**kwargs)
    lr._USE_CUDA_CULL = False
    try:
        keep_ref = lr._lidar_cull_keep(**kwargs)
    finally:
        lr._USE_CUDA_CULL = True
    assert torch.equal(keep_cuda, keep_ref)
    # The wedge must actually cull (π/8 of the circle plus margins).
    assert keep_cuda.sum() < n // 2


@cuda
def test_azimuth_wedge_keeps_far_side_of_seam() -> None:
    """A wedge centred at +π must keep Gaussians just across the -π seam."""
    from splatsim import lidar_renderer as lr

    # Two Gaussians a hair on either side of the seam, one at the wedge's
    # antipode.
    means = torch.tensor(
        [
            [-50.0, 0.5, 0.0],  # az ≈ +π - ε  (inside)
            [-50.0, -0.5, 0.0],  # az ≈ -π + ε (inside via wrap)
            [50.0, 0.0, 0.0],  # az = 0 (antipode, outside)
        ],
        device="cuda",
    )
    scales = torch.full((3, 3), 0.1, device="cuda")
    keep = lr._lidar_cull_keep(
        means=means,
        scales=scales,
        sensor_to_world=torch.eye(4, device="cuda"),
        min_range_m=0.1,
        max_range_m=None,
        cull_scale_sigmas=3.0,
        elev_fov_cull=False,
        sin_min=0.0,
        cos_min=1.0,
        sin_max=0.0,
        cos_max=1.0,
        azim_cull=True,
        az_center_rad=math.pi,
        az_halfwidth_rad=math.pi / 8,
        az_pad_rad=0.01,
    )
    assert keep.tolist() == [True, True, False]


# ── sector geometry (CPU) ────────────────────────────────────────────────


def _spec(n_columns: int = 128, n_rows: int = 8):
    from splatsim.lidar_renderer import LidarSensorSpec

    return LidarSensorSpec(
        name="t",
        sensor_type="",
        n_columns=n_columns,
        s2b=np.eye(4),
        el_lo_rad=-0.35,
        el_hi_rad=0.26,
        n_rows_uniform=n_rows,
        spinning_frequency_hz=10.0,
    )


def test_sector_geometry_slices_the_full_grid() -> None:
    from splatsim.lidar_renderer import _panorama_geometry

    spec = _spec(n_columns=128)
    full = _panorama_geometry(spec, "cpu")
    s_count = 4
    secs = [_panorama_geometry(spec, "cpu", (k, s_count)) for k in range(s_count)]

    assert torch.equal(torch.cat([s.azs_cw for s in secs]), full.azs_cw)
    assert torch.equal(torch.cat([s.dirs for s in secs], dim=1), full.dirs)
    # Sector roll times are relative to the sector's own mid-window pose;
    # shifting each by its window midpoint must reproduce the full timeline.
    sweep = full.window_s
    rt = torch.cat(
        [
            torch.flip(s.roll_time_s, [0]) + ((2 * k + 1) / (2 * s_count) - 0.5) * sweep
            for k, s in enumerate(secs)
        ]
    )
    assert torch.allclose(rt, torch.flip(full.roll_time_s, [0]), atol=1e-7)
    assert secs[0].window_s == pytest.approx(sweep / s_count)
    # Ascending-grid tile offsets: sector S-1 (CW end = ascending start) is 0.
    widths = [s.azs_cw.shape[0] for s in secs]
    assert all(w == 128 // s_count for w in widths)
    assert [s.tile_col_offset for s in reversed(secs)] == [
        i * (128 // s_count) // 16 for i in range(s_count)
    ]


def test_uneven_sector_counts_split_on_tile_boundaries() -> None:
    """3 sectors over 8 tiles: widths differ by one tile but still slice the
    full grid exactly (the real rig has 3600-column sensors, where no sector
    count divides into equal tile-aligned wedges)."""
    from splatsim.lidar_renderer import _panorama_geometry

    spec = _spec(n_columns=128)
    full = _panorama_geometry(spec, "cpu")
    secs = [_panorama_geometry(spec, "cpu", (k, 3)) for k in range(3)]
    assert [s.azs_cw.shape[0] for s in secs] == [32, 48, 48]
    assert all(s.azs_cw.shape[0] % 16 == 0 for s in secs)
    assert torch.equal(torch.cat([s.azs_cw for s in secs]), full.azs_cw)
    # Roll times must still be the full timeline shifted by each POSE window's
    # midpoint (k/3 fractions), even though the columns lean past the window.
    sweep = full.window_s
    rt = torch.cat(
        [
            torch.flip(s.roll_time_s, [0]) + ((2 * k + 1) / 6 - 0.5) * sweep
            for k, s in enumerate(secs)
        ]
    )
    assert torch.allclose(rt, torch.flip(full.roll_time_s, [0]), atol=1e-7)
    # The kernel's extent expansion must cover the actual roll-time span.
    for s in secs:
        assert 2 * s.roll_time_s.abs().max().item() <= s.rs_time_s + 1e-6


def test_sector_rendering_needs_tile_aligned_columns() -> None:
    from splatsim.lidar_renderer import _panorama_geometry

    with pytest.raises(ValueError, match="multiple"):
        _panorama_geometry(_spec(n_columns=100), "cpu", (0, 2))
    with pytest.raises(ValueError, match="exceeds"):
        _panorama_geometry(_spec(n_columns=128), "cpu", (0, 9))


# ── sector render equivalence (GPU) ──────────────────────────────────────


def _gaussians(n: int = 20_000) -> dict[str, Any]:
    g = torch.Generator(device="cuda").manual_seed(0)
    return dict(
        means=(torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 120.0,
        quats=torch.randn(n, 4, device="cuda", generator=g),
        scales=torch.rand(n, 3, device="cuda", generator=g) * 0.3 + 0.02,
        opacities=torch.rand(n, device="cuda", generator=g) * 0.8 + 0.1,
        intensity_sig=torch.rand(n, device="cuda", generator=g),
        raydrop_logit=torch.randn(n, device="cuda", generator=g) - 2.0,
    )


@cuda
@pytest.mark.parametrize("sector_mask", [False, True], ids=["compact", "mask"])
def test_static_sector_concat_matches_full_render_exactly(
    sector_mask: bool, monkeypatch
) -> None:
    """Both cull applications — compaction (default) and the mask-based
    projection (SPLATSIM_LIDAR_SECTOR_MASK=1) — must reproduce the full
    render exactly."""
    from splatsim import lidar_renderer as lr
    from splatsim.lidar_renderer import render_lidar_panorama

    monkeypatch.setattr(lr, "_SECTOR_MASK", sector_mask)
    spec = _spec(n_columns=512, n_rows=32)
    kw: dict[str, Any] = dict(
        **_gaussians(), lidar_spec=spec, min_range_m=0.5, max_range_m=100.0
    )
    s2w = torch.eye(4, device="cuda")
    s2w[2, 3] = 1.5

    full = render_lidar_panorama(sensor_to_world=s2w, **kw)
    s_count = 8
    parts = [
        render_lidar_panorama(sensor_to_world=s2w, sector=(k, s_count), **kw)
        for k in range(s_count)
    ]
    for key in ("alpha", "distance", "points", "intensity", "raydrop_logit"):
        cat = torch.cat([p[key] for p in parts], dim=1)
        assert torch.equal(cat, full[key]), key


@cuda
def test_translating_sector_concat_stays_close_to_full_render() -> None:
    from splatsim.lidar_renderer import render_lidar_panorama

    spec = _spec(n_columns=512, n_rows=32)
    kw: dict[str, Any] = dict(
        **_gaussians(), lidar_spec=spec, min_range_m=0.5, max_range_m=100.0
    )
    sweep = 0.1
    vel = torch.tensor([8.0, 2.0, 0.0], device="cuda")

    def pose_at(f: float) -> torch.Tensor:
        m = torch.eye(4, device="cuda")
        m[:3, 3] = torch.tensor([0.0, 0.0, 1.5], device="cuda") + vel * (f * sweep)
        return m

    s_count = 8
    full = render_lidar_panorama(
        sensor_to_world=pose_at(0.0), sensor_to_world_end=pose_at(1.0), **kw
    )
    parts = [
        render_lidar_panorama(
            sensor_to_world=pose_at(k / s_count),
            sensor_to_world_end=pose_at((k + 1) / s_count),
            sector=(k, s_count),
            **kw,
        )
        for k in range(s_count)
    ]
    d = (torch.cat([p["distance"] for p in parts], dim=1) - full["distance"]).abs()
    q = d.flatten().sort().values
    # Both are approximations of the same swept scan (per-sector poses vs one
    # whole-sweep pair); they must agree everywhere except a handful of cells
    # sitting on depth discontinuities, where a sub-column azimuth shift flips
    # which surface the median lands on.
    assert q[int(0.999 * q.numel())].item() < 0.05
    a = (torch.cat([p["alpha"] for p in parts], dim=1) - full["alpha"]).abs()
    assert a.mean().item() < 5e-3


@cuda
def test_depth_lanes_match_serial_rasterization(monkeypatch) -> None:
    """The depth-parallel rasterizer (16 lanes/pixel, associative composition)
    must reproduce the serial kernel: alpha and median depth measure
    bit-identical (the median is an exact selection; the thresholds are
    resolved by re-walking the crossing lane with the serial logic), features
    to float epsilon (a lane partial is scaled by the prefix transmittance,
    which rounds differently from the serial running product)."""
    from splatsim import lidar_renderer as lr
    from splatsim.lidar_renderer import render_lidar_panorama

    spec = _spec(n_columns=512, n_rows=32)
    kw: dict[str, Any] = dict(
        **_gaussians(), lidar_spec=spec, min_range_m=0.5, max_range_m=100.0
    )
    start = torch.eye(4, device="cuda")
    start[2, 3] = 1.5
    end = start.clone()
    end[:3, 3] += torch.tensor([0.8, 0.2, 0.0], device="cuda")

    outs = {}
    for lanes in (False, True):
        monkeypatch.setattr(lr, "_DEPTH_LANES", lanes)
        outs[lanes] = {
            "static": render_lidar_panorama(sensor_to_world=start, **kw),
            "rolling": render_lidar_panorama(
                sensor_to_world=start, sensor_to_world_end=end, **kw
            ),
        }
    for mode in ("static", "rolling"):
        a, b = outs[False][mode], outs[True][mode]
        assert torch.equal(a["alpha"], b["alpha"]), mode
        assert torch.equal(a["distance"], b["distance"]), mode
        for key in ("intensity", "raydrop_logit"):
            assert torch.allclose(a[key], b[key], atol=1e-5), (mode, key)


# ── the gRPC sector loop ─────────────────────────────────────────────────


def _pose_msg(t_ns: int, x: float):
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=0.0, z=0.0),
            rotation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
        actors=(),
    )


def test_sector_loop_schedules_windows_and_publishes_revolutions() -> None:
    from splatsim.grpc_service.server import RenderingServiceServicer, _SectorHooks

    servicer = RenderingServiceServicer()
    period = 100_000_000  # 100 ms sweep
    n_sectors = 4
    rendered: list[tuple[int, int, int]] = []  # (k, start_ns, end_ns)
    published: list[tuple[int, int]] = []  # (n_outputs, stamp_ns)
    states: list[dict] = []
    done = threading.Event()

    def render_sector(start, end, k, state):
        if not states or states[-1] is not state:
            states.append(state)
        state["gathers"] = state.get("gathers", 0) + 1
        rendered.append((k, start.time_ns, end.time_ns))
        return k

    def publish_revolution(outputs, end_pose, stamp_ns):
        published.append((len(outputs), stamp_ns))
        if len(published) >= 2:
            done.set()

    hooks = _SectorHooks(
        n_sectors=n_sectors,
        spin_period_ns=period,
        render_sector=render_sector,
        publish_revolution=publish_revolution,
    )

    t0 = 1_000_000_000

    def poses():
        # First pose alone, then give the render thread time to anchor the
        # revolution at it (it anchors at the NEWEST pose on its first wake).
        yield _pose_msg(t0, x=0.0)
        time.sleep(0.1)
        # 10 ms pose cadence over two sweeps + one extra pose so the second
        # revolution's last window closes.
        for i in range(1, 2 * 10 + 1):
            yield _pose_msg(t0 + i * 10_000_000, x=float(i))
        # Keep the stream open until the render thread has drained everything;
        # closing it immediately would tear the loop down mid-revolution.
        done.wait(timeout=5.0)

    summary = servicer._run_pose_stream(
        poses(),
        frame_rate=10.0,
        render_and_publish=lambda *a: pytest.fail("plain loop must not run"),
        sweep_time_ns=period,
        sector_hooks=hooks,
    )

    assert done.is_set(), "expected two published revolutions"
    assert summary.frames_rendered == len(published) >= 2
    # First revolution: anchored at the first pose, 4 windows of 25 ms.
    first = rendered[:n_sectors]
    assert [k for k, _, _ in first] == list(range(n_sectors))
    for k, start_ns, end_ns in first:
        assert start_ns == t0 + k * period // n_sectors
        assert end_ns == t0 + (k + 1) * period // n_sectors
    # Revolution stamp = its sweep-end time; revolutions are back-to-back.
    assert published[0][0] == n_sectors
    assert published[0][1] == t0 + period
    assert published[1][1] == t0 + 2 * period
    # One shared-state dict per revolution (the per-revolution gather cache).
    assert all(s["gathers"] == n_sectors for s in states[:2])
    assert states[0] is not states[1]


def test_sector_loop_resyncs_after_a_pose_gap() -> None:
    from splatsim.grpc_service.server import RenderingServiceServicer, _SectorHooks

    servicer = RenderingServiceServicer()
    period = 100_000_000
    n_sectors = 4
    rendered: list[tuple[int, int, int]] = []
    published: list[int] = []
    done = threading.Event()

    def render_sector(start, end, k, state) -> None:
        rendered.append((k, start.time_ns, end.time_ns))

    def publish_revolution(outputs, end_pose, stamp_ns) -> None:
        published.append(stamp_ns)
        if len(published) >= 2:
            done.set()

    hooks = _SectorHooks(
        n_sectors=n_sectors,
        spin_period_ns=period,
        render_sector=render_sector,
        publish_revolution=publish_revolution,
    )

    t0 = 1_000_000_000
    gap_t0 = t0 + period + 5 * period  # 5 sweeps of silence after one sweep

    def poses():
        # Let the loop anchor the first revolution at t0 (it anchors at the
        # newest buffered pose on its first wake).
        yield _pose_msg(t0, x=0.0)
        time.sleep(0.1)
        for i in range(1, 11):  # one full sweep
            yield _pose_msg(t0 + i * 10_000_000, x=float(i))
        # The first revolution must be out the door before the gap pose lands,
        # otherwise the loop legitimately resyncs straight past it.
        deadline = time.monotonic() + 2.0
        while not published and time.monotonic() < deadline:
            time.sleep(0.005)
        # ...long gap, then a fresh burst. Anchor again before streaming it.
        yield _pose_msg(gap_t0, x=100.0)
        time.sleep(0.1)
        for i in range(1, 11):
            yield _pose_msg(gap_t0 + i * 10_000_000, x=100.0 + i)
        done.wait(timeout=5.0)

    servicer._run_pose_stream(
        poses(),
        frame_rate=10.0,
        render_and_publish=lambda *a: pytest.fail("plain loop must not run"),
        sweep_time_ns=period,
        sector_hooks=hooks,
    )

    assert done.is_set(), "expected a revolution on each side of the gap"
    assert published[0] == t0 + period
    # After the gap the loop must anchor at the fresh poses, not grind through
    # the 5 silent sweeps: the second revolution starts at the first post-gap
    # pose.
    second = rendered[n_sectors : 2 * n_sectors]
    assert second[0][1] == gap_t0
    assert published[1] == gap_t0 + period


def test_sector_count_env_parsing(monkeypatch) -> None:
    from splatsim.grpc_service import server

    monkeypatch.delenv("SPLATSIM_LIDAR_SECTORS", raising=False)
    assert server._lidar_sector_count() == 1
    monkeypatch.setenv("SPLATSIM_LIDAR_SECTORS", "8")
    assert server._lidar_sector_count() == 8
    monkeypatch.setenv("SPLATSIM_LIDAR_SECTORS", "0")
    assert server._lidar_sector_count() == 1
    monkeypatch.setenv("SPLATSIM_LIDAR_SECTORS", "banana")
    assert server._lidar_sector_count() == 1
