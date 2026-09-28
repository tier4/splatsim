"""Rig rendering: one shared Gaussian gather for N sensors.

``gather_lidar_rig`` / ``gather_camera_rig`` collect once for a whole rig so N
sensors cost one transient Gaussian buffer instead of N. The contract they must
keep is that the shared set is a SUPERSET of what any single sensor would have
selected on its own — LOD tiers are chosen per cell from the nearest mount, so
no sensor is ever handed a coarser tier than it asked for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from splatsim._conversions import GaussianTensors
from splatsim.dataclass.lod_config import LodConfig
from splatsim.lidar_renderer import (
    LidarRenderer,
    LidarSensorSpec,
    gather_lidar_rig,
    render_lidars_concurrent,
)
from splatsim.lod import LodManager
from splatsim.renderer import Renderer, gather_camera, gather_camera_rig
from splatsim.scene import Scene

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _tensors(n: int, device) -> GaussianTensors:
    g = torch.Generator(device=device).manual_seed(0)
    return GaussianTensors(
        means=(torch.rand(n, 3, device=device, generator=g) - 0.5) * 400.0,
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
        scales=torch.full((n, 3), 0.2, device=device),
        opacities=torch.full((n,), 0.9, device=device),
        colors=torch.rand(n, 3, device=device, generator=g),
        sh_degree=0,
        intensity_raw=torch.zeros(n, device=device),
        raydrop_logit=torch.full((n,), -6.0, device=device),
    )


class _Bg:
    """Minimal Background stand-in carrying a pre-computed LOD index."""

    def __init__(self, tensors, lod_manager):
        self.tensors, self.lod_index = lod_manager.precompute(tensors)

    @property
    def num_gaussians(self):
        return int(self.tensors.means.shape[0])


def _scene(device):
    mgr = LodManager(LodConfig())
    bg = _Bg(_tensors(600_000, device), mgr)
    # _Bg is a structural stand-in for Background (Scene only reads .tensors /
    # .lod_index here), which the type checker cannot see.
    return Scene(background=bg, lod_manager=mgr)  # ty: ignore[invalid-argument-type]


def _spec(name, offset):
    s2b = np.eye(4)
    s2b[:3, 3] = offset
    return LidarSensorSpec(name=name, sensor_type="XT32", s2b=s2b, n_columns=512)


@cuda
def test_shared_lidar_gather_is_a_superset_of_each_sensor() -> None:
    device = torch.device("cuda")
    scene = _scene(device)
    rends = [
        LidarRenderer(_spec("top", (0.0, 0.0, 2.0)), device=device, max_range_m=120.0),
        LidarRenderer(
            _spec("front_left", (2.0, 1.0, 0.5)), device=device, max_range_m=100.0
        ),
        LidarRenderer(
            _spec("rear", (-2.0, 0.0, 0.5)), device=device, max_range_m=100.0
        ),
    ]
    b2w = torch.eye(4, device=device)

    shared = gather_lidar_rig(rends, b2w, scene)
    assert shared is not None
    shared_keys = {tuple(v) for v in shared.means.tolist()}

    for r in rends:
        own = r.gather(b2w, scene)
        assert own is not None
        own_keys = {tuple(v) for v in own.means.tolist()}
        missing = own_keys - shared_keys
        assert not missing, (
            f"{r.sensor_spec.name}: shared gather dropped {len(missing)} Gaussians "
            "the per-sensor gather kept"
        )
        assert shared.count >= own.count


@cuda
def test_shared_gather_costs_one_buffer_not_n() -> None:
    """The rig holds ~one gathered set, not one per sensor."""
    device = torch.device("cuda")
    scene = _scene(device)
    rends = [
        LidarRenderer(
            _spec(f"s{i}", (i * 1.0, 0.0, 1.0)), device=device, max_range_m=120.0
        )
        for i in range(4)
    ]
    b2w = torch.eye(4, device=device)

    shared = gather_lidar_rig(rends, b2w, scene)
    per_sensor = [r.gather(b2w, scene) for r in rends]
    assert all(g is not None for g in per_sensor)
    per_sensor_total = sum(g.nbytes() for g in per_sensor if g is not None)
    assert shared is not None
    # Union of 4 nearby mounts stays far below the sum of 4 separate gathers.
    assert shared.nbytes() < per_sensor_total * 0.6


@cuda
def test_render_lidars_concurrent_matches_shared_sequential() -> None:
    """Streams must not change the result — only the scheduling.

    Repeated, because the failure this guards against is a race: the shared
    gather is produced on the current stream while the per-sensor rasterizations
    run on side streams, which do NOT inherit that dependency. Without an
    explicit ``wait_stream`` the first (largest) sensor reads a partially
    written buffer and comes back empty — intermittently, so a single
    comparison can pass by luck. A tiny scene hides it further, hence the
    larger one and the repeats.
    """
    import os

    device = torch.device("cuda")
    scene = _scene(device)
    rends = [
        LidarRenderer(_spec("a", (0.0, 0.0, 1.5)), device=device, max_range_m=120.0),
        LidarRenderer(_spec("b", (1.0, 0.5, 1.0)), device=device, max_range_m=120.0),
        LidarRenderer(_spec("c", (-1.0, -0.5, 1.2)), device=device, max_range_m=120.0),
    ]
    b2w = torch.eye(4, device=device)

    os.environ["SPLATSIM_LIDAR_CONCURRENT"] = "0"
    seq = render_lidars_concurrent(rends, b2w, scene)
    try:
        os.environ["SPLATSIM_LIDAR_CONCURRENT"] = "1"
        for trial in range(8):
            # Queue heavy work on the current stream first. The gather is
            # enqueued behind it, so it finishes late in wall-clock time while
            # the side streams are ready to go immediately -- exactly the window
            # the wait_stream guard closes. Without the guard the first sensor
            # reads the not-yet-written gather.
            conc = render_lidars_concurrent(rends, b2w, scene)
            for r, a, b in zip(rends, seq, conc):
                name = r.sensor_spec.name
                assert torch.equal(a["distance"], b["distance"]), (
                    f"trial {trial}, sensor {name}: distance differs "
                    f"({int((a['alpha'] > 0.1).sum())} vs "
                    f"{int((b['alpha'] > 0.1).sum())} hits)"
                )
                assert torch.equal(a["alpha"], b["alpha"])
                assert torch.equal(a["intensity"], b["intensity"])
    finally:
        os.environ.pop("SPLATSIM_LIDAR_CONCURRENT", None)


@cuda
def test_render_cameras_concurrent_matches_sequential() -> None:
    """Same race guard for the camera rig path."""
    import os

    from splatsim.renderer import render_cameras_concurrent

    device = torch.device("cuda")
    scene = _scene(device)
    K = torch.tensor(
        [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]], device=device
    )
    rends, viewmats = [], []
    for dx in (0.0, 2.0, -2.0):
        rends.append(Renderer(width=64, height=48, device=device))
        vm = torch.eye(4, device=device)
        vm[0, 3] = dx
        vm[2, 3] = 5.0
        viewmats.append(vm)
    Ks = [K] * len(rends)

    os.environ["SPLATSIM_CAMERA_CONCURRENT"] = "0"
    seq = render_cameras_concurrent(rends, viewmats, Ks, scene=scene)
    try:
        os.environ["SPLATSIM_CAMERA_CONCURRENT"] = "1"
        for trial in range(8):
            conc = render_cameras_concurrent(rends, viewmats, Ks, scene=scene)
            for i, (a, b) in enumerate(zip(seq, conc)):
                assert torch.equal(a, b), f"trial {trial}, camera {i}: image differs"
    finally:
        os.environ.pop("SPLATSIM_CAMERA_CONCURRENT", None)


@cuda
def test_shared_camera_gather_is_a_superset_of_each_camera() -> None:
    device = torch.device("cuda")
    scene = _scene(device)
    viewmats = []
    for dx in (0.0, 3.0, -3.0):
        vm = torch.eye(4, device=device)
        vm[0, 3] = dx
        viewmats.append(vm)

    shared = gather_camera_rig(scene, viewmats)
    assert shared is not None
    shared_keys = {tuple(v) for v in shared.means.tolist()}

    for vm in viewmats:
        pos = -(vm[:3, :3].T @ vm[:3, 3])
        own = gather_camera(scene, pos)
        assert own is not None
        assert not ({tuple(v) for v in own.means.tolist()} - shared_keys)


@cuda
def test_camera_render_accepts_a_shared_gather() -> None:
    device = torch.device("cuda")
    scene = _scene(device)
    r = Renderer(width=64, height=48, device=device)
    vm = torch.eye(4, device=device)
    vm[2, 3] = 5.0
    K = torch.tensor(
        [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]], device=device
    )
    shared = gather_camera_rig(scene, [vm])
    own = r.render(vm, K, scene=scene)
    via_shared = r.render(vm, K, scene=scene, shared=shared)
    assert own.shape == via_shared.shape == (48, 64, 3)
    # A single-camera rig gathers exactly what the camera would on its own.
    assert torch.equal(own, via_shared)


@cuda
def test_side_streams_wait_for_the_shared_gather() -> None:
    """Every side stream must wait on the stream that produced the gather.

    This is the contract behind a real, intermittent failure: the shared gather
    is enqueued on the current stream, but ``with torch.cuda.stream(st)`` does
    NOT make ``st`` wait for it. Without an explicit ``wait_stream`` the first
    (largest) sensor rasterized a partially written buffer — on a 27M-Gaussian
    scene that produced an EMPTY panorama in 11 of 12 runs.

    It is asserted structurally rather than behaviourally on purpose: a
    synthetic scene's gather finishes too fast to open the window, and making
    the GPU busy enough to open it also delays the side streams past it, so an
    output comparison here would pass either way (it does).
    """
    import splatsim.lidar_renderer as lr_mod
    from splatsim import renderer as cam_mod

    device = torch.device("cuda")
    scene = _scene(device)
    waited: list[tuple[int, int]] = []

    class _RecordingStream(torch.cuda.Stream):
        def wait_stream(self, other):  # ty: ignore[invalid-method-override]
            # Compare raw stream handles: torch.cuda.current_stream() hands back
            # a fresh wrapper object each call, so Python identity is useless.
            waited.append((self.cuda_stream, other.cuda_stream))
            return super().wait_stream(other)

    real_stream = torch.cuda.Stream
    # The rig caches its side streams, so the pool must be (re)built from the
    # recording subclass for this test to observe the waits.
    lr_mod._STREAM_POOL.clear()
    torch.cuda.Stream = _RecordingStream  # ty: ignore[invalid-assignment]
    try:
        rends = [
            LidarRenderer(
                _spec("a", (0.0, 0.0, 1.5)), device=device, max_range_m=120.0
            ),
            LidarRenderer(
                _spec("b", (1.0, 0.5, 1.0)), device=device, max_range_m=120.0
            ),
        ]
        waited.clear()
        lr_mod.render_lidars_concurrent(rends, torch.eye(4, device=device), scene)
        current_id = torch.cuda.current_stream().cuda_stream
        # Each of the two side streams waited on the current (gathering) stream.
        assert sum(1 for _s, o in waited if o == current_id) >= len(rends), (
            "LiDAR side streams did not wait on the gathering stream"
        )

        cams = [Renderer(width=32, height=24, device=device) for _ in range(2)]
        vms = []
        for dx in (0.0, 2.0):
            vm = torch.eye(4, device=device)
            vm[0, 3] = dx
            vm[2, 3] = 5.0
            vms.append(vm)
        K = torch.tensor(
            [[25.0, 0.0, 16.0], [0.0, 25.0, 12.0], [0.0, 0.0, 1.0]], device=device
        )
        waited.clear()
        cam_mod.render_cameras_concurrent(cams, vms, [K, K], scene=scene)
        cam_current = torch.cuda.current_stream().cuda_stream
        assert sum(1 for _s, o in waited if o == cam_current) >= len(cams), (
            "camera side streams did not wait on the gathering stream"
        )
    finally:
        torch.cuda.Stream = real_stream  # type: ignore[misc]
        lr_mod._STREAM_POOL.clear()


@cuda
def test_side_streams_are_reused_across_frames() -> None:
    """The rig must not allocate fresh CUDA streams per frame.

    Creating torch.cuda.Stream objects every frame churns the caching
    allocator's per-stream blocks, and the cost shows up as a bimodal stall:
    measured on a 27M-Gaussian scene with 5 sensors, every other frame jumped
    151 -> 498 ms (13x on a light frame). Reusing a pool removes it (spread
    1.02x) and is what makes the concurrent path beat sequential at all.
    """
    device = torch.device("cuda")
    scene = _scene(device)
    rends = [
        LidarRenderer(_spec("a", (0.0, 0.0, 1.5)), device=device, max_range_m=120.0),
        LidarRenderer(_spec("b", (1.0, 0.5, 1.0)), device=device, max_range_m=120.0),
    ]
    b2w = torch.eye(4, device=device)

    seen: list[tuple[int, ...]] = []
    real_wait = torch.cuda.Stream.wait_stream

    def _spy(self, other):
        seen.append(self.cuda_stream)
        return real_wait(self, other)

    torch.cuda.Stream.wait_stream = _spy  # ty: ignore[invalid-assignment]
    try:
        rounds = []
        for _ in range(4):
            seen.clear()
            render_lidars_concurrent(rends, b2w, scene)
            # wait_stream is called on each side stream (before) and on the
            # current stream (after); the side-stream handles are what matter.
            rounds.append(sorted(set(seen)))
    finally:
        torch.cuda.Stream.wait_stream = real_wait  # type: ignore[assignment]

    assert all(r == rounds[0] for r in rounds), (
        f"the rig used different CUDA streams across frames: {rounds}; "
        "it should reuse a cached pool"
    )
    assert len(rounds[0]) >= len(rends)


@cuda
def test_stream_pool_grows_for_a_bigger_rig() -> None:
    """The cache must serve a larger rig, not hand back too few streams."""
    from splatsim.lidar_renderer import _side_streams

    device = torch.device("cuda")
    small = _side_streams(2, device)
    big = _side_streams(6, device)
    assert len(small) == 2 and len(big) == 6
    # Existing streams are kept (the pool grows, it does not reallocate).
    assert [s.cuda_stream for s in big[:2]] == [s.cuda_stream for s in small]
    assert len({s.cuda_stream for s in big}) == 6, "pooled streams must be distinct"
