"""Rolling-shutter pose plumbing in the gRPC pose-stream loop.

The kernel-side sweep compensation is covered by test_rolling_shutter.py;
these tests guard the server side: the render loop must hand each frame the
pose one LiDAR sweep BEFORE the rendered (latest) pose, reconstructed from
the buffered pose queue, and consumed poses must be trimmed afterwards.
"""

from __future__ import annotations

import time

from splatsim.grpc_service._generated import rendering_service_pb2 as pb2
from splatsim.grpc_service.pose_buffer import PoseBuffer, TimestampedPose
from splatsim.grpc_service.server import RenderingServiceServicer, _sweep_time_ns


def test_pose_buffer_get_earliest() -> None:
    buf = PoseBuffer()
    assert buf.get_earliest() is None
    for t in (10, 20, 30):
        buf.append(TimestampedPose(t, (float(t), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    earliest = buf.get_earliest()
    latest = buf.get_latest()
    assert earliest is not None and latest is not None
    assert earliest.time_ns == 10
    assert latest.time_ns == 30


def test_sweep_time_ns() -> None:
    assert _sweep_time_ns(10.0) == 100_000_000
    assert _sweep_time_ns(20.0) == 50_000_000


def test_sweep_time_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("SPLATSIM_LIDAR_ROLLING_SHUTTER", "0")
    assert _sweep_time_ns(10.0) is None


def _lidar_data(time_ns: int, x: float) -> pb2.LidarData:
    msg = pb2.LidarData()
    msg.stamp.sec, msg.stamp.nanosec = divmod(time_ns, 1_000_000_000)
    msg.pose.position.x = x
    msg.pose.rotation.w = 1.0
    return msg


def _run_stream(poses, sweep_time_ns):
    """Drive _run_pose_stream with a slow enough producer that every pose
    becomes a frame, and record (pose, render_time_ns, sweep_start) calls."""
    servicer = RenderingServiceServicer()
    calls: list[tuple[TimestampedPose, int, TimestampedPose | None]] = []

    def record(pose, render_time_ns, sweep_start=None):
        calls.append((pose, render_time_ns, sweep_start))

    def producer():
        for p in poses:
            yield p
            # Let the render thread consume this pose before the next arrives.
            time.sleep(0.03)

    summary = servicer._run_pose_stream(
        producer(),
        frame_rate=1000.0,
        render_and_publish=record,
        sweep_time_ns=sweep_time_ns,
    )
    return summary, calls


def test_stream_supplies_sweep_start_one_sweep_back() -> None:
    sweep = 100  # ns; poses every 50 ns so the sweep start needs interpolation
    poses = [_lidar_data(t, float(t)) for t in (0, 50, 100, 150, 200)]
    summary, calls = _run_stream(poses, sweep_time_ns=sweep)

    assert summary.poses_received == 5
    assert len(calls) >= 2
    by_time = {render_t: start for _, render_t, start in calls}

    # The very first frame has no history: static render.
    if 0 in by_time:
        assert by_time[0] is None

    for render_t, start in by_time.items():
        if render_t == 0 or start is None:
            continue
        # Either a full sweep of history existed (exact) or the queue was
        # shorter and the earliest buffered pose was used as fallback.
        assert start.time_ns <= render_t - sweep or start.time_ns < render_t
        if start.time_ns == render_t - sweep:
            # Positions grow linearly with time, so interpolation must too.
            assert abs(start.position[0] - float(start.time_ns)) < 1e-6

    # At least one late frame must have had the full sweep of history.
    assert any(
        start is not None and start.time_ns == render_t - sweep
        for _, render_t, start in calls
        if render_t >= 2 * sweep
    )


def test_stream_without_sweep_stays_static() -> None:
    poses = [_lidar_data(t, float(t)) for t in (0, 50, 100)]
    _, calls = _run_stream(poses, sweep_time_ns=None)
    assert calls
    assert all(start is None for _, _, start in calls)
