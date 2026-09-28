"""Pose accumulation and interpolation for frame-rate rendering."""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass

from splatsim._geometry import lerp, slerp


@dataclass
class TimestampedPose:
    """A camera pose with an associated timestamp in nanoseconds."""

    time_ns: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # wxyz quaternion


class PoseBuffer:
    """Thread-safe pose buffer with interpolation.

    Accumulates timestamped poses and interpolates at arbitrary times.
    A ``threading.Event`` is set whenever a new pose is appended so that
    a consumer thread can wake up immediately.
    """

    def __init__(self, max_size: int = 100) -> None:
        self._max_size = max_size
        self._poses: list[TimestampedPose] = []
        self._lock = threading.Lock()
        self.new_pose_event = threading.Event()

    def append(self, pose: TimestampedPose) -> None:
        """Add a new pose.  Poses must arrive in non-decreasing time order."""
        with self._lock:
            self._poses.append(pose)
            if len(self._poses) > self._max_size:
                self._poses.pop(0)
        self.new_pose_event.set()

    def interpolate(self, time_ns: int) -> TimestampedPose | None:
        """Interpolate pose at *time_ns*.

        Returns ``None`` if *time_ns* is outside the buffered range.
        """
        with self._lock:
            return self._interpolate_unlocked(time_ns)

    def get_latest(self) -> TimestampedPose | None:
        """Return the most recently appended pose, or ``None``."""
        with self._lock:
            return self._poses[-1] if self._poses else None

    def get_earliest(self) -> TimestampedPose | None:
        """Return the oldest buffered pose, or ``None``."""
        with self._lock:
            return self._poses[0] if self._poses else None

    def trim_before(self, time_ns: int) -> None:
        """Remove poses older than *time_ns*, keeping one for interpolation."""
        with self._lock:
            times = [p.time_ns for p in self._poses]
            idx = bisect.bisect_left(times, time_ns)
            keep_from = max(0, idx - 1)
            if keep_from > 0:
                del self._poses[:keep_from]

    @property
    def latest_time_ns(self) -> int | None:
        with self._lock:
            return self._poses[-1].time_ns if self._poses else None

    @property
    def earliest_time_ns(self) -> int | None:
        with self._lock:
            return self._poses[0].time_ns if self._poses else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._poses)

    # ── private ───────────────────────────────────────────────────────

    def _interpolate_unlocked(self, time_ns: int) -> TimestampedPose | None:
        if not self._poses:
            return None

        times = [p.time_ns for p in self._poses]
        idx = bisect.bisect_right(times, time_ns)

        if idx == 0:
            return None  # before all poses
        if idx >= len(self._poses):
            if self._poses[-1].time_ns == time_ns:
                return self._poses[-1]
            return None

        p0 = self._poses[idx - 1]
        p1 = self._poses[idx]

        if p0.time_ns == p1.time_ns:
            return p0

        t = (time_ns - p0.time_ns) / (p1.time_ns - p0.time_ns)
        pos = lerp(p0.position, p1.position, t)
        rot = slerp(p0.rotation, p1.rotation, t)  # wxyz in, wxyz out
        return TimestampedPose(
            time_ns=time_ns,
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            rotation=(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
        )
