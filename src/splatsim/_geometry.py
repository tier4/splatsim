"""Shared geometry primitives: quaternion / rotation / pose-interpolation math.

Before this module the same handful of transforms was re-implemented across
``_usdz``, ``lidar_renderer``, ``grpc_service.pose_buffer``,
``carla_integration`` and ``eval/eval_lidar`` — each with its own quaternion
order and its own SLERP/LERP copy. They now all delegate here.

The rotation math itself is delegated to :mod:`scipy.spatial.transform`
(``Rotation`` / ``Slerp``); this module only pins the **quaternion-order
convention** and adapts to it.

Quaternion order convention
---------------------------
Unless a function's ``order`` argument says otherwise, quaternions are
**(w, x, y, z)** ("wxyz") — the gsplat / ROS / ``spz`` convention and the
majority order across splatsim. scipy works internally in **(x, y, z, w)**
("xyzw", the scipy / Eigen / glTF order, and the order
``3dgs_io.parse_rig_trajectories`` exposes), so callers using that order pass
``order="xyzw"`` and skip the reordering.

Every function operates on plain array-likes and returns ``float64`` numpy
arrays. The differentiable, torch-based rotation used inside the Gaussian
render path is deliberately kept separate as
:func:`splatsim._conversions.quat_to_rotation_matrix` (same wxyz convention).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

QuatOrder = str  # "wxyz" | "xyzw"
# Anything convertible to a small fixed-length float vector (tuple/list/ndarray).
Vec = Sequence[float] | np.ndarray


def _to_xyzw(q: Vec, order: QuatOrder) -> np.ndarray:
    """Return ``[x, y, z, w]`` (scipy's order) from a quaternion given in ``order``."""
    a = np.asarray(q, dtype=np.float64)
    if order == "xyzw":
        return a
    if order == "wxyz":
        return a[[1, 2, 3, 0]]
    raise ValueError(f"quaternion order must be 'wxyz' or 'xyzw', got {order!r}")


def _from_xyzw(q_xyzw: np.ndarray, order: QuatOrder) -> np.ndarray:
    """Reorder a scipy ``[x, y, z, w]`` quaternion back to ``order``."""
    if order == "xyzw":
        return q_xyzw
    return q_xyzw[[3, 0, 1, 2]]  # -> wxyz


def quat_xyzw_to_wxyz(q_xyzw: Vec) -> tuple[float, float, float, float]:
    """Reorder an ``xyzw`` quaternion to ``wxyz``.

    Scene bundles (track poses, rig poses, sensor extrinsics) are ``xyzw``;
    every splatsim pose is ``wxyz``. Reordering by hand is silent when it goes
    wrong — the result is still a unit quaternion, just the wrong rotation — so
    the conversion lives here with the rest of the frame plumbing.
    """
    x, y, z, w = (float(v) for v in q_xyzw)
    return (w, x, y, z)


def quat_to_matrix(q: Vec, *, order: QuatOrder = "wxyz") -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix (float64).

    The quaternion is normalized first (by scipy), so callers need not
    pre-normalize; a zero-norm quaternion raises ``ValueError``. ``order``
    selects the input component order (see the module docstring); the default
    is ``"wxyz"``.
    """
    return Rotation.from_quat(_to_xyzw(q, order)).as_matrix()


def mat4(r: np.ndarray, t: Vec) -> np.ndarray:
    """Assemble a 4x4 rigid transform from a 3x3 rotation and a translation."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.asarray(r, dtype=np.float64)
    m[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return m


def rpy_deg_to_matrix(rpy_deg: Vec) -> np.ndarray:
    """3x3 rotation from intrinsic roll/pitch/yaw in **degrees** (Z·Y·X order)."""
    return Rotation.from_euler(
        "xyz", np.asarray(rpy_deg, dtype=np.float64), degrees=True
    ).as_matrix()


def slerp(q0: Vec, q1: Vec, t: float, *, order: QuatOrder = "wxyz") -> np.ndarray:
    """Shortest-path spherical linear interpolation between two unit quaternions.

    Both inputs are normalized first and the result is a unit quaternion in the
    same ``order`` as the inputs (default ``"wxyz"``). scipy's :class:`Slerp`
    handles the shortest-path sign choice; ``t`` must lie in ``[0, 1]``.
    """
    rots = Rotation.from_quat(np.stack([_to_xyzw(q0, order), _to_xyzw(q1, order)]))
    interp = Slerp((0.0, 1.0), rots)(float(t))
    return _from_xyzw(interp.as_quat(), order)


def lerp(p0: Vec, p1: Vec, t: float) -> np.ndarray:
    """Linear interpolation ``(1 - t) * p0 + t * p1`` (float64 array)."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    return a + t * (b - a)
