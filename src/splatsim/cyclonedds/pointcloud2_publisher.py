"""Publish ``sensor_msgs/PointCloud2`` from a splatsim LiDAR panorama."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from numpy.typing import NDArray

from splatsim.cyclonedds._util import to_dds_topic
from splatsim.cyclonedds.msg_types import (
    Header,
    PointCloud2,
    PointField,
    Time,
)

if TYPE_CHECKING:
    from cyclonedds.domain import DomainParticipant


# sensor_msgs/PointField datatype constants.
_PF_UINT8 = 2
_PF_UINT16 = 4
_PF_FLOAT32 = 7

# Byte size of each PointField datatype code we support. Extend this map when
# adding new field types to _POINT_FIELDS.
_DTYPE_SIZE: dict[int, int] = {_PF_UINT8: 1, _PF_UINT16: 2, _PF_FLOAT32: 4}

# Point layout (16 bytes/point): float32 x/y/z, then Autoware-style
# uint8 intensity, uint8 return_type, and a uint16 channel (ring / laser-beam
# index). Offsets are explicit so the packed record below matches byte-for-byte.
_POINT_FIELDS: tuple[PointField, ...] = (
    PointField(name="x", offset=0, datatype=_PF_FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=_PF_FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=_PF_FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=_PF_UINT8, count=1),
    PointField(name="return_type", offset=13, datatype=_PF_UINT8, count=1),
    PointField(name="channel", offset=14, datatype=_PF_UINT16, count=1),
)

# Derive point_step from the field layout so extending _POINT_FIELDS stays
# consistent without editing a magic number.
_POINT_STEP: int = max(
    f.offset + _DTYPE_SIZE[f.datatype] * f.count for f in _POINT_FIELDS
)

# Structured record mirroring the wire layout exactly. Explicit offsets keep
# it aligned with _POINT_FIELDS, and the itemsize is pinned to _POINT_STEP so
# ``tobytes()`` yields ``N * point_step`` bytes with no surprise padding.
_POINT_RECORD = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "return_type", "channel"],
        "formats": ["<f4", "<f4", "<f4", "u1", "u1", "<u2"],
        "offsets": [0, 4, 8, 12, 13, 14],
        "itemsize": _POINT_STEP,
    }
)


class PointCloud2Publisher:
    """Publishes point clouds as ``sensor_msgs/PointCloud2``.

    The wire layout is 16 bytes/point: float32 ``x/y/z`` followed by a uint8
    ``intensity`` (quantised from [0, 1]), a uint8 ``return_type`` (always 0),
    and a uint16 ``channel`` (ring / laser-beam index). See ``_POINT_FIELDS``.
    """

    def __init__(
        self,
        participant: DomainParticipant,
        *,
        topic_name: str = "/splatsim/lidar/pointcloud",
        frame_id: str = "splatsim_lidar",
    ) -> None:
        self._frame_id = frame_id
        topic = Topic(participant, to_dds_topic(topic_name), PointCloud2)
        self._writer = DataWriter(participant, topic)

    def publish(
        self,
        xyz: NDArray[np.float32],
        intensity: NDArray[np.float32],
        *,
        channel: NDArray[np.integer] | None = None,
        stamp: Time,
    ) -> None:
        """Publish one point cloud frame.

        Args:
            xyz: (N, 3) float32 coordinates in the sensor frame.
            intensity: (N,) float32 in [0, 1]; quantised to uint8 (``* 255``).
            channel: optional (N,) ring / laser-beam index per point. When the
                source lacks per-beam indices (e.g. the CARLA (N, 4) path) it is
                omitted and every point is published on channel 0.
            stamp: ROS 2 timestamp for the frame; callers must supply the clock
                that matches the rest of the graph (sim time under CARLA co-sim,
                wall clock only for standalone dev viewers).
        """
        msg = _make_pointcloud2_message(
            xyz,
            intensity,
            channel=channel,
            stamp=stamp,
            frame_id=self._frame_id,
        )
        self._writer.write(msg)

    def publish_packed(
        self,
        records: NDArray[np.uint8],
        count: int,
        *,
        stamp: Time,
    ) -> None:
        """Publish pre-packed point records (fast path).

        Args:
            records: ``(count * 16,)`` uint8 buffer already laid out in this
                publisher's point record format (x, y, z float32 little-endian |
                intensity uint8 | return_type uint8 | channel uint16), e.g. from
                :meth:`splatsim.lidar_renderer.LidarRenderer.panorama_to_pointcloud2_data`.
                Skips the per-frame CPU re-packing of xyz / intensity / channel.
            count: Number of points in ``records``.
            stamp: ROS 2 timestamp for the frame.
        """
        if records.size != count * _POINT_STEP:
            raise ValueError(
                f"records must be (count * {_POINT_STEP},) uint8; "
                f"got {records.size} bytes for {count} points"
            )
        msg = PointCloud2(
            header=Header(stamp=stamp, frame_id=self._frame_id),
            height=1,
            width=count,
            fields=list(_POINT_FIELDS),
            is_bigendian=False,
            point_step=_POINT_STEP,
            row_step=_POINT_STEP * count,
            data=records.tobytes(),
            is_dense=False,
        )
        self._writer.write(msg)


def _make_pointcloud2_message(
    xyz: NDArray[np.float32],
    intensity: NDArray[np.float32],
    *,
    channel: NDArray[np.integer] | None = None,
    stamp: Time,
    frame_id: str,
) -> PointCloud2:
    """Build the packed ``sensor_msgs/PointCloud2`` message."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3); got shape {xyz.shape}")
    if intensity.ndim != 1 or intensity.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"intensity must be (N,) matching xyz rows; got {intensity.shape} vs {xyz.shape}",
        )

    n = xyz.shape[0]

    if channel is None:
        channel_arr = np.zeros(n, dtype=np.uint16)
    else:
        channel_arr = np.asarray(channel).reshape(-1)
        if channel_arr.shape[0] != n:
            raise ValueError(
                f"channel must be (N,) matching xyz rows; got {channel_arr.shape} vs {xyz.shape}",
            )
        channel_arr = channel_arr.astype(np.uint16, copy=False)

    # Quantise reflectance [0, 1] -> uint8 [0, 255], clamping out-of-range.
    intensity_u8 = (
        np.clip(intensity.astype(np.float32, copy=False), 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    # Pack into a raw (N, point_step) byte buffer matching _POINT_RECORD's
    # layout. Building it from typed byte views (rather than assigning into a
    # structured array) keeps the write little-endian-explicit and avoids the
    # mixed-dtype field assignment that confuses static type inference.
    # return_type (byte 13) is left at its zeroed default (single return).
    buf = np.zeros((n, _POINT_STEP), dtype=np.uint8)
    xyz_le = np.ascontiguousarray(xyz, dtype="<f4")  # (N, 3)
    buf[:, 0:12] = xyz_le.view(np.uint8).reshape(n, 12)
    buf[:, 12] = intensity_u8
    chan_le = np.ascontiguousarray(channel_arr, dtype="<u2")  # (N,)
    buf[:, 14:16] = chan_le.view(np.uint8).reshape(n, 2)

    return PointCloud2(
        header=Header(stamp=stamp, frame_id=frame_id),
        height=1,
        width=n,
        fields=list(_POINT_FIELDS),
        is_bigendian=False,
        point_step=_POINT_STEP,
        row_step=_POINT_STEP * n,
        data=buf.tobytes(),
        # This builder does not scan for NaN/Inf, so we cannot honestly claim
        # density. Consumers must handle invalid returns.
        is_dense=False,
    )
