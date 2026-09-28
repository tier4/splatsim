"""The byte-sequence CDR fast path must be bit-identical to stock cyclonedds.

splatsim.cyclonedds._fast_cdr replaces
``PlainCdrV2SequenceOfPrimitiveMachine.serialize`` with a memcpy for
bytes-like values of 1-byte primitive sequences (the multi-megabyte ``data``
fields of PointCloud2 / Image). The declared IDL type is untouched, so the
only thing that needs proving is that the emitted CDR does not change.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cyclonedds")

from cyclonedds.idl._machinery import PlainCdrV2SequenceOfPrimitiveMachine

import splatsim.cyclonedds  # noqa: F401  (applies the fast path)
from splatsim.cyclonedds._fast_cdr import _orig_serialize, _serialize_bytes_fast
from splatsim.cyclonedds.msg_types import (
    Header,
    Image,
    PointCloud2,
    PointField,
    Time,
)


def _pointcloud(n_points: int) -> PointCloud2:
    rng = np.random.default_rng(0)
    records = rng.integers(0, 256, size=n_points * 16, dtype=np.uint8)
    return PointCloud2(
        header=Header(stamp=Time(sec=1, nanosec=2), frame_id="lidar"),
        height=1,
        width=n_points,
        fields=[PointField(name="x", offset=0, datatype=7, count=1)],
        is_bigendian=False,
        point_step=16,
        row_step=16 * n_points,
        data=records.tobytes(),
        is_dense=False,
    )


def _image() -> Image:
    rng = np.random.default_rng(1)
    h, w = 24, 32
    return Image(
        header=Header(stamp=Time(sec=3, nanosec=4), frame_id="cam"),
        height=h,
        width=w,
        encoding="bgr8",
        is_bigendian=0,
        step=w * 3,
        data=rng.integers(0, 256, size=h * w * 3, dtype=np.uint8).tobytes(),
    )


@pytest.mark.parametrize(
    "msg",
    [_pointcloud(0), _pointcloud(1), _pointcloud(1000), _image()],
    ids=["pc-empty", "pc-1", "pc-1000", "image"],
)
def test_fast_path_cdr_is_bit_identical(msg) -> None:
    fast = type(msg).__idl__.serialize(msg)

    machine = PlainCdrV2SequenceOfPrimitiveMachine
    machine.serialize = _orig_serialize
    try:
        stock = type(msg).__idl__.serialize(msg)
    finally:
        machine.serialize = _serialize_bytes_fast  # ty: ignore[invalid-assignment]

    assert fast == stock


def test_fast_path_roundtrip() -> None:
    msg = _pointcloud(257)
    buf = PointCloud2.__idl__.serialize(msg)
    out = PointCloud2.__idl__.deserialize(buf)
    assert isinstance(out, PointCloud2)
    assert bytes(bytearray(out.data)) == msg.data
    assert out.width == msg.width


def test_fast_path_active() -> None:
    assert PlainCdrV2SequenceOfPrimitiveMachine.serialize is _serialize_bytes_fast


def test_non_byte_sequences_fall_through() -> None:
    # CameraInfo.d is sequence[float64]: size != 1 so it must take the stock
    # path and still serialize correctly.
    from splatsim.cyclonedds.msg_types import CameraInfo

    msg = CameraInfo(d=[0.1, 0.2, 0.3])
    out = CameraInfo.__idl__.deserialize(CameraInfo.__idl__.serialize(msg))
    assert isinstance(out, CameraInfo)
    assert out.d == pytest.approx([0.1, 0.2, 0.3])
