"""Fast CDR serialization for byte payloads in cyclonedds-python.

The stock ``PlainCdrV2SequenceOfPrimitiveMachine.serialize`` writes a
``sequence[uint8]`` field with ``buffer.write_multi(f"{n}B", n, *value)``:
it unpacks the entire payload into ``n`` individual Python int arguments and
packs them one by one. For the multi-megabyte ``data`` fields of PointCloud2 /
Image messages that is a multi-millisecond, non-yielding GIL hold per publish
— measured to stall splatsim's gRPC pose-ingestion thread by up to ~98 ms per
frame (81% of all GIL-holding time under a 5-LiDAR rig).

For a 1-byte-aligned primitive sequence the XCDR2 body after the uint32
length prefix is exactly the raw bytes, so when the value is already a
``bytes``-like object we memcpy it instead. The wire format is bit-identical
(covered by test_fast_cdr.py) and the declared IDL type is untouched, so
XTypes discovery and ROS 2 interop are unaffected.

Importing :mod:`splatsim.cyclonedds` applies the patch process-wide.
"""

from __future__ import annotations

from cyclonedds.idl._machinery import KeyEnabled, PlainCdrV2SequenceOfPrimitiveMachine
from cyclonedds.idl._support import SerializeKind

_orig_serialize = PlainCdrV2SequenceOfPrimitiveMachine.serialize


def _serialize_bytes_fast(  # noqa: ANN001
    self,
    buffer,
    value,
    serialize_kind=SerializeKind.DataSample,
    key_enabled=KeyEnabled.InKeylist,
):
    if self.size == 1 and isinstance(value, (bytes, bytearray, memoryview)):
        if self.max_length is not None and len(value) > self.max_length:
            raise ValueError(
                f"sequence longer than bound: {len(value)} > {self.max_length}"
            )
        buffer.align(4)
        buffer.write("I", 4, len(value))
        if value:
            buffer.write_bytes(bytes(value))
        return None
    return _orig_serialize(self, buffer, value, serialize_kind, key_enabled)


def apply() -> None:
    """Install the byte-sequence fast path (idempotent)."""
    machine = PlainCdrV2SequenceOfPrimitiveMachine
    machine.serialize = _serialize_bytes_fast  # ty: ignore[invalid-assignment]
