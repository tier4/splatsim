try:
    import cyclonedds as _cyclonedds  # noqa: F401
except ImportError:
    raise ImportError(
        "splatsim.cyclonedds requires CycloneDDS.\n"
        "Install with: pip install splatsim[dds]"
    ) from None

from splatsim.cyclonedds import _fast_cdr
from splatsim.cyclonedds.camera_info_publisher import CameraInfoPublisher
from splatsim.cyclonedds.image_publisher import ImagePublisher
from splatsim.cyclonedds.pointcloud2_publisher import PointCloud2Publisher

# Serialize byte payloads (PointCloud2 / Image data) with a memcpy instead of
# cyclonedds' per-element pack; see _fast_cdr for the measurements.
_fast_cdr.apply()

__all__ = [
    "CameraInfoPublisher",
    "ImagePublisher",
    "PointCloud2Publisher",
]
