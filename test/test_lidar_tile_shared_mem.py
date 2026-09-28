"""The LiDAR tile shape must fit the device's per-block shared-memory cap.

The SplatAD rasterizer stages ``tile_width * tile_height * LIDAR_BATCH_MULT``
Gaussians in dynamic shared memory; above the per-block opt-in limit the launch
fails deep in CUDA with a bare "Failed to set maximum shared memory size". The
guard in ``lidar_renderer`` surfaces that here, before the launch, with the
numbers and the fix. The classic tripwire is the pre-tuning 4x64 default, which
needs 128 KB at BATCH_MULT=16 — over the ~99 KB an sm_86/sm_89 block allows.
"""

from __future__ import annotations

import pytest
import torch

from splatsim.lidar_renderer import (
    _assert_tile_fits_shared_mem,
    _LIDAR_BATCH_MULT,
    _tile_shared_mem_bytes,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_shared_mem_matches_kernel_formula() -> None:
    # static path: two float4 per staged Gaussian; velocity path adds a float2.
    assert _tile_shared_mem_bytes(16, 1, False) == 16 * 1 * _LIDAR_BATCH_MULT * 32
    assert _tile_shared_mem_bytes(16, 1, True) == 16 * 1 * _LIDAR_BATCH_MULT * 40
    # the 4x64 tripwire is exactly 128 KB at the shipped BATCH_MULT.
    assert _tile_shared_mem_bytes(64, 4, False) == 131072


@cuda
def test_shipped_tiling_fits() -> None:
    # 1x16 (shipped) and the 4090-optimal 1x24 must pass on any supported GPU.
    _assert_tile_fits_shared_mem.cache_clear()
    _assert_tile_fits_shared_mem(16, 1, False, 0)
    _assert_tile_fits_shared_mem(24, 1, False, 0)
    _assert_tile_fits_shared_mem(16, 1, True, 0)  # rolling shutter


@cuda
def test_oversized_tiling_is_rejected_with_a_useful_message() -> None:
    cap = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    if _tile_shared_mem_bytes(64, 4, False) <= cap:
        pytest.skip("this device's shared-memory cap is large enough for 4x64")
    _assert_tile_fits_shared_mem.cache_clear()
    with pytest.raises(ValueError) as ei:
        _assert_tile_fits_shared_mem(64, 4, False, 0)
    msg = str(ei.value)
    assert "4x64" in msg  # tile_height x tile_width
    assert "shared memory" in msg
    assert "KB" in msg  # reports the requirement and the cap
