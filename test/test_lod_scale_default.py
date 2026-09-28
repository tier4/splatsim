"""The LiDAR LOD thinning default is a measured trade-off, so pin it.

0.25 was chosen because LiDAR rays stop at a median ~11 m: the first surface a
ray crosses is decided by a cell's most important Gaussians, so halving again
from 0.5 costs 1.6% of returns while nearly doubling throughput. What degrades
is the RETURN RATE, not range accuracy -- hence the separate assertion that the
knob still reaches 1.0 (keep everything) for fidelity-critical runs.
"""

from __future__ import annotations

import pytest

from splatsim.lidar_renderer import _lidar_lod_scale


def test_default_is_the_measured_choice() -> None:
    assert _lidar_lod_scale() == 0.25


def test_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLATSIM_LIDAR_LOD_SCALE", "1.0")
    assert _lidar_lod_scale() == 1.0
    monkeypatch.setenv("SPLATSIM_LIDAR_LOD_SCALE", "0.1")
    assert _lidar_lod_scale() == 0.1


def test_scale_is_a_fraction() -> None:
    """A scale above 1 would ask for more Gaussians than a cell holds."""
    assert 0.0 < _lidar_lod_scale() <= 1.0
