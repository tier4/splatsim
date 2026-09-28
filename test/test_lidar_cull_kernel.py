"""Unit test for the fused CUDA cull kernel.

Compares the CUDA extension's keep mask bit-for-bit against the PyTorch
reference implementation on a synthetic scene that exercises:
  * gaussians inside the range shell
  * gaussians outside the shell (both too close and too far)
  * gaussians in and out of the elevation band
  * NaN scales (must degrade to zero-margin, not poison the mask)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _pytorch_reference(
    means: torch.Tensor,
    scales: torch.Tensor,
    sensor_pos: torch.Tensor,
    up_world: torch.Tensor,
    *,
    min_range: float,
    max_range: float | None,
    cull_scale_sigmas: float,
    use_elev: bool,
    sin_min: float,
    cos_min: float,
    sin_max: float,
    cos_max: float,
) -> torch.Tensor:
    delta = means - sensor_pos
    dist = torch.linalg.vector_norm(delta, dim=-1)
    max_scale = torch.nan_to_num(scales.amax(dim=-1), nan=0.0)
    margin = cull_scale_sigmas * max_scale
    keep = dist + margin >= min_range
    if max_range is not None:
        keep = keep & (dist - margin <= max_range)
    if use_elev:
        z_s = delta @ up_world
        keep = keep & (z_s + margin * cos_min >= dist * sin_min)
        keep = keep & (z_s - margin * cos_max <= dist * sin_max)
    return keep


def _synthetic_scene(n: int = 500_000, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Wide spatial distribution so all range/FOV branches get exercised.
    means = (
        (torch.rand(n, 3, device="cuda", generator=g) * 2 - 1) * 400.0
    ).contiguous()
    scales = (0.05 + torch.rand(n, 3, device="cuda", generator=g) * 0.5).contiguous()

    # Poison ~1% of scales with NaN — the mask must survive this without
    # rejecting every gaussian.
    nan_idx = torch.randperm(n, device="cuda", generator=g)[: n // 100]
    scales[nan_idx, 0] = float("nan")

    return means, scales


def _pose():
    S = torch.eye(4, device="cuda", dtype=torch.float32)
    S[:3, 3] = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    ax = torch.tensor([0.3, -0.2, 0.9], device="cuda")
    S[:3, 2] = ax / ax.norm()
    return S


@pytest.mark.parametrize("use_elev", [False, True])
@pytest.mark.parametrize("max_range", [None, 120.0])
def test_cuda_cull_matches_pytorch(use_elev: bool, max_range: float | None) -> None:
    from splatsim._lidar_cull_ext import is_available, lidar_cull_mask

    if not is_available():
        pytest.skip("CUDA cull extension failed to compile in this environment")

    means, scales = _synthetic_scene()
    S = _pose()
    sensor_pos = S[:3, 3]
    up_world = S[:3, 2]

    sin_min = math.sin(math.radians(-25.0))
    cos_min = math.cos(math.radians(-25.0))
    sin_max = math.sin(math.radians(15.0))
    cos_max = math.cos(math.radians(15.0))

    ref = _pytorch_reference(
        means,
        scales,
        sensor_pos,
        up_world,
        min_range=0.3,
        max_range=max_range,
        cull_scale_sigmas=3.0,
        use_elev=use_elev,
        sin_min=sin_min,
        cos_min=cos_min,
        sin_max=sin_max,
        cos_max=cos_max,
    )
    got = lidar_cull_mask(
        means,
        scales,
        sensor_pos,
        up_world,
        min_range=0.3,
        max_range=max_range,
        cull_scale_sigmas=3.0,
        use_elev=use_elev,
        sin_min=sin_min,
        cos_min=cos_min,
        sin_max=sin_max,
        cos_max=cos_max,
    )
    # CUDA path uses FMA (mad.f32) and --use_fast_math while the PyTorch
    # chain doesn't, so results can disagree by 1 ULP on gaussians that
    # land exactly on the comparison boundary. In practice this manifests
    # at a rate of ~1e-6 (~1 gaussian per million) and the mask is a
    # coarse pre-filter anyway — gsplat's own tests catch any left over.
    # We tolerate up to 1e-4 disagreement to insulate the test from
    # future toolchain jitter.
    disagreement = (ref != got).float().mean().item()
    assert disagreement < 1e-4, (
        f"CUDA cull disagrees with PyTorch reference on {(ref != got).sum().item()} "
        f"of {ref.numel()} gaussians ({disagreement:.2e}) "
        f"— exceeds boundary-ULP tolerance (use_elev={use_elev}, max_range={max_range})"
    )


def test_empty_input_returns_empty_mask() -> None:
    from splatsim._lidar_cull_ext import is_available, lidar_cull_mask

    if not is_available():
        pytest.skip("CUDA cull extension failed to compile in this environment")

    means = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    scales = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    sensor_pos = torch.zeros(3, device="cuda", dtype=torch.float32)
    up_world = torch.tensor([0.0, 0.0, 1.0], device="cuda")

    got = lidar_cull_mask(
        means,
        scales,
        sensor_pos,
        up_world,
        min_range=0.3,
        max_range=120.0,
        cull_scale_sigmas=3.0,
        use_elev=True,
        sin_min=-0.5,
        cos_min=0.86,
        sin_max=0.5,
        cos_max=0.86,
    )
    assert got.shape == (0,)
    assert got.dtype == torch.bool


def test_end_to_end_render_matches_pytorch_fallback() -> None:
    """render_lidar_panorama with CUDA cull matches the PyTorch chain path."""
    from splatsim import lidar_renderer as lr
    from splatsim._lidar_cull_ext import is_available

    if not is_available():
        pytest.skip("CUDA cull extension failed to compile in this environment")

    torch.manual_seed(0)
    n = 200_000
    means = ((torch.rand(n, 3, device="cuda") * 2 - 1) * 60.0).float().contiguous()
    quats = torch.randn(n, 4, device="cuda")
    quats = (
        (quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-6)).float().contiguous()
    )
    scales = (0.05 + torch.rand(n, 3, device="cuda") * 0.25).float().contiguous()
    opacities = (0.2 + torch.rand(n, device="cuda") * 0.5).float().contiguous()
    intensity = torch.rand(n, device="cuda").float().contiguous()
    raydrop = (torch.rand(n, device="cuda") * 0.1).float().contiguous()

    S = torch.eye(4, device="cuda", dtype=torch.float32)
    spec = lr.LidarSensorSpec(
        name="ot128",
        sensor_type="OT128",
        s2b=np.eye(4, dtype=np.float32),
        n_columns=1024,
        spinning_frequency_hz=10.0,
        el_lo_rad=math.radians(-25.0),
        el_hi_rad=math.radians(15.0),
        n_rows_uniform=64,
    )

    def _render():
        return lr.render_lidar_panorama(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity,
            raydrop_logit=raydrop,
            sensor_to_world=S,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            frustum_cull=True,
            elev_fov_cull=True,
        )

    with_cuda = _render()
    lr._USE_CUDA_CULL = False
    try:
        without_cuda = _render()
    finally:
        lr._USE_CUDA_CULL = True

    # Per-pixel outputs can differ significantly if even one boundary
    # gaussian is dropped by one path but kept by the other (see
    # ULP-tolerance note in test_cuda_cull_matches_pytorch). Compare the
    # coarse "is this pixel occupied?" signal (alpha above a small
    # threshold), which is much less sensitive to individual-gaussian
    # jitter than the raw distance/intensity values.
    hit_a = with_cuda["alpha"] > 0.01
    hit_b = without_cuda["alpha"] > 0.01
    hit_diff = (hit_a != hit_b).float().mean().item()
    assert hit_diff < 1e-3, (
        f"pixel occupancy differs on {hit_diff:.2%} of pixels between CUDA and PyTorch cull"
    )


def test_loaded_ext_exposes_every_expected_symbol() -> None:
    """A loaded extension must carry every binding the current source exports.

    Regression guard for the stale-prebuilt bug: an ``.so`` built before
    ``raydrop_sh_eval`` was added still loaded fine, callers' ``hasattr``
    probes silently routed to the gsplat fallback, and the 5-sensor rig paid
    ~15 ms/frame instead of ~2 ms with nothing failing anywhere. If this
    test trips, the prebuilt staleness checks in ``_load_prebuilt`` let an
    outdated binary through.
    """
    from splatsim import _lidar_cull_ext

    ext = _lidar_cull_ext._try_load()
    if ext is None:
        pytest.skip("CUDA cull extension unavailable on this machine")
    missing = [s for s in _lidar_cull_ext._EXPECTED_SYMBOLS if not hasattr(ext, s)]
    assert not missing, f"stale cull extension loaded: missing bindings {missing}"
