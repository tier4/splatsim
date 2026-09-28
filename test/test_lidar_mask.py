"""Per-Gaussian ``lidar_mask`` sidecar channel (3dgs_io >= v1.1.0).

Covers the consumer wiring end to end:

* :class:`GaussianTensors` carries an optional boolean ``lidar_mask`` that
  survives slicing (:meth:`__getitem__`) and rigid transforms
  (:func:`apply_rigid_transform`).
* The scene loader (:func:`_usdz.load_spz_scene`) restores the mask from the
  sidecar when present and leaves it ``None`` for old 2-channel sidecars, and
  :func:`_usdz._concat_tensors` applies the same all-or-nothing policy across
  chunks as the other LiDAR attributes.
* :class:`LidarRenderer` hard-excludes ``mask == False`` Gaussians from the
  LiDAR geometry pass (CUDA-gated).
"""

from __future__ import annotations

import importlib
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from splatsim import _usdz
from splatsim._conversions import GaussianTensors, apply_rigid_transform

_3dgs_io = importlib.import_module("3dgs_io")
FRAME_CONVENTION = _3dgs_io.FRAME_CONVENTION


# ── CPU: dataclass propagation ──────────────────────────────────────


def _tensors(n: int, *, mask=None) -> GaussianTensors:
    return GaussianTensors(
        means=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        scales=torch.ones((n, 3)),
        opacities=torch.ones(n),
        colors=torch.zeros((n, 3)),
        sh_degree=0,
        lidar_mask=mask,
    )


def test_lidar_mask_defaults_to_none() -> None:
    assert _tensors(3).lidar_mask is None


def test_getitem_indexes_lidar_mask() -> None:
    mask = torch.tensor([True, False, True, False])
    sliced = _tensors(4, mask=mask)[torch.tensor([0, 1])]
    assert sliced.lidar_mask is not None
    assert sliced.lidar_mask.tolist() == [True, False]


def test_getitem_leaves_none_mask_none() -> None:
    assert _tensors(4)[torch.tensor([0, 2])].lidar_mask is None


def test_apply_rigid_transform_propagates_mask() -> None:
    mask = torch.tensor([True, False, True])
    out = apply_rigid_transform(
        _tensors(3, mask=mask),
        position=torch.zeros(3),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    assert out.lidar_mask is not None
    assert torch.equal(out.lidar_mask, mask)


# ── CPU: _concat_tensors all-or-nothing ─────────────────────────────


def test_concat_keeps_mask_when_every_chunk_has_it() -> None:
    a = _tensors(2, mask=torch.tensor([True, False]))
    b = _tensors(3, mask=torch.tensor([False, True, True]))
    out = _usdz._concat_tensors([a, b])
    assert out.lidar_mask is not None
    assert out.lidar_mask.dtype == torch.bool
    assert out.lidar_mask.tolist() == [True, False, False, True, True]


def test_concat_drops_mask_when_any_chunk_missing_it() -> None:
    a = _tensors(2, mask=torch.tensor([True, False]))
    b = _tensors(3)  # no mask
    assert _usdz._concat_tensors([a, b]).lidar_mask is None


def test_concat_mask_none_when_no_chunk_has_it() -> None:
    assert _usdz._concat_tensors([_tensors(2), _tensors(3)]).lidar_mask is None


# ── CPU: loader restores the sidecar mask ───────────────────────────


def _scene_doc() -> dict:
    return {
        "schema": "splatsim.scene/v2",
        "world": {
            "frame_convention": FRAME_CONVENTION,
            "ecef_anchor": np.eye(4).tolist(),
        },
        "gaussians": {
            "frame": "world",
            "tileset": "tileset.json",
            "ext_attributes": {
                "extension": "EXT_gaussian_lidar",
                "sidecar_suffix": ".lidar",
                "attributes": ["lidar_intensity_raw", "lidar_raydrop_logit"],
            },
        },
    }


def _write_scene(tmp_path, attrs: dict, count: int):
    usdz_path = tmp_path / "scene.usdz"
    sidecar = _3dgs_io.encode_lidar_extension(attrs, count=count)
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr("scene.json", json.dumps(_scene_doc()))
        zf.writestr("chunks/chunk_000000.spz", b"spz")
        zf.writestr("chunks/chunk_000000.lidar", sidecar)
    return usdz_path


def _patch_loader(monkeypatch, count: int) -> None:
    monkeypatch.setattr(
        _usdz, "_load_spz", lambda _path: SimpleNamespace(num_points=count)
    )
    monkeypatch.setattr(
        _usdz,
        "cloud_to_tensors",
        lambda _cloud, device, *, use_sh: GaussianTensors(
            means=torch.zeros((count, 3), device=device),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(count, 1),
            scales=torch.ones((count, 3), device=device),
            opacities=torch.ones(count, device=device),
            colors=torch.zeros((count, 3), device=device),
            sh_degree=0,
        ),
    )


def test_loader_restores_lidar_mask(tmp_path, monkeypatch) -> None:
    usdz_path = _write_scene(
        tmp_path,
        {
            "lidar_intensity_raw": np.array([0.2, 0.7, 0.1], dtype=np.float32),
            "lidar_raydrop_logit": np.array([-1.0, -2.0, -3.0], dtype=np.float32),
            "lidar_mask": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        },
        count=3,
    )
    _patch_loader(monkeypatch, count=3)

    tensors, _anchor = _usdz.load_spz_scene(usdz_path, torch.device("cpu"))

    assert tensors.lidar_mask is not None
    assert tensors.lidar_mask.dtype == torch.bool
    assert tensors.lidar_mask.shape == (3,)
    assert tensors.lidar_mask.tolist() == [True, False, True]


def test_loader_leaves_mask_none_for_two_channel_sidecar(tmp_path, monkeypatch) -> None:
    usdz_path = _write_scene(
        tmp_path,
        {
            "lidar_intensity_raw": np.array([0.2], dtype=np.float32),
            "lidar_raydrop_logit": np.array([-1.0], dtype=np.float32),
        },
        count=1,
    )
    _patch_loader(monkeypatch, count=1)

    tensors, _anchor = _usdz.load_spz_scene(usdz_path, torch.device("cpu"))

    assert tensors.lidar_mask is None


# ── CUDA: renderer hard-excludes masked Gaussians ───────────────────

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class _FakeScene:
    """Minimal stand-in exposing only what ``LidarRenderer.render`` needs."""

    def __init__(self, tensors: GaussianTensors) -> None:
        self._tensors = tensors

    def collect_tensors(  # noqa: D401 - signature match
        self,
        _cam_pos,
        lod_count_scale: float = 1.0,
        lidar_view: bool = False,
        lod_max_distance: float | None = None,
    ):
        return [self._tensors]


def _gauss(means, *, mask=None) -> GaussianTensors:
    n = means.shape[0]
    return GaussianTensors(
        means=means,
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=means.device).repeat(n, 1),
        scales=torch.full((n, 3), 0.3, device=means.device),
        opacities=torch.full((n,), 0.99, device=means.device),
        colors=torch.zeros((n, 3), device=means.device),
        sh_degree=0,
        intensity_raw=torch.full((n,), 1.0, device=means.device),
        raydrop_logit=torch.full((n,), -6.0, device=means.device),
        lidar_mask=mask,
    )


def _renderer(**kw):
    from splatsim.lidar_renderer import LidarRenderer, LidarSensorSpec

    spec = LidarSensorSpec(name="t", sensor_type="XT32", s2b=np.eye(4), n_columns=512)
    return LidarRenderer(spec, device="cuda", **kw)


@cuda
def test_all_true_mask_matches_none() -> None:
    # gsplat's rasterizer accumulates with atomic adds, so it is only
    # deterministic up to a handful of near-tangent boundary cells run to
    # run. Use a dense scene and compare hit-mask IoU + shared-cell distance
    # (the same tolerance strategy as the rolling-shutter test) so an all-True
    # mask — which keeps every Gaussian, exactly like ``lidar_mask=None`` — is
    # verified to leave the render unchanged.
    g = torch.Generator(device="cuda").manual_seed(0)
    means = (torch.rand(2000, 3, device="cuda", generator=g) - 0.5) * 40.0
    means[:, 0] = torch.rand(2000, device="cuda", generator=g) * 30.0 + 5.0
    eye = torch.eye(4, device="cuda")
    r = _renderer()

    base = r.render(eye, scene=_FakeScene(_gauss(means, mask=None)))
    all_true = r.render(
        eye,
        scene=_FakeScene(
            _gauss(means, mask=torch.ones(2000, dtype=torch.bool, device="cuda"))
        ),
    )

    hb = base["alpha"] > 0.1
    ht = all_true["alpha"] > 0.1
    assert hb.any(), "sanity: baseline should have returns"
    iou = (hb & ht).sum().item() / max((hb | ht).sum().item(), 1)
    assert iou > 0.995, f"all-True mask changed the hit mask (IoU {iou:.4f})"
    both = hb & ht
    d = (base["distance"][both] - all_true["distance"][both]).abs()
    assert d.mean().item() < 0.01


@cuda
def test_all_false_mask_yields_empty_output() -> None:
    means = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], device="cuda")
    out = _renderer().render(
        torch.eye(4, device="cuda"),
        scene=_FakeScene(
            _gauss(means, mask=torch.zeros(2, dtype=torch.bool, device="cuda"))
        ),
    )
    assert torch.count_nonzero(out["alpha"]) == 0
    assert torch.count_nonzero(out["distance"]) == 0


@cuda
def test_masked_gaussian_does_not_contribute() -> None:
    """Masking a well-separated Gaussian removes its returns; the kept one stays."""
    # G0 straight ahead (+x, azimuth 0), G1 to the left (+y, azimuth +pi/2).
    means = torch.tensor([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0]], device="cuda")
    eye = torch.eye(4, device="cuda")
    r = _renderer()

    both = r.render(eye, scene=_FakeScene(_gauss(means, mask=None)))
    keep_g0 = r.render(
        eye,
        scene=_FakeScene(
            _gauss(means, mask=torch.tensor([True, False], device="cuda"))
        ),
    )

    hit_both = both["alpha"] > 0.1
    hit_g0 = keep_g0["alpha"] > 0.1

    # Excluding G1 removes returns overall.
    assert hit_g0.sum() < hit_both.sum()

    # Column bands: columns sweep +pi -> -pi across width W. G1 lives near
    # azimuth +pi/2 -> ~W/4; G0 near azimuth 0 -> ~W/2.
    w = both["alpha"].shape[1]
    g1_band = slice(int(0.20 * w), int(0.30 * w))
    g0_band = slice(int(0.45 * w), int(0.55 * w))
    # G1's band had returns unmasked and loses ALL of them once masked out.
    # This is robust to gsplat's boundary-cell jitter: G1 is simply absent
    # from the renderer input, so it cannot reappear anywhere.
    assert hit_both[:, g1_band].any()
    assert not hit_g0[:, g1_band].any()
    # G0's band (the kept Gaussian) still produces returns and is essentially
    # unchanged by masking G1.
    assert hit_g0[:, g0_band].any()
    assert hit_both[:, g0_band].any()
