"""Per-Gaussian view-dependent (SH) raydrop (3dgs_io >= v1.2.0).

Covers the consumer wiring for the version-2 ``EXT_gaussian_lidar`` sidecar's
``raydrop_sh`` trailing block (the higher-order SH bands; the DC term stays in
the scalar ``lidar_raydrop_logit``):

* :class:`GaussianTensors` carries an optional ``raydrop_sh`` ``[N, coefs]`` that
  survives slicing (:meth:`__getitem__`) and rigid transforms
  (:func:`apply_rigid_transform`).
* The scene loader (:func:`_usdz.load_spz_scene`) restores ``raydrop_sh`` from a
  version-2 sidecar and leaves it ``None`` for version-1 sidecars;
  :func:`_usdz._concat_tensors` applies the same all-or-nothing policy across
  chunks as the other LiDAR attributes.
* :func:`lidar_renderer._eval_view_dependent_raydrop` reproduces the scalar logit
  at degree 0 and adds view dependence at higher degrees; the renderer wires the
  per-group bands into a single scene tensor (padding scalar-only groups with
  zeros) and evaluates them per beam (CUDA-gated).
"""

from __future__ import annotations

import importlib
import json
import zipfile
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from splatsim import _usdz
from splatsim._conversions import GaussianTensors, apply_rigid_transform
from splatsim.lidar_renderer import (
    _eval_view_dependent_raydrop,
    _raydrop_sh_degree_from_coefs,
)
from splatsim.scene import Scene

_3dgs_io = importlib.import_module("3dgs_io")
FRAME_CONVENTION = _3dgs_io.FRAME_CONVENTION


# ── CPU: degree <-> coefs helper ────────────────────────────────────


def test_raydrop_sh_degree_from_coefs() -> None:
    assert _raydrop_sh_degree_from_coefs(3) == 1  # (1+1)^2 - 1
    assert _raydrop_sh_degree_from_coefs(8) == 2  # (2+1)^2 - 1
    assert _raydrop_sh_degree_from_coefs(15) == 3  # (3+1)^2 - 1


def test_raydrop_sh_degree_from_coefs_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="not .* for any integer degree"):
        _raydrop_sh_degree_from_coefs(4)


# ── CPU: dataclass propagation ──────────────────────────────────────


def _tensors(n: int, *, sh=None) -> GaussianTensors:
    return GaussianTensors(
        means=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        scales=torch.ones((n, 3)),
        opacities=torch.ones(n),
        colors=torch.zeros((n, 3)),
        sh_degree=0,
        raydrop_sh=sh,
    )


def test_raydrop_sh_defaults_to_none() -> None:
    assert _tensors(3).raydrop_sh is None


def test_getitem_indexes_raydrop_sh() -> None:
    sh = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    sliced = _tensors(4, sh=sh)[torch.tensor([0, 2])]
    assert sliced.raydrop_sh is not None
    assert torch.equal(sliced.raydrop_sh, sh[torch.tensor([0, 2])])


def test_getitem_leaves_none_raydrop_sh_none() -> None:
    assert _tensors(4)[torch.tensor([0, 2])].raydrop_sh is None


def test_apply_rigid_transform_propagates_raydrop_sh() -> None:
    sh = torch.randn(3, 3)
    out = apply_rigid_transform(
        _tensors(3, sh=sh),
        position=torch.zeros(3),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    assert out.raydrop_sh is not None
    assert torch.equal(out.raydrop_sh, sh)


# ── CPU: _concat_tensors all-or-nothing ─────────────────────────────


def test_concat_keeps_raydrop_sh_when_every_chunk_has_it() -> None:
    a = _tensors(2, sh=torch.zeros(2, 3))
    b = _tensors(3, sh=torch.ones(3, 3))
    out = _usdz._concat_tensors([a, b])
    assert out.raydrop_sh is not None
    assert out.raydrop_sh.shape == (5, 3)


def test_concat_drops_raydrop_sh_when_any_chunk_missing_it() -> None:
    a = _tensors(2, sh=torch.zeros(2, 3))
    b = _tensors(3)  # no raydrop_sh
    assert _usdz._concat_tensors([a, b]).raydrop_sh is None


def test_concat_raydrop_sh_none_when_no_chunk_has_it() -> None:
    assert _usdz._concat_tensors([_tensors(2), _tensors(3)]).raydrop_sh is None


# ── CPU: loader restores the sidecar SH bands ───────────────────────


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


def test_loader_restores_raydrop_sh(tmp_path, monkeypatch) -> None:
    sh = np.random.randn(3, 3).astype(np.float32)
    usdz_path = _write_scene(
        tmp_path,
        {
            "lidar_intensity_raw": np.array([0.2, 0.7, 0.1], dtype=np.float32),
            "lidar_raydrop_logit": np.array([-1.0, -2.0, -3.0], dtype=np.float32),
            "raydrop_sh": sh,
        },
        count=3,
    )
    _patch_loader(monkeypatch, count=3)

    tensors, _anchor = _usdz.load_spz_scene(usdz_path, torch.device("cpu"))

    assert tensors.raydrop_sh is not None
    assert tensors.raydrop_sh.shape == (3, 3)
    # float16 round-trip in the sidecar → compare with a loose tolerance.
    assert np.allclose(tensors.raydrop_sh.cpu().numpy(), sh, atol=1e-2)


def test_loader_leaves_raydrop_sh_none_for_version1_sidecar(
    tmp_path, monkeypatch
) -> None:
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

    assert tensors.raydrop_sh is None


# ── CPU: per-group SH concatenation + padding ───────────────────────


def _concat(sh_list, counts):
    # Exercise the padding logic without constructing a CUDA renderer.
    from splatsim.lidar_renderer import LidarRenderer

    return LidarRenderer._concat_raydrop_sh(sh_list, counts, torch.device("cpu"))


def test_concat_raydrop_sh_scalar_only_when_no_group_has_bands() -> None:
    assert _concat([None, None], [2, 3]) is None


def test_concat_raydrop_sh_pads_scalar_only_groups_with_zeros() -> None:
    bands = torch.ones(2, 3)
    out = _concat([bands, None], [2, 3])
    assert out is not None and out.shape == (5, 3)
    # First group keeps its bands; the scalar-only group is zero-padded.
    assert torch.equal(out[:2], bands)
    assert torch.count_nonzero(out[2:]) == 0


def test_concat_raydrop_sh_rejects_mixed_widths() -> None:
    with pytest.raises(ValueError, match="Mixed raydrop_sh"):
        _concat([torch.ones(2, 3), torch.ones(1, 8)], [2, 1])


# ── CUDA: SH evaluation ─────────────────────────────────────────────

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class _FakeScene:
    """Minimal stand-in exposing only what ``LidarRenderer.render`` needs."""

    def __init__(self, tensors: GaussianTensors) -> None:
        self._tensors = tensors

    def collect_tensors(self, _cam_pos, **_kwargs):  # noqa: D401 - signature match
        # LOD / lidar-view kwargs (lod_count_scale, lidar_view, lod_max_distance)
        # are accepted and ignored: this fake has no LOD index to filter.
        return [self._tensors]


@cuda
def test_eval_no_bands_returns_scalar_unchanged() -> None:
    means = torch.randn(5, 3, device="cuda")
    logit = torch.randn(5, device="cuda")
    out = _eval_view_dependent_raydrop(means, means[0], logit, None)
    assert torch.equal(out, logit)


@cuda
def test_eval_zero_bands_reproduces_scalar() -> None:
    # With all higher bands zero, the SH evaluation must collapse to the scalar
    # DC logit (the backward-compatibility guarantee) at any view direction.
    means = torch.randn(6, 3, device="cuda")
    logit = torch.randn(6, device="cuda")
    sh = torch.zeros(6, 3, device="cuda")  # degree 1, all-zero higher bands
    out = _eval_view_dependent_raydrop(means, torch.zeros(3, device="cuda"), logit, sh)
    assert torch.allclose(out, logit, atol=1e-4)


@cuda
def test_eval_is_view_dependent() -> None:
    # A non-zero band-1 coefficient must make the drop logit depend on the ray
    # direction: the same Gaussian evaluated from opposite sides differs.
    mean = torch.tensor([[0.0, 5.0, 0.0]], device="cuda")
    logit = torch.zeros(1, device="cuda")
    sh = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")  # band-1 (m=-1, y) active
    front = _eval_view_dependent_raydrop(
        mean, torch.tensor([0.0, -5.0, 0.0], device="cuda"), logit, sh
    )
    back = _eval_view_dependent_raydrop(
        mean, torch.tensor([0.0, 15.0, 0.0], device="cuda"), logit, sh
    )
    assert not torch.allclose(front, back, atol=1e-3)
    # Band-0 basis is 1/SH_C0 * SH_C0 = 1, so the average across opposite rays
    # stays near the scalar (0 here) — the higher band is a pure view delta.
    assert abs(float(front + back) / 2.0) < 1.0


@cuda
def test_renderer_all_zero_bands_matches_scalar_render() -> None:
    """A degree>0 scene whose higher bands are all zero renders identically to
    the scalar-raydrop path (end-to-end backward compatibility)."""
    from splatsim.lidar_renderer import LidarRenderer, LidarSensorSpec

    # A handful of well-separated Gaussians straight ahead (+x) so every one is a
    # clean return; kept small so the test leaves no GPU state that could amplify
    # gsplat's atomic-add non-determinism in later tests.
    n = 8
    means = torch.zeros(n, 3, device="cuda")
    means[:, 0] = torch.linspace(6.0, 30.0, n, device="cuda")
    means[:, 2] = torch.linspace(-1.0, 1.0, n, device="cuda")

    def _gauss(sh):
        return GaussianTensors(
            means=means,
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").repeat(n, 1),
            scales=torch.full((n, 3), 0.3, device="cuda"),
            opacities=torch.full((n,), 0.99, device="cuda"),
            colors=torch.zeros((n, 3), device="cuda"),
            sh_degree=0,
            intensity_raw=torch.full((n,), 1.0, device="cuda"),
            raydrop_logit=torch.full((n,), -1.5, device="cuda"),
            raydrop_sh=sh,
        )

    spec = LidarSensorSpec(name="t", sensor_type="XT32", s2b=np.eye(4), n_columns=512)
    r = LidarRenderer(spec, device="cuda")
    eye = torch.eye(4, device="cuda")

    # _FakeScene duck-types the single method render() calls; cast for the checker.
    scalar = r.render(eye, scene=cast(Scene, _FakeScene(_gauss(None))))
    zeros = r.render(
        eye, scene=cast(Scene, _FakeScene(_gauss(torch.zeros(n, 3, device="cuda"))))
    )

    both = (scalar["alpha"] > 0.1) & (zeros["alpha"] > 0.1)
    assert both.any()
    d = (scalar["raydrop_logit"][both] - zeros["raydrop_logit"][both]).abs()
    assert d.max().item() < 1e-3
