"""Tests for reading rigid dynamic-object ("actor") assets out of a scene USDZ.

The bank (3dgs_io ``splatsim.actor_assets/v1``) carries object-local Gaussian
clouds; splatsim turns them into posable :class:`RigidBody` instances that a
scenario drives from outside. What is worth defending here is what a consumer
cannot recover on its own: that an asset's Gaussians and its per-Gaussian LiDAR
attributes come through the same readers a background chunk uses, that a
world-frame pose lands in the renderer's tile-local frame, and that posing an
actor re-expresses its view-dependent colour instead of spinning the highlights
with the car.
"""

from __future__ import annotations

import importlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from splatsim._conversions import (
    MAX_NON_YAW_RAD,
    GaussianTensors,
    apply_rigid_transform,
    rotate_raydrop_sh_about_z,
    rotate_sh_about_z,
)
from splatsim.actor_assets import (
    ActorAssetLibrary,
    pose_from_track_frame,
    world_to_tile_local,
)
from splatsim.dataclass import LodConfig
from splatsim.lod import LodManager
from splatsim.rigid_body import RigidBody
from splatsim.scene import Scene

_3dgs_io = importlib.import_module("3dgs_io")
_spz = importlib.import_module("spz")

FRAME_CONVENTION = _3dgs_io.FRAME_CONVENTION
ActorAssetSource = _3dgs_io.ActorAssetSource
ActorInstance = _3dgs_io.ActorInstance
build_actor_asset_bank = _3dgs_io.build_actor_asset_bank
serialize_actor_assets = _3dgs_io.serialize_actor_assets

CPU = torch.device("cpu")
CAR_SIZE = (4.5, 1.9, 1.5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _object_local_cloud(n: int = 32, *, seed: int = 0, sh_degree: int = 0):
    """A car-shaped cloud in the object frame (centred on the origin)."""
    rng = np.random.default_rng(seed)
    gc = _spz.GaussianCloud()  # ty: ignore[unresolved-attribute]
    gc.antialiased = False
    box = (rng.random((n, 3)) - 0.5) * np.array(CAR_SIZE)
    gc.positions = box.astype(np.float32).reshape(-1)
    quats = rng.standard_normal((n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.astype(np.float32).reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.0, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    per_ch = (sh_degree + 1) ** 2 - 1
    if per_ch:
        gc.sh_degree = sh_degree
        gc.sh = rng.standard_normal(n * per_ch * 3).astype(np.float32)
    else:
        gc.sh_degree = 0
        gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def _source(
    asset_id: str = "sedan_0007",
    *,
    n: int = 32,
    seed: int = 0,
    with_lidar: bool = False,
    raydrop_sh_degree: int = 0,
    sh_degree: int = 0,
) -> Any:
    rng = np.random.default_rng(seed + 7)
    ext: dict[str, np.ndarray] = {}
    if with_lidar:
        ext = {
            "lidar_intensity_raw": np.linspace(-1.0, 1.0, n).astype(np.float32),
            "lidar_raydrop_logit": np.full(n, -0.5, dtype=np.float32),
            "lidar_mask": (rng.random(n) > 0.5).astype(np.float32),
        }
        if raydrop_sh_degree:
            coefs = (raydrop_sh_degree + 1) ** 2 - 1
            ext["raydrop_sh"] = rng.standard_normal((n, coefs)).astype(np.float32)
    return ActorAssetSource(
        asset_id=asset_id,
        cloud=_object_local_cloud(n, seed=seed, sh_degree=sh_degree),
        class_name="automobile",
        size=CAR_SIZE,
        ext_attrs=ext,
    )


def _write_scene_usdz(
    path: Path,
    sources: list[Any],
    instances: list[Any] | None = None,
) -> Path:
    """A minimal scene/v3 bundle carrying an actor asset bank."""
    bank, payloads = build_actor_asset_bank(sources, instances or [])
    scene: dict[str, Any] = {
        "schema": "splatsim.scene/v3",
        "world": {
            "frame_convention": FRAME_CONVENTION,
            "ecef_anchor": np.eye(4).tolist(),
        },
        "gaussians": {
            "frame": "world",
            # Placeholder: nothing under test decodes the background chunk —
            # read_scene_json only validates that the index is there.
            "chunks": [{"uri": "chunks/chunk_000000.spz", "n_points": 4}],
        },
        "extras": {"actor_assets": "actor_assets.json"},
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("scene.json", json.dumps(scene))
        zf.writestr("chunks/chunk_000000.spz", b"")
        zf.writestr("actor_assets.json", json.dumps(serialize_actor_assets(bank)))
        for asset in bank.assets:
            zf.writestr(str(asset.uri), payloads[asset.asset_id])
    return path


# ---------------------------------------------------------------------------
# Loading the bank
# ---------------------------------------------------------------------------


def test_library_reads_every_asset_and_its_index_metadata(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(
        tmp_path / "scene.usdz",
        [_source("sedan_0007", n=32), _source("truck_0001", n=16, seed=2)],
    )
    library = ActorAssetLibrary(usdz, device=CPU)

    assert sorted(library.asset_ids) == ["sedan_0007", "truck_0001"]

    info = library.info("sedan_0007")
    assert info.class_name == "automobile"
    assert info.n_points == 32
    assert info.size == pytest.approx(CAR_SIZE)
    assert info.has_lidar_attributes is False


def test_library_reports_an_unknown_asset(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    with pytest.raises(KeyError, match="no actor asset 'missing'"):
        library.spawn("missing")


def test_library_rejects_a_bundle_without_a_bank(tmp_path: Path) -> None:
    """A pre-v2.1.0 bundle has no extras pointer and must say so, not crash."""
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    stripped = tmp_path / "no_bank.usdz"
    with zipfile.ZipFile(usdz) as src, zipfile.ZipFile(stripped, "w") as dst:
        for name in src.namelist():
            if name == "scene.json":
                scene = json.loads(src.read(name)) | {"extras": {}}
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))

    with pytest.raises(ValueError, match="no actor asset bank"):
        ActorAssetLibrary(stripped, device=CPU)


def test_lidar_attributes_come_through_the_background_reader(tmp_path: Path) -> None:
    """An actor's per-Gaussian LiDAR attrs are read exactly like a chunk's."""
    n = 32
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source(n=n, with_lidar=True)])
    library = ActorAssetLibrary(usdz, device=CPU)
    assert library.info("sedan_0007").has_lidar_attributes is True

    body = library.spawn("sedan_0007")
    base = body.base_tensors
    assert base.intensity_raw is not None
    assert base.raydrop_logit is not None
    torch.testing.assert_close(
        base.raydrop_logit, torch.full((n,), -0.5), atol=0.02, rtol=0
    )
    assert base.lidar_mask is not None
    assert base.lidar_mask.dtype == torch.bool


def test_bound_track_ids_are_exposed_but_do_not_drive_poses(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(
        tmp_path / "scene.usdz",
        [_source("sedan_0007")],
        [
            ActorInstance(track_id="100", asset_id="sedan_0007"),
            ActorInstance(track_id="101", asset_id="sedan_0007"),
        ],
    )
    library = ActorAssetLibrary(usdz, device=CPU)
    assert library.bound_track_ids("sedan_0007") == ["100", "101"]
    # Spawning is independent of the bindings: the caller decides.
    assert library.spawn("sedan_0007").position.tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Spawning and posing
# ---------------------------------------------------------------------------


def test_instances_share_their_base_tensors(tmp_path: Path) -> None:
    """Fifty of the same sedan must cost one upload, not fifty."""
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    a = library.spawn("sedan_0007")
    b = library.spawn("sedan_0007")
    assert a is not b
    assert a.base_tensors.means.data_ptr() == b.base_tensors.means.data_ptr()


def test_spawn_applies_the_initial_pose(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    yaw = math.radians(90.0)
    body = library.spawn(
        "sedan_0007",
        position=(10.0, -3.0, 0.75),
        rotation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
    )
    assert body.position.tolist() == pytest.approx([10.0, -3.0, 0.75])

    # A +90 degrees yaw sends the car's own +x to world +y, so the posed means
    # are the object-local ones through R_z(90) plus the translation.
    local = body.base_tensors.means
    turned = local @ torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(
        body.tensors.means - body.position, turned, atol=1e-5, rtol=0
    )


class _FakeBackground:
    """A stand-in for the one attribute :class:`TileLocalFrame` asks for."""

    def __init__(self, centroid: tuple[float, float, float]) -> None:
        self.tile_local_centroid = torch.tensor(centroid, dtype=torch.float32)


def test_world_positions_are_recentred_into_the_tile_local_frame(
    tmp_path: Path,
) -> None:
    """A world-frame pose must have the background's centroid subtracted.

    Without it a car tracked at ENU (113, -58, 1.9) renders hundreds of metres
    from the scene, because Background re-centres its cloud on load.
    """
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    background = _FakeBackground((100.0, -50.0, 1.0))

    world = (113.62, -58.55, 1.92)
    torch.testing.assert_close(
        world_to_tile_local(world, background),
        torch.tensor([13.62, -8.55, 0.92]),
        atol=1e-4,
        rtol=0,
    )

    body = library.spawn("sedan_0007", position=world, background=background)
    torch.testing.assert_close(
        body.position, torch.tensor([13.62, -8.55, 0.92]), atol=1e-4, rtol=0
    )

    # Without a background the position is taken as already tile-local.
    plain = library.spawn("sedan_0007", position=world)
    torch.testing.assert_close(plain.position, torch.tensor(world), atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# View-dependent colour
# ---------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def _sh_basis(d: np.ndarray) -> np.ndarray:
    """The 3DGS real-SH basis, DC first — the layout ``colors`` uses."""
    x, y, z = d
    return np.array(
        [
            _SH_C0,
            _SH_C1 * -y,
            _SH_C1 * z,
            _SH_C1 * -x,
            _SH_C2[0] * x * y,
            _SH_C2[1] * y * z,
            _SH_C2[2] * (2 * z * z - x * x - y * y),
            _SH_C2[3] * x * z,
            _SH_C2[4] * (x * x - y * y),
            _SH_C3[0] * y * (3 * x * x - y * y),
            _SH_C3[1] * x * y * z,
            _SH_C3[2] * y * (4 * z * z - x * x - y * y),
            _SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y),
            _SH_C3[4] * x * (4 * z * z - x * x - y * y),
            _SH_C3[5] * z * (x * x - y * y),
            _SH_C3[6] * x * (x * x - 3 * y * y),
        ]
    )


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, -117.0, 180.0])
def test_sh_z_rotation_is_exact(yaw_deg: float) -> None:
    """``f_world(d) == f_object(R_z(yaw) @ d)`` for every direction."""
    rng = np.random.default_rng(5)
    colors = torch.tensor(rng.standard_normal((2, 16, 3)), dtype=torch.float64)
    yaw = math.radians(yaw_deg)
    rotated = rotate_sh_about_z(colors, torch.tensor(yaw, dtype=torch.float64))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for _ in range(20):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        np.testing.assert_allclose(
            _sh_basis(d) @ rotated[0].numpy(),
            _sh_basis(rot @ d) @ colors[0].numpy(),
            atol=1e-12,
        )


def test_sh_z_rotation_leaves_the_dc_band_alone() -> None:
    """Band 0 is rotation-invariant, so a view-independent asset is untouched."""
    colors = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    rotated = rotate_sh_about_z(colors, torch.tensor(1.234))
    torch.testing.assert_close(rotated[:, 0, :], colors[:, 0, :])
    assert not torch.allclose(rotated[:, 1:, :], colors[:, 1:, :])


def _tensors_with_sh(n: int = 8, k: int = 16) -> GaussianTensors:
    rng = np.random.default_rng(11)
    return GaussianTensors(
        means=torch.tensor(rng.standard_normal((n, 3)), dtype=torch.float32),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
        scales=torch.ones(n, 3),
        opacities=torch.ones(n),
        colors=torch.tensor(rng.standard_normal((n, k, 3)), dtype=torch.float32),
        sh_degree=3,
    )


def test_rigid_transform_rotates_sh_only_when_asked() -> None:
    """Existing rigid bodies keep their behaviour; actors opt in."""
    base = _tensors_with_sh()
    yaw = math.radians(40.0)
    rotation = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    position = torch.zeros(3)

    plain = apply_rigid_transform(base, position, rotation)
    torch.testing.assert_close(plain.colors, base.colors)

    turned = apply_rigid_transform(base, position, rotation, rotate_sh=True)
    assert not torch.allclose(turned.colors, base.colors)
    torch.testing.assert_close(
        turned.colors, rotate_sh_about_z(base.colors, torch.tensor(-yaw))
    )


def test_rgb_mode_is_unaffected_by_rotate_sh() -> None:
    """An asset loaded without SH has [N, 3] colors and nothing to rotate."""
    base = GaussianTensors(
        means=torch.zeros(4, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
        scales=torch.ones(4, 3),
        opacities=torch.ones(4),
        colors=torch.rand(4, 3),
        sh_degree=0,
    )
    out = apply_rigid_transform(
        base, torch.zeros(3), torch.tensor([0.7, 0.0, 0.0, 0.714]), rotate_sh=True
    )
    torch.testing.assert_close(out.colors, base.colors)


def test_spawned_actors_rotate_their_sh(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source(sh_degree=3)])
    library = ActorAssetLibrary(usdz, device=CPU, use_sh=True)
    body = library.spawn("sedan_0007")

    yaw = math.radians(55.0)
    body.set_pose((0.0, 0.0, 0.0), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    assert not torch.allclose(body.tensors.colors, body.base_tensors.colors)


@pytest.mark.parametrize(
    ("rotate_sh", "euler", "expected_tilt_deg"),
    [
        pytest.param(True, ("z", 30.0), 0.0, id="yaw-is-exact"),
        pytest.param(True, ("y", 20.0), 20.0, id="pitch-is-approximate"),
        # Nothing is being approximated when the body does not rotate its SH.
        pytest.param(False, ("y", 20.0), 0.0, id="no-sh-rotation"),
    ],
)
def test_sh_rotation_reports_how_far_a_pose_is_from_a_pure_yaw(
    rotate_sh: bool, euler: tuple[str, float], expected_tilt_deg: float
) -> None:
    body = RigidBody(_tensors_with_sh(), device=CPU, rotate_sh=rotate_sh)
    axis, degrees = euler
    half = math.radians(degrees) / 2
    quat = {
        "z": (math.cos(half), 0.0, 0.0, math.sin(half)),
        "y": (math.cos(half), 0.0, math.sin(half), 0.0),
    }[axis]
    body.set_pose((0.0, 0.0, 0.0), quat)

    assert body.sh_rotation_tilt == pytest.approx(
        math.radians(expected_tilt_deg), abs=1e-5
    )
    assert body.sh_rotation_is_exact == (
        math.radians(expected_tilt_deg) <= MAX_NON_YAW_RAD
    )


# ---------------------------------------------------------------------------
# Bundle poses
# ---------------------------------------------------------------------------


def test_track_poses_are_reordered_from_xyzw_to_wxyz() -> None:
    """Bundle poses are xyzw; splatsim is wxyz. Getting that wrong is silent."""
    frame = _3dgs_io.TrackFrame(
        timestamp_us=1_000_000,
        translation=(113.62, -58.55, 1.92),
        rotation=(0.1, 0.2, 0.3, 0.927),  # xyzw
    )
    position, rotation = pose_from_track_frame(frame)
    assert position == pytest.approx((113.62, -58.55, 1.92))
    assert rotation == pytest.approx((0.927, 0.1, 0.2, 0.3))


def test_spawning_from_a_track_pose_lands_on_the_tracked_box(tmp_path: Path) -> None:
    """The documented path end to end: bundle pose -> wxyz -> tile-local -> body."""
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    background = _FakeBackground((100.0, -50.0, 1.0))

    yaw = math.radians(33.0)
    frame = _3dgs_io.TrackFrame(
        timestamp_us=1_000_000,
        translation=(113.62, -58.55, 1.92),
        rotation=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),  # xyzw
    )
    position, rotation = pose_from_track_frame(frame)
    body = library.spawn(
        "sedan_0007", position=position, rotation=rotation, background=background
    )

    # Undo the pose the track asked for: the gaussians return to the object box.
    c, s = math.cos(yaw), math.sin(yaw)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    back = (body.tensors.means - body.position) @ rot
    assert bool((back.abs() <= torch.tensor(CAR_SIZE) / 2 + 1e-4).all())


# ---------------------------------------------------------------------------
# View-dependent raydrop (the LiDAR half of the same problem)
# ---------------------------------------------------------------------------


def _eval_raydrop(bands: np.ndarray, logit: float, d: np.ndarray) -> float:
    """Evaluate raydrop the way ``lidar_renderer`` packs and evaluates it.

    The renderer puts ``raydrop_logit / SH_C0`` in the DC slot and the bands at
    ``1:k``, then reads channel 0 back out of ``gsplat.spherical_harmonics`` —
    so the bands ride the same 3DGS basis the colours do, one index apart.
    """
    coeffs = np.zeros(len(bands) + 1)
    coeffs[0] = logit / _SH_C0
    coeffs[1:] = bands
    return float(_sh_basis(d)[: len(coeffs)] @ coeffs)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("yaw_deg", [0.0, 47.0, -133.0, 180.0])
def test_raydrop_sh_z_rotation_is_exact(degree: int, yaw_deg: float) -> None:
    """``f_world(d) == f_object(R_z(yaw) @ d)`` for the DC-less raydrop layout."""
    rng = np.random.default_rng(7)
    coefs = (degree + 1) ** 2 - 1
    bands = torch.tensor(rng.standard_normal((1, coefs)), dtype=torch.float64)
    logit = float(rng.standard_normal())
    yaw = math.radians(yaw_deg)
    rotated = rotate_raydrop_sh_about_z(bands, torch.tensor(yaw, dtype=torch.float64))

    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for _ in range(20):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        assert _eval_raydrop(rotated[0].numpy(), logit, d) == pytest.approx(
            _eval_raydrop(bands[0].numpy(), logit, rot @ d), abs=1e-12
        )


def test_raydrop_sh_rotation_rejects_the_colour_layout() -> None:
    with pytest.raises(ValueError, match=r"expects \[N, K\] bands"):
        rotate_raydrop_sh_about_z(torch.zeros(4, 15, 3), torch.tensor(0.5))


def _tensors_with_raydrop(n: int = 6, coefs: int = 15, *, sh_degree: int = 3):
    rng = np.random.default_rng(13)
    colors = (
        torch.tensor(rng.standard_normal((n, 16, 3)), dtype=torch.float32)
        if sh_degree
        else torch.rand(n, 3)
    )
    return GaussianTensors(
        means=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
        scales=torch.ones(n, 3),
        opacities=torch.ones(n),
        colors=colors,
        sh_degree=sh_degree,
        raydrop_logit=torch.tensor(rng.standard_normal(n), dtype=torch.float32),
        raydrop_sh=torch.tensor(rng.standard_normal((n, coefs)), dtype=torch.float32),
    )


def test_rigid_transform_rotates_raydrop_bands_with_the_colours() -> None:
    base = _tensors_with_raydrop()
    yaw = math.radians(40.0)
    rotation = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    plain = apply_rigid_transform(base, torch.zeros(3), rotation)
    torch.testing.assert_close(plain.raydrop_sh, base.raydrop_sh)

    turned = apply_rigid_transform(base, torch.zeros(3), rotation, rotate_sh=True)
    assert base.raydrop_sh is not None
    torch.testing.assert_close(
        turned.raydrop_sh,
        rotate_raydrop_sh_about_z(base.raydrop_sh, torch.tensor(-yaw)),
    )
    # The band-0 drop logit is rotation-invariant, so it must come through as-is.
    torch.testing.assert_close(turned.raydrop_logit, base.raydrop_logit)


def test_raydrop_bands_rotate_even_for_an_rgb_mode_asset() -> None:
    """``raydrop_sh`` does not depend on ``sh_degree`` — an RGB asset can have it."""
    base = _tensors_with_raydrop(sh_degree=0)
    yaw = math.radians(70.0)
    turned = apply_rigid_transform(
        base,
        torch.zeros(3),
        torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
        rotate_sh=True,
    )
    torch.testing.assert_close(turned.colors, base.colors)  # nothing to rotate
    assert turned.raydrop_sh is not None and base.raydrop_sh is not None
    assert not torch.allclose(turned.raydrop_sh, base.raydrop_sh)


def test_spawned_actors_carry_and_rotate_their_raydrop_bands(tmp_path: Path) -> None:
    """End to end: a bank asset's view-dependent raydrop survives and follows the pose."""
    n = 32
    usdz = _write_scene_usdz(
        tmp_path / "scene.usdz",
        [_source(n=n, with_lidar=True, raydrop_sh_degree=2)],
    )
    library = ActorAssetLibrary(usdz, device=CPU)
    body = library.spawn("sedan_0007")

    base = body.base_tensors
    assert base.raydrop_sh is not None
    assert base.raydrop_sh.shape == (n, (2 + 1) ** 2 - 1)

    yaw = math.radians(55.0)
    body.set_pose((0.0, 0.0, 0.0), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    torch.testing.assert_close(
        body.tensors.raydrop_sh,
        rotate_raydrop_sh_about_z(base.raydrop_sh, torch.tensor(-yaw)),
    )


def test_the_lod_lidar_gather_still_hands_back_rotated_raydrop(
    tmp_path: Path,
) -> None:
    """The path the LiDAR renderer actually takes: LOD gather, then pose.

    ``Scene.collect_tensors(lidar_view=True)`` thins the actor's *object-frame*
    Gaussians and then poses the subset, so the raydrop bands must come back
    re-expressed in world — the LOD branch used to be a second place that could
    forget.
    """
    usdz = _write_scene_usdz(
        tmp_path / "scene.usdz", [_source(n=64, with_lidar=True, raydrop_sh_degree=2)]
    )
    manager = LodManager(LodConfig())
    library = ActorAssetLibrary(usdz, device=CPU, lod_manager=manager)
    body = library.spawn("sedan_0007")
    yaw = math.radians(25.0)
    body.set_pose((0.0, 0.0, 0.0), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

    scene = Scene(rigid_bodies={"car_01": body}, lod_manager=manager)
    (posed,) = scene.collect_tensors(torch.zeros(3), lidar_view=True)

    assert posed.raydrop_sh is not None
    base_bands = body.base_tensors.raydrop_sh
    assert base_bands is not None
    # The gather selects a subset in the object frame; whichever rows survive,
    # they must be the rotated ones.
    expected = rotate_raydrop_sh_about_z(base_bands, torch.tensor(-yaw))
    assert posed.raydrop_sh.shape[1] == expected.shape[1]
    for row in posed.raydrop_sh:
        assert bool((expected - row).abs().sum(dim=1).min() < 1e-5)


# ---------------------------------------------------------------------------
# The invariant the band rotation actually has to satisfy
# ---------------------------------------------------------------------------
#
# Asserting that `apply_rigid_transform` agrees with `rotate_*_about_z` only
# proves the two agree — it says nothing about which way round the rotation
# goes, and an inverted one is silently plausible. These two evaluate the bands
# along a real ray instead: a posed actor must answer, in the world frame, what
# the object-frame asset answers along the same ray seen from the object frame.


def _object_frame_ray_setup(yaw_deg: float = 63.0):
    """A posed body plus the sensor position expressed in both frames."""
    yaw = math.radians(yaw_deg)
    rot = Rotation.from_euler("z", yaw)
    translation = torch.tensor([10.0, -4.0, 0.8])
    sensor_world = torch.tensor([30.0, 12.0, 2.0])
    sensor_object = torch.tensor(
        rot.inv().apply((sensor_world - translation).numpy()), dtype=torch.float32
    )
    return yaw, rot, translation, sensor_world, sensor_object


def test_a_posed_actor_drops_rays_the_way_its_object_frame_asset_does() -> None:
    """The LiDAR invariant: world-frame raydrop == object-frame raydrop, same ray."""
    from splatsim.lidar_renderer import _eval_view_dependent_raydrop

    rng = np.random.default_rng(4)
    n, coefs = 40, 8
    base = GaussianTensors(
        means=torch.tensor(rng.standard_normal((n, 3)), dtype=torch.float32),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
        scales=torch.ones(n, 3),
        opacities=torch.ones(n),
        colors=torch.rand(n, 3),
        sh_degree=0,
        raydrop_logit=torch.tensor(rng.standard_normal(n), dtype=torch.float32),
        raydrop_sh=torch.tensor(rng.standard_normal((n, coefs)), dtype=torch.float32),
    )
    yaw, rot, translation, sensor_world, sensor_object = _object_frame_ray_setup()

    body = RigidBody(base, device=CPU, rotate_sh=True)
    body.set_pose(
        tuple(translation.tolist()), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    )
    posed = body.tensors
    assert posed.raydrop_logit is not None and base.raydrop_logit is not None

    got = _eval_view_dependent_raydrop(
        posed.means, sensor_world, posed.raydrop_logit, posed.raydrop_sh
    )
    want = _eval_view_dependent_raydrop(
        base.means, sensor_object, base.raydrop_logit, base.raydrop_sh
    )
    torch.testing.assert_close(got, want, atol=1e-4, rtol=0)

    # And the unrotated bands really would have been wrong — otherwise this
    # test would pass for a no-op implementation.
    unrotated = apply_rigid_transform(base, body.position, body.rotation)
    assert unrotated.raydrop_logit is not None
    wrong = _eval_view_dependent_raydrop(
        unrotated.means, sensor_world, unrotated.raydrop_logit, unrotated.raydrop_sh
    )
    assert float((wrong - want).abs().max()) > 0.5


def test_a_posed_actor_shades_the_way_its_object_frame_asset_does() -> None:
    """The same invariant for colour SH, evaluated along the camera ray."""
    import gsplat

    n = 24
    base = _tensors_with_sh(n=n)
    yaw, rot, translation, eye_world, eye_object = _object_frame_ray_setup(-41.0)

    body = RigidBody(base, device=CPU, rotate_sh=True)
    body.set_pose(
        tuple(translation.tolist()), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    )
    posed = body.tensors

    got = gsplat.spherical_harmonics(3, posed.means - eye_world, posed.colors)
    want = gsplat.spherical_harmonics(3, base.means - eye_object, base.colors)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=0)
