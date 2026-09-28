"""Rigid dynamic-object ("actor") assets read from a scene USDZ.

A scene bundle's ``sequence_tracks.json`` says where every dynamic object is at
every moment; ``actor_assets.json`` (3dgs_io's ``splatsim.actor_assets/v1``,
from v2.1.0) says what those objects *look like*. This module is splatsim's
reader for that bank.

Each asset is a Gaussian cloud authored in the canonical object-local frame —
``+x`` forward, ``+y`` left, ``+z`` up, origin at the centre of the object's
bounding box, metric scale — in the same NGSP v4 SPZ container the background
chunks use, carrying the same optional per-Gaussian LiDAR extension record. So
an actor's Gaussians go through :func:`~splatsim._conversions.cloud_to_tensors`
and :func:`~splatsim._conversions.attach_lidar_attrs` exactly like a chunk
does; nothing about an actor is a special kind of Gaussian. That holds on the
LiDAR path too: intensity, raydrop and the participation mask reach the LiDAR
renderer through the same fields a background chunk uses, and posing an actor
re-expresses its view-dependent ``raydrop_sh`` in the world frame the same way
it does its colour SH.

Poses come from outside
-----------------------

The library hands out :class:`~splatsim.rigid_body.RigidBody` instances and
nothing more: where an actor goes each frame is the caller's business — a CARLA
bridge, a scenario runner, the gRPC service. The bundle's own tracks are
metadata here, exposed through :meth:`ActorAssetLibrary.bound_track_ids` for
callers that want to replay them, not a playback engine.

Spawning is cheap. Every instance of one asset shares the same base tensors on
the device; only the per-frame posed copy is allocated, so fifty of the same
sedan cost one upload.

Tile-local frame
----------------

:class:`~splatsim.background.Background` re-centres the scene by its Gaussian
centroid for numerical stability, so a pose expressed in the bundle's ENU world
frame (a track pose, a map waypoint) must have that centroid subtracted before
it reaches a rigid body. :func:`world_to_tile_local` does that conversion, and
:meth:`ActorAssetLibrary.spawn` takes the pose in whichever frame the caller
names.
"""

from __future__ import annotations

import importlib as _importlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from splatsim._conversions import (
    GaussianTensors,
    attach_lidar_attrs,
    cloud_to_tensors,
)
from splatsim._geometry import quat_xyzw_to_wxyz
from splatsim._usdz import read_scene_json
from splatsim.lod import LodManager
from splatsim.rigid_body import RigidBody

_3dgs_io = _importlib.import_module("3dgs_io")
_parse_actor_assets = _3dgs_io.parse_actor_assets
_decode_actor_asset = _3dgs_io.decode_actor_asset

ACTOR_ASSETS_SCENE_KEY = "actor_assets"


class TileLocalFrame(Protocol):
    """Anything that knows the scene's tile-local recentring offset.

    :class:`~splatsim.background.Background` is the one that matters — it
    subtracts its Gaussian centroid from the cloud on load — but only that one
    attribute is needed to place an actor, so the conversion does not drag the
    whole background in.
    """

    @property
    def tile_local_centroid(self) -> Tensor: ...


@dataclass(frozen=True)
class ActorAssetInfo:
    """What the bank's index says about one asset, before it is spawned."""

    asset_id: str
    class_name: str
    size: tuple[float, float, float]
    """Declared box ``(dx, dy, dz)`` in metres, along object ``+x/+y/+z``."""
    n_points: int
    sh_degree: int
    has_lidar_attributes: bool


def world_to_tile_local(
    position: tuple[float, float, float] | Tensor,
    background: TileLocalFrame,
) -> Tensor:
    """Convert an ENU world-frame position into the renderer's tile-local frame.

    Track poses, map waypoints and anything else read out of a scene bundle are
    world-frame; :class:`~splatsim.background.Background` subtracts its Gaussian
    centroid from the cloud, so a body posed with raw world coordinates lands
    hundreds of metres from the scene. Subtract the same centroid here.
    """
    centroid = background.tile_local_centroid
    if not isinstance(position, Tensor):
        position = torch.tensor(position, device=centroid.device, dtype=torch.float32)
    return position.to(device=centroid.device, dtype=torch.float32) - centroid


def pose_from_track_frame(
    frame: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a scene-bundle pose into the ``(position, wxyz)`` pair splatsim uses.

    Every pose inside a scene bundle — ``TrackFrame``, ``RigPose``, sensor
    extrinsics — carries an **xyzw** quaternion, while
    :meth:`~splatsim.rigid_body.RigidBody.set_pose` and every splatsim config
    take **wxyz**. Reordering by hand is the easy way to end up with an actor
    that faces a plausible but wrong direction, so route bundle poses through
    here. The translation stays in the bundle's ENU world frame — pass it to
    :meth:`ActorAssetLibrary.spawn` with a ``background`` (or through
    :func:`world_to_tile_local`) to reach the renderer's frame.

    Accepts anything with ``translation`` and ``rotation`` attributes.
    """
    tx, ty, tz = (float(v) for v in frame.translation)
    qw, qx, qy, qz = quat_xyzw_to_wxyz(frame.rotation)
    return (tx, ty, tz), (qw, qx, qy, qz)


class ActorAssetLibrary:
    """The rigid actor assets bundled in one scene USDZ, ready to spawn.

    Load once per scene, then :meth:`spawn` as many instances as the scenario
    needs. Instances share their base tensors, so spawning is a pose and a
    dictionary entry, not another upload.
    """

    def __init__(
        self,
        source_path: str | Path,
        *,
        device: torch.device = torch.device("cuda"),
        use_sh: bool = False,
        lod_manager: LodManager | None = None,
    ) -> None:
        path = Path(source_path)
        meta = read_scene_json(path)
        uri = (meta.get("extras") or {}).get(ACTOR_ASSETS_SCENE_KEY)
        if not uri:
            raise ValueError(
                f"{path}: no actor asset bank "
                f"(scene.json.extras.{ACTOR_ASSETS_SCENE_KEY} is unset)"
            )

        self._path = path
        self._device = device
        self._lod_manager = lod_manager
        self._infos: dict[str, ActorAssetInfo] = {}
        self._base: dict[str, GaussianTensors] = {}
        self._bound_tracks: dict[str, list[str]] = {}

        with zipfile.ZipFile(path) as zf:
            bank = _parse_actor_assets(json.loads(zf.read(uri)))
            for asset in bank.assets:
                cloud, attrs = _decode_actor_asset(asset, zf.read(str(asset.uri)))
                tensors = cloud_to_tensors(cloud, device, use_sh=use_sh)
                if attrs:
                    attach_lidar_attrs(
                        tensors,
                        attrs,
                        cloud.num_points,
                        device,
                        source=f"{path}: {asset.uri}",
                    )
                self._base[asset.asset_id] = tensors
                self._infos[asset.asset_id] = ActorAssetInfo(
                    asset_id=asset.asset_id,
                    class_name=asset.class_name,
                    size=tuple(
                        float(v) for v in asset.size
                    ),  # ty: ignore[invalid-argument-type]
                    n_points=int(asset.n_points),
                    sh_degree=int(asset.sh_degree),
                    has_lidar_attributes=asset.ext_attributes is not None,
                )
        for instance in bank.instances:
            self._bound_tracks.setdefault(instance.asset_id, []).append(
                instance.track_id
            )

    # --- inspection ----------------------------------------------------------

    @property
    def asset_ids(self) -> list[str]:
        return list(self._base)

    def info(self, asset_id: str) -> ActorAssetInfo:
        """Index metadata for one asset (class, box size, point count)."""
        self._require(asset_id)
        return self._infos[asset_id]

    def bound_track_ids(self, asset_id: str) -> list[str]:
        """Tracks the bundle binds to this asset, in the order recorded.

        Informational: this library does not drive poses from tracks. A caller
        replaying a recorded scenario can use it to decide what to spawn.
        """
        self._require(asset_id)
        return list(self._bound_tracks.get(asset_id, ()))

    # --- spawning ------------------------------------------------------------

    def spawn(
        self,
        asset_id: str,
        *,
        position: tuple[float, float, float] | Tensor | None = None,
        rotation: tuple[float, float, float, float] | Tensor | None = None,
        background: TileLocalFrame | None = None,
    ) -> RigidBody:
        """Instantiate ``asset_id`` as a posable rigid body.

        ``rotation`` is a ``wxyz`` quaternion, matching
        :meth:`~splatsim.rigid_body.RigidBody.set_pose` — note that scene-bundle
        poses (``TrackFrame.rotation``, rig poses) are ``xyzw`` and must be
        reordered by the caller.

        Pass ``background`` to give ``position`` in the bundle's ENU world frame;
        it is converted to the renderer's tile-local frame through
        :func:`world_to_tile_local`. Without it, ``position`` is taken as
        already tile-local.

        The returned body re-expresses its colour SH when posed, because a
        moving actor's heading changes every frame — see
        :func:`splatsim._conversions.apply_rigid_transform`.
        """
        self._require(asset_id)
        body = RigidBody(
            self._base[asset_id],
            device=self._device,
            lod_manager=self._lod_manager,
            rotate_sh=True,
        )
        if position is not None or rotation is not None:
            if position is None:
                position = body.position
            elif background is not None:
                position = world_to_tile_local(position, background)
            body.set_pose(position, rotation)
        return body

    def _require(self, asset_id: str) -> None:
        if asset_id not in self._base:
            raise KeyError(
                f"{self._path}: no actor asset {asset_id!r}; "
                f"available: {sorted(self._base)}"
            )
