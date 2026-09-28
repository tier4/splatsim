from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors, quat_to_rotation_matrix
from splatsim.actor_assets import ActorAssetLibrary
from splatsim.background import Background
from splatsim.dataclass import SceneConfig
from splatsim.lod import LodIndex, LodManager
from splatsim.renderer import Renderer
from splatsim.rigid_body import RigidBody

if TYPE_CHECKING:
    from splatsim.cyclonedds.camera_info_publisher import CameraInfoPublisher
    from splatsim.cyclonedds.image_publisher import ImagePublisher
    from splatsim.ppisp import PpispTables
    from splatsim.viewer import Viewer


class Scene:
    """Manages background and rigid bodies that compose a renderable scene.

    Rigid bodies are accessible by name for external pose manipulation::

        scene = Scene.from_config(cfg)
        scene["car_01"].set_pose((10.0, 0.0, -5.0))
    """

    def __init__(
        self,
        background: Background | None = None,
        rigid_bodies: dict[str, RigidBody] | None = None,
        lod_manager: LodManager | None = None,
        ppisp_tables: "PpispTables | None" = None,
        actor_library: "ActorAssetLibrary | None" = None,
    ) -> None:
        self.background = background
        self._rigid_bodies: dict[str, RigidBody] = rigid_bodies or {}
        self._lod_manager = lod_manager
        self._lod_enabled = lod_manager is not None
        self.ppisp_tables = ppisp_tables
        self._actor_library = actor_library
        # An optional equirect sky panorama (see _usdz.load_skybox); the Renderer
        # samples it by ray direction behind the Gaussians.
        self.skybox: Tensor | None = None

    # --- rigid body access ---------------------------------------------------

    def __getitem__(self, name: str) -> RigidBody:
        return self._rigid_bodies[name]

    def __contains__(self, name: str) -> bool:
        return name in self._rigid_bodies

    @property
    def rigid_bodies(self) -> dict[str, RigidBody]:
        return self._rigid_bodies

    @property
    def rigid_body_list(self) -> list[RigidBody]:
        """The bodies in the scene, as a snapshot.

        Gathers and the viewer iterate this rather than the live dict, so a
        body added or removed while one is running lands in the next frame
        instead of raising mid-iteration.
        """
        return list(self._rigid_bodies.values())

    def add_rigid_body(self, name: str, rigid_body: RigidBody) -> None:
        """Add a body to the scene, refusing one a gather could not pack.

        A gather concatenates the colour block of every source it collects
        (see :func:`splatsim.renderer._pack`), so a body whose colours are
        shaped differently from what the scene already holds — an SH object in
        a scene loaded without SH, or two different SH degrees — would abort a
        render thread mid-frame with a bare ``torch.cat`` error. Every spawn
        path funnels through here, which makes this the one place to turn that
        into something the caller can still act on.
        """
        mismatch = self._colors_mismatch(rigid_body)
        if mismatch is not None:
            raise ValueError(f"{name}: {mismatch}")
        self._rigid_bodies[name] = rigid_body

    def _colors_mismatch(self, rigid_body: RigidBody) -> str | None:
        """Why *rigid_body* could not be concatenated with this scene, if so."""
        if self.background is not None:
            held = self.background.tensors
        elif self._rigid_bodies:
            held = next(iter(self._rigid_bodies.values())).base_tensors
        else:
            return None
        colors = rigid_body.base_tensors.colors
        if held.colors.shape[1:] == colors.shape[1:]:
            return None
        return (
            f"colours {tuple(colors.shape[1:])} (SH degree "
            f"{rigid_body.base_tensors.sh_degree}) cannot be rendered together "
            f"with the scene's {tuple(held.colors.shape[1:])} (SH degree "
            f"{held.sh_degree})"
        )

    @property
    def actor_library(self) -> ActorAssetLibrary | None:
        """The background bundle's rigid actor assets, if it ships any.

        ``None`` until something loads it — :meth:`from_config` does so when
        the config lists actors, and :meth:`ensure_actor_library` (which
        :meth:`spawn_actor` goes through) on demand.
        """
        return self._actor_library

    def ensure_actor_library(self) -> ActorAssetLibrary:
        """Return the bundle's actor asset bank, loading it on first use.

        :meth:`spawn_actor` goes through here; so does anything that only wants
        to *look* at what the bundle ships (the gRPC ``ListActorAssets``
        handler) without spawning anything. Raises when there is no background
        USDZ to read a bank from, or when that bundle carries none.
        """
        if self._actor_library is None:
            if self.background is None:
                raise ValueError(
                    "no actor asset bank loaded and no background USDZ to load one from"
                )
            self._actor_library = ActorAssetLibrary(
                self.background.source_path,
                device=self.background.tile_local_centroid.device,
                use_sh=self.background.use_sh,
                lod_manager=self._lod_manager,
            )
        return self._actor_library

    def spawn_actor(
        self,
        asset_id: str,
        name: str,
        *,
        position: tuple[float, float, float] | Tensor | None = None,
        rotation: tuple[float, float, float, float] | Tensor | None = None,
        world_position: bool = True,
    ) -> RigidBody:
        """Add a rigid dynamic object from the bundle's actor asset bank.

        The scenario owns the pose from here on: call
        :meth:`set_pose` with ``name`` each frame. ``position`` is read in the
        bundle's ENU world frame unless ``world_position`` is ``False``;
        ``rotation`` is ``wxyz``.
        """
        if name in self._rigid_bodies:
            raise ValueError(f"a rigid body named {name!r} is already in the scene")
        body = self.ensure_actor_library().spawn(
            asset_id,
            position=position,
            rotation=rotation,
            background=self.background if world_position else None,
        )
        self.add_rigid_body(name, body)
        return body

    def remove_rigid_body(self, name: str) -> RigidBody:
        return self._rigid_bodies.pop(name)

    # --- pose helpers --------------------------------------------------------

    def set_pose(
        self,
        name: str,
        position: tuple[float, float, float] | Tensor,
        rotation: tuple[float, float, float, float] | Tensor | None = None,
    ) -> None:
        """Set the pose of a rigid body by name."""
        self._rigid_bodies[name].set_pose(position, rotation)

    # --- LOD-aware tensor collection -----------------------------------------

    @property
    def lod_enabled(self) -> bool:
        return self._lod_enabled

    @lod_enabled.setter
    def lod_enabled(self, value: bool) -> None:
        self._lod_enabled = value and self._lod_manager is not None

    @property
    def lod_manager(self) -> LodManager | None:
        return self._lod_manager

    def collect_tensors(
        self,
        camera_position: Tensor | None = None,
        lod_count_scale: float = 1.0,
        lidar_view: bool = False,
        lod_max_distance: float | None = None,
    ) -> list[GaussianTensors]:
        """Collect Gaussian tensors from all sources, applying LOD if enabled.

        Args:
            camera_position: ``[3]`` float32 GPU tensor (or ``[S, 3]`` for a
                rig — see :meth:`LodManager.filter`), or None to skip LOD.
            lod_count_scale: Extra per-cell LOD decimation (``<1`` thins every
                cell further; forwarded to :meth:`LodManager.filter`). Used by
                the LiDAR renderer to cap Gaussian count on 360° scenes that the
                camera-tuned tiers leave too dense. ``1.0`` = camera behaviour.
            lidar_view: Forwarded to :meth:`LodManager.filter`: fuse the static
                ``lidar_mask`` into the LOD gather and skip the (LiDAR-unused)
                colors block. Only affects sources that go through the LOD
                filter; unfiltered sources come back unchanged and the caller's
                own ``lidar_mask`` handling still applies to them.
            lod_max_distance: Forwarded to :meth:`LodManager.filter` as
                ``max_distance`` — whole cells provably beyond this camera
                distance are dropped before the gather (see there for the
                exactness argument).
        """
        result: list[GaussianTensors] = []
        can_filter = (
            self._lod_enabled
            and self._lod_manager is not None
            and camera_position is not None
        )

        if self.background is not None:
            if can_filter and self.background.lod_index is not None:
                assert self._lod_manager is not None  # noqa: S101
                assert camera_position is not None  # noqa: S101
                tensors = self._lod_manager.filter(
                    self.background.tensors,
                    self.background.lod_index,
                    camera_position,
                    count_scale=lod_count_scale,
                    lidar_view=lidar_view,
                    max_distance=lod_max_distance,
                )
            else:
                tensors = self.background.tensors
            result.append(tensors)

        for rb in self.rigid_body_list:  # a snapshot; see the property
            if can_filter and rb.lod_index is not None:
                assert self._lod_manager is not None  # noqa: S101
                assert camera_position is not None  # noqa: S101
                # Transform camera position into the rigid body's local frame
                # so that octree cell distances are computed correctly.
                rot_mat = quat_to_rotation_matrix(rb.rotation)  # [3, 3]
                # Works for a single [3] position and for a rig's [S, 3].
                cam_local = (camera_position.reshape(-1, 3) - rb.position) @ rot_mat
                if camera_position.dim() == 1:
                    cam_local = cam_local.reshape(3)
                base = self._lod_manager.filter(
                    rb.base_tensors,
                    rb.lod_index,
                    cam_local,
                    count_scale=lod_count_scale,
                    lidar_view=lidar_view,
                    max_distance=lod_max_distance,
                )
                tensors = rb.posed(base)
            else:
                tensors = rb.tensors
            result.append(tensors)

        return result

    # --- construction --------------------------------------------------------

    @staticmethod
    def from_config(
        config: SceneConfig | str | Path,
        *,
        device: torch.device | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> Scene:
        """Build a Scene from a SceneConfig or a scene USDZ path.

        Args:
            progress: Optional callback ``(step, total, label)`` invoked
                after each major loading stage completes.
        """
        if not isinstance(config, SceneConfig):
            config = SceneConfig.from_source(config)

        if device is None:
            device = torch.device(config.renderer.device)

        has_bg = config.background_usdz is not None
        total = int(has_bg) + len(config.rigid_bodies) + len(config.actors)
        step = 0

        lod_manager: LodManager | None = None
        if config.lod.enabled:
            lod_manager = LodManager(config.lod)

        background: Background | None = None
        ppisp_tables: PpispTables | None = None
        if config.background_usdz is not None:
            background = Background(
                config.background_usdz,
                device=device,
                use_sh=config.use_sh,
                lod_manager=lod_manager,
            )
            step += 1
            if progress is not None:
                progress(step, total, "background")
            if background.lod_index is not None:
                _log_lod_tiers("background", background.lod_index)
            if config.renderer.use_ppisp:
                from splatsim._usdz import load_rig_trajectories
                from splatsim.ppisp import load_ppisp_tables

                ppisp_tables = load_ppisp_tables(
                    config.background_usdz,
                    load_rig_trajectories(config.background_usdz),
                    device=device,
                    centroid=background.tile_local_centroid,
                )

        if config.actors and background is None:
            raise ValueError(
                "scene config lists actors but has no background_usdz to load "
                "the actor asset bank from"
            )

        rigid_bodies: dict[str, RigidBody] = {}
        for rb_cfg in config.rigid_bodies:
            rb = RigidBody(
                rb_cfg.source,
                device=device,
                use_sh=rb_cfg.use_sh,
                lod_manager=lod_manager,
            )
            rb.set_pose(rb_cfg.position, rb_cfg.rotation)
            rigid_bodies[rb_cfg.name] = rb
            step += 1
            if progress is not None:
                progress(step, total, rb_cfg.name)
            if rb.lod_index is not None:
                _log_lod_tiers(rb_cfg.name, rb.lod_index)

        scene = Scene(
            background=background,
            rigid_bodies=rigid_bodies,
            lod_manager=lod_manager,
            ppisp_tables=ppisp_tables,
        )
        if config.background_usdz is not None:
            from splatsim._usdz import load_skybox

            scene.skybox = load_skybox(config.background_usdz, device)
            if scene.skybox is not None:
                h, w = int(scene.skybox.shape[0]), int(scene.skybox.shape[1])
                if progress is not None:
                    progress(step, total, f"skybox {w}x{h}")

        # Configured actors go through the same entry point a scenario uses, so
        # the bank is loaded once, on first spawn, and the name-collision rule
        # has one definition.
        for actor_cfg in config.actors:
            scene.spawn_actor(
                actor_cfg.asset_id,
                actor_cfg.name,
                position=actor_cfg.position,
                rotation=actor_cfg.rotation,
                world_position=actor_cfg.world_position,
            )
            step += 1
            if progress is not None:
                progress(step, total, actor_cfg.name)

        return scene


def _log_lod_tiers(name: str, lod_index: LodIndex) -> None:
    """Log the Gaussian distribution across LOD tiers."""
    total_n = max(lod_index.tier_counts) if lod_index.tier_counts else 0

    mode = "octree" if lod_index.cell_centers is not None else "centroid"
    lines = [f"  LOD tiers for '{name}' (total: {total_n:,} Gaussians, mode: {mode}):"]
    for i, (count, max_d) in enumerate(
        zip(lod_index.tier_counts, lod_index.tier_max_distances)
    ):
        pct = 100.0 * count / total_n if total_n > 0 else 0.0
        dist_str = f"{max_d:.0f}m" if max_d < float("inf") else "inf"
        lines.append(
            f"    Tier {i}: {count:>10,} Gaussians ({pct:5.1f}%) | max_distance={dist_str}"
        )

    if lod_index.cell_centers is not None and lod_index.cell_ranges is not None:
        num_cells = lod_index.cell_centers.shape[0]
        cell_counts = lod_index.cell_ranges[:, 1] - lod_index.cell_ranges[:, 0]
        lines.append(
            f"    Octree: {num_cells} cells "
            f"(min={cell_counts.min().item():,}, "
            f"max={cell_counts.max().item():,}, "
            f"mean={cell_counts.float().mean().item():,.0f} Gaussians/cell)"
        )

    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def print_progress(step: int, total: int, label: str) -> None:
    """Print a terminal progress bar to *stderr*."""
    width = 40
    filled = int(width * step / total) if total > 0 else width
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = 100.0 * step / total if total > 0 else 100.0
    sys.stderr.write(f"\r  Loading: |{bar}| {pct:5.1f}% ({label})")
    if step >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def resolve_initial_pose(
    config: SceneConfig,
    background: Background | None,
    *,
    override_position: tuple[float, float, float] | None = None,
    override_yaw_deg: float | None = None,
) -> tuple[tuple[float, float, float] | None, float | None]:
    """Resolve the initial camera pose, in the viewer's tile-local frame.

    Precedence (highest first): explicit override args, then
    ``viewer.initial_position``/``initial_yaw_deg`` (already tile-local),
    then ``initial_camera_world_position``/``initial_camera_yaw_deg`` from
    a scene USDZ (converted to tile-local by subtracting
    ``background.tile_local_centroid``).
    """
    vc = config.viewer
    if override_position is not None:
        position: tuple[float, float, float] | None = override_position
    elif vc.initial_position is not None:
        position = vc.initial_position
    elif config.initial_camera_world_position is not None and background is not None:
        centroid = (
            background.tile_local_centroid.detach().cpu().numpy().astype(np.float64)
        )
        world = np.asarray(config.initial_camera_world_position, dtype=np.float64)
        local = world - centroid
        position = (float(local[0]), float(local[1]), float(local[2]))
    else:
        position = None

    if override_yaw_deg is not None:
        yaw_deg: float | None = override_yaw_deg
    elif vc.initial_yaw_deg is not None:
        yaw_deg = vc.initial_yaw_deg
    elif config.initial_camera_yaw_deg is not None:
        yaw_deg = config.initial_camera_yaw_deg
    else:
        yaw_deg = None

    return position, yaw_deg


def load_scene(
    config: SceneConfig | str | Path,
    *,
    image_publisher: ImagePublisher | None = None,
    camera_info_publisher: CameraInfoPublisher | None = None,
    camera_name: str | None = None,
) -> Viewer:
    """Build a Viewer from a SceneConfig or a scene USDZ path.

    ``camera_name`` selects which PPISP camera profile the Viewer emulates
    when the scene has a PPISP payload; pass the ``name`` of one of the
    training cameras (see ``rig_trajectories.json``). ``None`` skips
    PPISP even when the scene has tables (falls back to the exposure
    scalar).
    """
    from splatsim.viewer import Viewer

    if not isinstance(config, SceneConfig):
        config = SceneConfig.from_source(config)

    device = torch.device(config.renderer.device)
    scene = Scene.from_config(config, device=device, progress=print_progress)

    rc = config.renderer
    renderer = Renderer(
        width=rc.width,
        height=rc.height,
        device=device,
        background_color=rc.background_color,
        near_plane=rc.near_plane,
        far_plane=rc.far_plane,
        radius_clip=rc.radius_clip,
        exposure=rc.exposure,
        ppisp_knn_k=rc.ppisp_knn_k,
    )

    vc = config.viewer
    initial_position, initial_yaw_deg = resolve_initial_pose(config, scene.background)
    return Viewer(
        renderer,
        scene=scene,
        fov_y_deg=vc.fov_y_deg,
        move_speed=vc.move_speed,
        rotate_speed=vc.rotate_speed,
        initial_position=initial_position,
        initial_yaw_deg=initial_yaw_deg,
        image_publisher=image_publisher,
        camera_info_publisher=camera_info_publisher,
        camera_name=camera_name,
    )
