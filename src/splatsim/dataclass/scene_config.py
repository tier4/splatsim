from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from splatsim._geometry import quat_xyzw_to_wxyz
from splatsim.dataclass.actor_config import ActorConfig
from splatsim.dataclass.lod_config import LodConfig
from splatsim.dataclass.lidar_config import LidarConfig
from splatsim.dataclass.renderer_config import RendererConfig
from splatsim.dataclass.rigid_body_config import RigidBodyConfig
from splatsim.dataclass.viewer_config import ViewerConfig


def _lidar_sensors_from_rigs(rigs) -> list[LidarConfig]:
    """Build :class:`LidarConfig` entries from a scene USDZ's rig calibrations.

    3dgs_io stores each LiDAR pose as sensor-in-rig: ``translation`` is the
    mount position in the ego/base frame directly (no inversion needed), and
    ``rotation`` is an ``xyzw`` unit quaternion — reordered here to the
    ``wxyz`` form :func:`build_lidar_sensors_from_config` expects. The
    intrinsics (row/column counts, spin rate, range, and the per-beam
    ``elevation_deg`` table) live in the free-form ``lidar_model.parameters``.
    """
    sensors: list[LidarConfig] = []
    for rig in rigs:
        for cal in getattr(rig, "lidars", None) or []:
            ext = cal.extrinsics
            tx, ty, tz = (float(v) for v in ext.translation)
            qw, qx, qy, qz = quat_xyzw_to_wxyz(ext.rotation)
            model = getattr(cal, "lidar_model", None)
            params = dict(model.parameters) if model is not None else {}
            elevation = params.get("elevation_deg")
            sensors.append(
                LidarConfig(
                    name=cal.name,
                    # Geometry is driven by the explicit elevation table below,
                    # so no built-in named table (OT128/XT32) is assumed here.
                    sensor_type="",
                    n_rows=int(params.get("n_rows", 128)),
                    n_columns=int(params.get("n_columns", 2048)),
                    fps=float(params.get("fps", 10.0)),
                    min_range_m=float(params.get("min_range_m", 0.3)),
                    max_range_m=float(params.get("max_range_m", 120.0)),
                    position=(tx, ty, tz),
                    rotation=(qw, qx, qy, qz),
                    elevation_deg=(
                        tuple(float(e) for e in elevation) if elevation else None
                    ),
                    pointcloud_topic=f"/sensing/lidar/{cal.name}/pointcloud",
                    frame_id=cal.name,
                )
            )
    return sensors


@dataclass
class SceneConfig:
    """Top-level scene configuration loaded from YAML."""

    background_usdz: str | None = None
    use_sh: bool = True
    rigid_bodies: list[RigidBodyConfig] = field(default_factory=list)
    # Rigid dynamic objects spawned from the background bundle's actor asset
    # bank (3dgs_io splatsim.actor_assets/v1). Empty by default: loading a
    # scene does not place actors, a scenario does. See splatsim.actor_assets.
    actors: list[ActorConfig] = field(default_factory=list)
    lidar_sensors: list[LidarConfig] = field(default_factory=list)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    lod: LodConfig = field(default_factory=LodConfig)
    # World-frame camera pose seeded from a scene USDZ. Translated into the
    # viewer's tile-local frame after the background is loaded (see
    # :func:`splatsim.scene.resolve_initial_pose`).
    initial_camera_world_position: tuple[float, float, float] | None = None
    initial_camera_yaw_deg: float | None = None

    @staticmethod
    def from_source(
        path: str | Path,
        *,
        camera_name: str | None = None,
        lod_enabled: bool | None = None,
    ) -> SceneConfig:
        """Build a SceneConfig from a scene USDZ.

        ``camera_name`` selects which rig camera seeds intrinsics and initial
        pose.

        ``lod_enabled`` overrides ``cfg.lod.enabled`` after loading when not
        ``None``. Pass ``True``/``False`` from the CLI to force LoD on/off
        regardless of the file's default.
        """
        path = Path(path)
        if path.suffix.lower() != ".usdz":
            raise ValueError(
                f"{path}: unsupported scene source; only a scene USDZ (.usdz) "
                "is supported"
            )
        cfg = SceneConfig.from_usdz(path, camera_name=camera_name)
        if lod_enabled is not None:
            cfg.lod.enabled = lod_enabled
        return cfg

    @staticmethod
    def from_usdz(path: str | Path, *, camera_name: str | None = None) -> SceneConfig:
        """Build a SceneConfig from a scene USDZ's embedded ``scene.json``.

        Only metadata is read here; the heavy SPZ chunks are loaded later
        by :class:`Background` when it sees the same ``.usdz`` path.

        If the USDZ ships a ``rig_trajectories.json`` sidecar containing
        cameras, the most forward-facing camera's intrinsics seed
        ``renderer.width/height`` and ``viewer.fov_y_deg``, and the
        composed ``RigPose × CameraExtrinsics`` at the first timestamp
        seeds ``initial_camera_world_position`` and
        ``initial_camera_yaw_deg``.
        """
        from splatsim._usdz import (
            camera_to_viewer_intrinsics,
            first_camera,
            initial_camera_pose_from_rig_trajectories,
            read_rig_trajectories,
            read_scene_json,
        )

        path = Path(path)
        meta = read_scene_json(path)

        rd = meta.get("render_defaults", {})
        renderer = RendererConfig(
            near_plane=rd.get("near_plane", RendererConfig.near_plane),
            far_plane=rd.get("far_plane", 60.0),
            exposure=rd.get("exposure", 1.0),
            # Match gaussian_factory's RGB reference render: discard
            # sub-pixel splats instead of accumulating their low-contribution
            # tails. USDZ scenes do not currently serialize this option.
            radius_clip=1.0,
        )
        viewer = ViewerConfig()
        initial_pos: tuple[float, float, float] | None = None
        initial_yaw: float | None = None
        lidar_sensors: list[LidarConfig] = []

        rig_uri = meta.get("extras", {}).get("rig_trajectories")
        if rig_uri:
            rigs = read_rig_trajectories(path, rig_uri)
            cam = first_camera(rigs, name=camera_name)
            if cam is not None:
                width, height, fov_y_deg = camera_to_viewer_intrinsics(cam)
                if width and height:
                    renderer.width = width
                    renderer.height = height
                if fov_y_deg is not None:
                    viewer.fov_y_deg = fov_y_deg

            pose = initial_camera_pose_from_rig_trajectories(rigs, name=camera_name)
            if pose is not None:
                initial_pos, initial_yaw = pose

            lidar_sensors = _lidar_sensors_from_rigs(rigs)

        return SceneConfig(
            background_usdz=str(path),
            use_sh=True,
            rigid_bodies=[],
            actors=[],
            lidar_sensors=lidar_sensors,
            renderer=renderer,
            viewer=viewer,
            lod=LodConfig(),
            initial_camera_world_position=initial_pos,
            initial_camera_yaw_deg=initial_yaw,
        )
