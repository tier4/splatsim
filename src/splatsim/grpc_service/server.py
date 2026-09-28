"""gRPC servicer for the SplatSim rendering service."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import grpc
import torch

from cyclonedds.domain import DomainParticipant

from splatsim._conversions import GaussianTensors
from splatsim.actor_assets import world_to_tile_local
from splatsim.background import Background
from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher
from splatsim.cyclonedds.msg_types import Time
from splatsim.cyclonedds.pointcloud2_publisher import PointCloud2Publisher
from splatsim.dataclass.lidar_config import LidarConfig, sensor_defaults
from splatsim.dataclass.lod_config import LodConfig
from splatsim.lod import LodIndex, LodManager
from splatsim.grpc_service._generated import (
    rendering_service_pb2 as pb2,
    rendering_service_pb2_grpc as pb2_grpc,
)
from splatsim.grpc_service.pose_buffer import PoseBuffer, TimestampedPose
from splatsim.grpc_service.viewmat_builder import (
    build_base_to_world_from_pose,
    build_intrinsics,
    build_viewmat_from_pose,
)
from splatsim.lidar_renderer import (
    LidarRenderer,
    build_lidar_sensors_from_config,
    gather_lidar_rig,
    render_lidars_concurrent,
)
from splatsim.renderer import Renderer, render_cameras_concurrent
from splatsim.rigid_body import RigidBody
from splatsim.scene import Scene

logger = logging.getLogger(__name__)


def _sweep_time_ns(spinning_frequency_hz: float) -> int | None:
    """One LiDAR revolution in nanoseconds, or ``None`` when rolling shutter
    is disabled via ``SPLATSIM_LIDAR_ROLLING_SHUTTER=0`` (static single-pose
    rendering, the pre-rolling-shutter behaviour)."""
    if os.environ.get("SPLATSIM_LIDAR_ROLLING_SHUTTER", "1") == "0":
        return None
    return _spin_period_ns(spinning_frequency_hz)


def _spin_period_ns(spinning_frequency_hz: float) -> int:
    """One LiDAR revolution in nanoseconds, regardless of the rolling-shutter
    env toggle (sector streaming needs the period even for static sectors)."""
    return int(1_000_000_000 / max(float(spinning_frequency_hz), 1e-6))


def _lidar_sector_count() -> int:
    """How many azimuth sectors each LiDAR revolution is rendered in.

    ``SPLATSIM_LIDAR_SECTORS=S`` (S >= 2) turns the pose-stream LiDAR loop
    into sector streaming: each revolution is rendered as S wedges, each from
    poses interpolated over ITS OWN slice of the sweep, and the wedges are
    concatenated and published once per revolution. This is both a better
    rolling shutter (per-sector motion instead of one whole-sweep pose pair)
    and faster (the fused CUDA azimuth cull drops ~(S-1)/S of the Gaussians
    per wedge). ``S`` must divide the panorama into tile-aligned slices — see
    :func:`splatsim.lidar_renderer._panorama_geometry`. Default 1 (off).
    """
    try:
        return max(1, int(os.environ.get("SPLATSIM_LIDAR_SECTORS", "1")))
    except ValueError:
        return 1


def _pose_from_msg(
    pose: pb2.Pose,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float] | None]:
    """Decode a wire pose into the ``(position, rotation)`` a body takes.

    An all-zero quaternion — what a client that fills in only a position sends,
    proto3 having no unset scalar — is not a rotation: it comes back as
    ``None``, which :meth:`~splatsim.rigid_body.RigidBody.set_pose` reads as
    "keep the one you have" rather than collapsing the body.
    """
    p, r = pose.position, pose.rotation
    rotation = (r.w, r.x, r.y, r.z)
    return (p.x, p.y, p.z), None if rotation == (0.0, 0.0, 0.0, 0.0) else rotation


@dataclass
class _SectorHooks:
    """Modality-specific callbacks for the sector-streaming pose loop.

    ``render_sector(start, end, k, state)`` renders sector ``k`` from the pose
    pair at the sector window's endpoints and returns an opaque per-sector
    output; ``state`` is a scratch dict living for one revolution (used to
    gather the Gaussian set once and reuse it across all sectors).
    ``publish_revolution(outputs, end_pose, stamp_ns)`` receives the S sector
    outputs in scan order plus the sweep-end pose/stamp.
    """

    n_sectors: int
    spin_period_ns: int
    render_sector: Callable[[TimestampedPose, TimestampedPose, int, dict], object]
    publish_revolution: Callable[[list, TimestampedPose, int], None]


@dataclass
class _PinholeConfig:
    """Minimal CameraConfig-protocol implementation for CameraInfoPublisher."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


@dataclass
class _RigLidar:
    """One LiDAR on the rig, with everything the frame loop needs."""

    name: str
    renderer: LidarRenderer
    publisher: PointCloud2Publisher
    drop_threshold: float
    alpha_threshold: float


@dataclass
class _RigCamera:
    """One camera on the rig; ``cam_to_base`` is composed with the ego pose."""

    name: str
    renderer: Renderer
    K: torch.Tensor
    cam_to_base: torch.Tensor  # (4, 4) camera→base_link
    image_pub: ImagePublisher | None
    info_pub: CameraInfoPublisher | None


class RenderingServiceServicer(pb2_grpc.RenderingServiceServicer):
    """gRPC servicer that manages scene loading, rendering, and DDS publishing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False

        self._scene: Scene | None = None
        self._renderer: Renderer | None = None
        self._K: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._dp: DomainParticipant | None = None
        self._image_pub: ImagePublisher | None = None
        self._camera_info_pub: CameraInfoPublisher | None = None
        self._frame_rate: float = 30.0
        self._clock_initial_ns: int = 0
        self._render_count: int = 0

        # LiDAR state (populated by InitializeLidar; shares self._scene).
        self._lidar_renderer: LidarRenderer | None = None
        self._pointcloud_pub: PointCloud2Publisher | None = None
        self._lidar_frame_rate: float = 10.0
        self._lidar_drop_threshold: float = 0.5
        self._lidar_alpha_threshold: float = 0.1
        self._lidar_render_count: int = 0

        # Rig state: every LiDAR / camera driven off ONE base_link pose stream.
        # Rendering them together is what lets a frame share a single Gaussian
        # gather (see render_lidars_concurrent / render_cameras_concurrent), so
        # N sensors cost one multi-million-Gaussian transient buffer, not N.
        self._rig_lidars: list[_RigLidar] = []
        self._rig_cameras: list[_RigCamera] = []
        self._rig_frame_rate: float = 10.0
        self._rig_render_count: int = 0

        # Dynamic objects ("actors"): rigid Gaussian bodies placed by
        # SpawnActor and driven per frame by the `actors` field of the pose
        # streams. Only the servicer-side bookkeeping lives here; the bodies
        # themselves are in the Scene, alongside the background.
        # Per instance: True when its poses arrive in the bundle's ENU world
        # frame and must have the scene origin subtracted (see SpawnActor).
        self._actor_world_frame: dict[str, bool] = {}
        # LoD-sorted Gaussians of standalone object files, keyed by path, so
        # every instance spawned from one file shares a single upload. Held
        # until the next Initialize and unbounded by design: the paths come
        # from an operator's scenario, not from arbitrary client input.
        self._actor_file_assets: dict[str, tuple[GaussianTensors, LodIndex | None]] = {}
        # Instance ids seen in a pose stream but never spawned; kept so the
        # warning is logged once instead of every frame.
        self._unknown_actor_ids: set[str] = set()

    def _warmup(self, what: str, render_once: Callable[[], object]) -> None:
        """Render one throwaway frame so CUDA module loads / kernel JIT happen
        here instead of stalling the first streamed pose (measured ~320 ms of
        GIL hold on the first frame otherwise). Nothing is published. Failures
        are logged and swallowed: a broken warmup pose must not fail Initialize
        when the real pose stream would still work.
        """
        try:
            t0 = time.monotonic()
            with torch.no_grad():
                render_once()
            logger.info(
                "Warmup render (%s) done in %.0f ms",
                what,
                (time.monotonic() - t0) * 1000,
            )
        except Exception:
            logger.exception("Warmup render (%s) failed; continuing", what)

    def Initialize(
        self,
        request: pb2.InitializeRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InitializeResponse:
        """Load scene, create renderer and DDS publishers."""
        with self._lock:
            try:
                device = torch.device(request.device or "cuda")
                self._device = device

                # Build the LoD manager unconditionally so the octree LoD
                # index is pre-computed at load time, letting SetLod toggle LoD
                # on/off at runtime without re-loading the scene. This is also
                # what makes LoD active by default in the Docker / gRPC path,
                # which loads scenes directly rather than via Scene.from_config.
                lod_manager = LodManager(LodConfig())

                logger.info("Loading scene: %s", request.scene_path)
                background = Background(
                    request.scene_path,
                    device=device,
                    use_sh=request.use_sh,
                    lod_manager=lod_manager,
                )
                # Scene enables LoD by default whenever a manager is present;
                # only an explicit enable_lod=false overrides that initial state.
                self._scene = Scene(background=background, lod_manager=lod_manager)
                if request.HasField("enable_lod"):
                    self._scene.lod_enabled = request.enable_lod
                # A new scene means new bodies: nothing spawned against the
                # previous one survives, so drop its bookkeeping too.
                self._actor_world_frame.clear()
                self._actor_file_assets.clear()
                self._unknown_actor_ids.clear()
                logger.info(
                    "Scene loaded: %d Gaussians (LoD %s)",
                    background.num_gaussians,
                    "on" if self._scene.lod_enabled else "off",
                )

                intr = request.intrinsics
                bg = request.background_color
                bg_color = (bg.x, bg.y, bg.z) if bg else (0.0, 0.0, 0.0)
                self._renderer = Renderer(
                    width=intr.width,
                    height=intr.height,
                    device=device,
                    background_color=bg_color,
                    near_plane=request.near_plane or 0.01,
                    far_plane=request.far_plane or 1000.0,
                    radius_clip=getattr(request, "radius_clip", 0.0) or 0.0,
                )

                self._K = build_intrinsics(intr.fx, intr.fy, intr.cx, intr.cy, device)

                dp = DomainParticipant()
                self._dp = dp  # prevent GC from destroying DDS entities
                frame_id = request.frame_id or "camera"
                self._image_pub = ImagePublisher(
                    dp,
                    topic_name=request.image_topic or "/splatsim/image_raw",
                    frame_id=frame_id,
                    compress_format=request.compress_format or "",
                )

                cam_config = _PinholeConfig(
                    fx=intr.fx,
                    fy=intr.fy,
                    cx=intr.cx,
                    cy=intr.cy,
                    image_width=intr.width,
                    image_height=intr.height,
                )
                self._camera_info_pub = CameraInfoPublisher(
                    dp,
                    cam_config,
                    topic_name=request.camera_info_topic or "/splatsim/camera_info",
                    frame_id=frame_id,
                )

                self._frame_rate = request.frame_rate or 30.0
                clk = request.clock_initial
                self._clock_initial_ns = (
                    clk.sec * 1_000_000_000 + clk.nanosec if clk else 0
                )

                self._initialized = True
                logger.info("Initialization complete (%.1f fps)", self._frame_rate)

                renderer, K, scene = self._renderer, self._K, self._scene
                self._warmup(
                    "camera",
                    lambda: renderer.render(
                        torch.eye(4, device=device), K, scene=scene
                    ),
                )

                centroid = background.tile_local_centroid
                scene_origin = pb2.Vector3(
                    x=float(centroid[0]),
                    y=float(centroid[1]),
                    z=float(centroid[2]),
                )
                ecef_t = background.ecef_translation
                ecef_r = background.ecef_rotation
                return pb2.InitializeResponse(
                    success=True,
                    scene_origin=scene_origin,
                    ecef_translation=pb2.Vector3(
                        x=float(ecef_t[0]),
                        y=float(ecef_t[1]),
                        z=float(ecef_t[2]),
                    ),
                    ecef_rotation=ecef_r.flatten().tolist(),
                )

            except Exception as exc:
                logger.exception("Initialize failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    def SetLod(
        self,
        request: pb2.SetLodRequest,
        context: grpc.ServicerContext,
    ) -> pb2.SetLodResponse:
        """Toggle Level-of-Detail filtering on the loaded scene at runtime.

        The render loop reads ``scene.lod_enabled`` per frame, so the new
        state takes effect on the next rendered frame. Enabling has no effect
        when the scene carries no pre-computed LoD index (the setter guards on
        the manager), in which case ``enabled`` in the response reflects the
        actual — still off — state.
        """
        with self._lock:
            if not self._initialized or self._scene is None:
                return pb2.SetLodResponse(
                    success=False,
                    enabled=False,
                    message="Scene not initialized. Call Initialize first.",
                )
            self._scene.lod_enabled = request.enabled
            effective = self._scene.lod_enabled
            logger.info(
                "SetLod: requested=%s effective=%s",
                request.enabled,
                effective,
            )
            message = ""
            if request.enabled and not effective:
                message = "LoD unavailable: no LoD index pre-computed for this scene."
            return pb2.SetLodResponse(
                success=True,
                enabled=effective,
                message=message,
            )

    # --- dynamic objects ("actors") ------------------------------------------

    def ListActorAssets(
        self,
        request: pb2.ListActorAssetsRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ListActorAssetsResponse:
        """Report the rigid dynamic-object assets the loaded scene bundle ships.

        The bank is read from the scene USDZ on first use, so a scene that
        carries none answers with ``success=False`` and the reason rather than
        an empty list that looks like a loaded-but-empty bank.
        """
        with self._lock:
            scene = self._scene
            if not self._initialized or scene is None:
                return pb2.ListActorAssetsResponse(
                    success=False,
                    message="Scene not initialized. Call Initialize first.",
                )
            try:
                library = scene.ensure_actor_library()
            except Exception as exc:
                logger.info("No actor asset bank available: %s", exc)
                return pb2.ListActorAssetsResponse(success=False, message=str(exc))

            assets = []
            for asset_id in library.asset_ids:
                info = library.info(asset_id)
                dx, dy, dz = info.size
                assets.append(
                    pb2.ActorAssetInfo(
                        asset_id=info.asset_id,
                        class_name=info.class_name,
                        size=pb2.Vector3(x=dx, y=dy, z=dz),
                        n_points=info.n_points,
                        sh_degree=info.sh_degree,
                        has_lidar_attributes=info.has_lidar_attributes,
                    )
                )
            return pb2.ListActorAssetsResponse(success=True, assets=assets)

    def SpawnActor(
        self,
        request: pb2.SpawnActorRequest,
        context: grpc.ServicerContext,
    ) -> pb2.SpawnActorResponse:
        """Place a dynamic object in the already-loaded scene.

        The body joins the scene the render loop is already gathering, so it
        shows up in the next frame of every stream — camera, LiDAR and rig
        alike — and from then on its pose is whatever the streams' ``actors``
        field last said (see :meth:`_apply_actor_poses`).

        Gaussians come either from the bundle's own asset bank (``asset_id``)
        or from a standalone object file on the server (``asset_path``), for
        scenes whose bundle ships no bank. Either way instances of one asset
        share a single upload.

        A body the scene cannot render alongside what it already holds — an SH
        object in a non-SH scene — is refused by
        :meth:`~splatsim.scene.Scene.add_rigid_body` and comes back as a failed
        response, rather than aborting the render thread a frame later.
        """
        with self._lock:
            scene = self._scene
            if not self._initialized or scene is None:
                return pb2.SpawnActorResponse(
                    success=False,
                    message="Scene not initialized. Call Initialize first.",
                )
            name = request.instance_id
            if not name:
                return pb2.SpawnActorResponse(
                    success=False, message="instance_id is required"
                )
            if name in scene:
                return pb2.SpawnActorResponse(
                    success=False,
                    message=f"a rigid body named {name!r} is already in the scene",
                )
            if request.world_frame and scene.background is None:
                return pb2.SpawnActorResponse(
                    success=False,
                    message=(
                        "world_frame needs a scene background to take the "
                        "tile-local origin from"
                    ),
                )

            try:
                position, rotation = (
                    _pose_from_msg(request.pose)
                    if request.HasField("pose")
                    else (None, None)
                )
                if request.asset_path:
                    body = self._rigid_body_from_path(request.asset_path)
                    if position is not None:
                        body.set_pose(
                            self._to_tile_local(position, request.world_frame),
                            rotation,
                        )
                    scene.add_rigid_body(name, body)
                elif request.asset_id:
                    body = scene.spawn_actor(
                        request.asset_id,
                        name,
                        position=position,
                        rotation=rotation,
                        world_position=request.world_frame,
                    )
                else:
                    raise ValueError("either asset_id or asset_path is required")
            except Exception as exc:
                logger.exception("SpawnActor(%s) failed", name)
                return pb2.SpawnActorResponse(success=False, message=str(exc))

            self._actor_world_frame[name] = request.world_frame
            logger.info(
                "Spawned actor %r (%s, %d Gaussians, %s frame)",
                name,
                request.asset_path or request.asset_id,
                body.num_gaussians,
                "world" if request.world_frame else "tile-local",
            )
            return pb2.SpawnActorResponse(success=True, n_gaussians=body.num_gaussians)

    def RemoveActor(
        self,
        request: pb2.RemoveActorRequest,
        context: grpc.ServicerContext,
    ) -> pb2.RemoveActorResponse:
        """Take a spawned dynamic object out of the scene.

        It disappears from the next rendered frame; its asset stays cached, so
        spawning it again does not re-upload the Gaussians.
        """
        with self._lock:
            scene = self._scene
            if not self._initialized or scene is None:
                return pb2.RemoveActorResponse(
                    success=False,
                    message="Scene not initialized. Call Initialize first.",
                )
            name = request.instance_id
            if name not in scene:
                return pb2.RemoveActorResponse(
                    success=False,
                    message=f"no rigid body named {name!r} in the scene",
                )
            scene.remove_rigid_body(name)
            self._actor_world_frame.pop(name, None)
            logger.info("Removed actor %r", name)
            return pb2.RemoveActorResponse(success=True)

    def _rigid_body_from_path(self, asset_path: str) -> RigidBody:
        """Instantiate a standalone Gaussian object file as a rigid body.

        The file is read, and its LoD tiers computed, once per path: every
        instance spawned from it then shares those Gaussians, the way the
        bundle's asset bank shares its own — fifty copies of one vehicle cost
        one upload and fifty poses. ``rotate_sh``, as for bank actors: a
        dynamic object changes heading, and leaving the specular bands in the
        object frame spins the highlights with it.
        """
        device = self._device or torch.device("cuda")
        cached = self._actor_file_assets.get(asset_path)
        if cached is None:
            scene = self._scene
            background = scene.background if scene is not None else None
            first = RigidBody(
                asset_path,
                device=device,
                # The scene's own setting: a body whose colours are shaped
                # differently from the background's cannot be gathered with it.
                use_sh=background.use_sh if background is not None else False,
                lod_manager=scene.lod_manager if scene is not None else None,
            )
            cached = (first.base_tensors, first.lod_index)
            self._actor_file_assets[asset_path] = cached
        tensors, lod_index = cached
        return RigidBody(tensors, device=device, lod_index=lod_index, rotate_sh=True)

    def _to_tile_local(
        self, position: tuple[float, float, float], world_frame: bool
    ) -> tuple[float, float, float] | torch.Tensor:
        """Recentre *position* onto the tile when it arrived world-frame.

        The renderer works in the tile-local frame
        :class:`~splatsim.background.Background` recentres a scene into, and
        the pose streams are already expressed there; an instance spawned with
        ``world_frame`` sends ENU world coordinates instead and has the scene
        origin subtracted here, exactly as
        :func:`splatsim.actor_assets.world_to_tile_local` does for scenario
        code.
        """
        scene = self._scene
        if not world_frame or scene is None or scene.background is None:
            return position
        return world_to_tile_local(position, scene.background)

    def _apply_actor_poses(self, actors: Sequence[pb2.ActorPose]) -> None:
        """Move spawned dynamic objects to the poses a stream message carried.

        Called from the render thread immediately before a frame is rendered,
        so the actors a frame shows are the ones its own pose message
        described — and, on a rig, every sensor sees them at the same instant.

        A pose for an instance that was never spawned is ignored (warned about
        once per id): a stale entry in a client's actor list must not kill a
        running stream.
        """
        scene = self._scene
        if scene is None:
            return
        for actor in actors:
            body = scene.rigid_bodies.get(actor.instance_id)
            if body is None:
                if actor.instance_id not in self._unknown_actor_ids:
                    self._unknown_actor_ids.add(actor.instance_id)
                    logger.warning(
                        "Ignoring pose for actor %r: not spawned", actor.instance_id
                    )
                continue
            if not actor.HasField("pose"):
                continue
            position, rotation = _pose_from_msg(actor.pose)
            body.set_pose(
                self._to_tile_local(
                    position, self._actor_world_frame.get(actor.instance_id, False)
                ),
                rotation,
            )

    def StreamCameraData(
        self,
        request_iterator: Iterator[pb2.CameraData],
        context: grpc.ServicerContext,
    ) -> pb2.StreamSummary:
        """Consume timestamped camera poses, render, and publish via DDS.

        Reading from the gRPC stream and GPU rendering run on separate
        threads so that slow rendering never blocks pose ingestion.
        The render loop always uses the latest available pose, and old
        poses are automatically dropped from the buffer.
        """
        if not self._initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Service not initialized. Call Initialize first.",
            )

        assert self._renderer is not None  # noqa: S101
        assert self._scene is not None  # noqa: S101
        assert self._K is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        return self._run_pose_stream(
            request_iterator,
            frame_rate=self._frame_rate,
            render_and_publish=self._render_and_publish,
        )

    def _run_pose_stream(
        self,
        request_iterator: Iterator[pb2.CameraData | pb2.LidarData | pb2.RigData],
        *,
        frame_rate: float,
        render_and_publish: Callable[
            [TimestampedPose, int, TimestampedPose | None], None
        ],
        sweep_time_ns: int | None = None,
        sector_hooks: "_SectorHooks | None" = None,
    ) -> pb2.StreamSummary:
        """Shared two-thread pose-streaming loop for camera, LiDAR and rig.

        Reading from the gRPC stream and GPU rendering run on separate
        threads so slow rendering never blocks pose ingestion. The render
        loop always uses the latest available pose at ``frame_rate`` cadence,
        publishing through the supplied ``render_and_publish`` callback, and
        poses consumed by a frame are dropped from the buffer afterwards.

        ``sweep_time_ns`` enables rolling shutter for spinning LiDARs: each
        frame then also receives the pose one sweep BEFORE the rendered
        (latest) pose, interpolated from the buffered pose queue — matching a
        real spinning LiDAR, whose cloud stamped at ``t`` was swept over
        ``[t - sweep, t]``. ``None`` keeps the static single-pose behaviour.

        Dynamic objects ride along on the same messages: the newest ``actors``
        the stream has delivered are applied to the scene by the render thread
        just before it renders, and only when they have changed since the last
        frame. A message without ``actors`` leaves the previous poses standing.

        ``sector_hooks`` switches the render thread to SECTOR STREAMING (see
        :func:`_lidar_sector_count`): instead of one whole-sweep render per
        frame, revolutions are pinned to the pose timeline and each azimuth
        sector is rendered as soon as the poses covering its slice of the
        sweep have arrived, from poses interpolated at the slice endpoints.
        The finished revolution is concatenated and published with the
        sweep-end stamp. ``frame_rate`` is ignored in this mode — the loop is
        paced by the pose timeline itself.
        """
        pose_buffer = PoseBuffer()
        stream_done = threading.Event()
        render_failed = threading.Event()
        frames_rendered = 0
        poses_received = 0
        # Actor poses from the newest message, read by the render thread. The
        # reader swaps in a whole new tuple per message and the render thread
        # only ever reads the reference, so this needs no lock; a message
        # without actors leaves the previous poses standing.
        latest_actors: tuple[pb2.ActorPose, ...] = ()

        frame_period_s = 1.0 / frame_rate

        def _render_loop() -> None:
            """Render at frame_rate using the latest buffered pose."""
            nonlocal frames_rendered
            # Actor poses already pushed to the scene. The reader swaps in a
            # new tuple only for a message that carries actors, so identity is
            # enough to skip re-uploading poses that did not move.
            applied_actors: tuple[pb2.ActorPose, ...] | None = None
            try:
                while not stream_done.is_set():
                    # Wait for a new pose; clear so we block again next iteration
                    if not pose_buffer.new_pose_event.wait(timeout=1.0):
                        continue
                    pose_buffer.new_pose_event.clear()

                    render_start = time.monotonic()

                    latest = pose_buffer.get_latest()
                    if latest is None:
                        continue

                    render_time_ns = latest.time_ns

                    # Rolling shutter: reconstruct the pose one sweep back from
                    # the queued poses. Early frames without enough history fall
                    # back to the oldest buffered pose (a shorter sweep), and a
                    # single-pose queue renders static.
                    sweep_start: TimestampedPose | None = None
                    if sweep_time_ns:
                        sweep_start = pose_buffer.interpolate(
                            render_time_ns - sweep_time_ns
                        )
                        if sweep_start is None:
                            earliest = pose_buffer.get_earliest()
                            if (
                                earliest is not None
                                and earliest.time_ns < render_time_ns
                            ):
                                sweep_start = earliest

                    if frames_rendered <= 3 or frames_rendered % 100 == 0:
                        logger.info(
                            "Render #%d: render_t=%d pos=(%.4f, %.4f, %.4f)",
                            frames_rendered,
                            render_time_ns,
                            latest.position[0],
                            latest.position[1],
                            latest.position[2],
                        )

                    if latest_actors is not applied_actors:
                        self._apply_actor_poses(latest_actors)
                        applied_actors = latest_actors
                    render_and_publish(latest, render_time_ns, sweep_start)
                    frames_rendered += 1
                    pose_buffer.trim_before(render_time_ns)

                    elapsed = time.monotonic() - render_start
                    sleep_time = frame_period_s - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception:
                logger.exception("Render loop failed")
                render_failed.set()

        def _sector_render_loop() -> None:
            """Render sectors as the pose timeline covers their sweep slices.

            A revolution is anchored at ``rev_start_ns`` (the first buffered
            pose, then back-to-back). Sector ``k`` of ``S`` covers
            ``[rev_start + k/S * period, rev_start + (k+1)/S * period]``; it is
            rendered the moment the newest buffered pose passes the window's
            end, from poses interpolated at the window endpoints. After sector
            ``S-1`` the revolution is published with the sweep-end stamp.

            If the pose timeline runs away from the loop (rendering slower
            than real time, or a source gap), the next revolution resyncs to
            the newest pose instead of grinding through stale sweeps — the
            sector-mode analogue of the plain loop's latest-pose behaviour.
            """
            nonlocal frames_rendered
            assert sector_hooks is not None  # noqa: S101
            n_sectors = sector_hooks.n_sectors
            period = sector_hooks.spin_period_ns
            applied_actors: tuple[pb2.ActorPose, ...] | None = None
            rev_start_ns: int | None = None
            k = 0
            outputs: list = []
            state: dict = {}

            def _pose_at(t_ns: int) -> TimestampedPose | None:
                pose = pose_buffer.interpolate(t_ns)
                if pose is not None:
                    return pose
                # Before the buffered range (first revolution): clamp to the
                # oldest pose, mirroring the plain loop's short-sweep fallback.
                earliest = pose_buffer.get_earliest()
                if earliest is not None and t_ns <= earliest.time_ns:
                    return earliest
                return pose_buffer.get_latest()

            try:
                while not stream_done.is_set():
                    if not pose_buffer.new_pose_event.wait(timeout=1.0):
                        continue
                    pose_buffer.new_pose_event.clear()
                    # Drain: render every sector whose window the newest pose
                    # has passed (several per wake-up when catching up).
                    while not stream_done.is_set():
                        latest_ns = pose_buffer.latest_time_ns
                        if latest_ns is None:
                            break
                        if rev_start_ns is None:
                            rev_start_ns = latest_ns
                        if k == 0 and latest_ns - rev_start_ns > 2 * period:
                            logger.warning(
                                "Sector loop resync: poses ran %.0f ms ahead "
                                "of the revolution start; restarting at the "
                                "newest pose",
                                (latest_ns - rev_start_ns) / 1e6,
                            )
                            rev_start_ns = latest_ns
                        sector_end_ns = rev_start_ns + (k + 1) * period // n_sectors
                        if latest_ns < sector_end_ns:
                            break
                        sector_start_ns = rev_start_ns + k * period // n_sectors
                        start = _pose_at(sector_start_ns)
                        end = _pose_at(sector_end_ns)
                        if start is None or end is None:
                            break
                        if k == 0 and latest_actors is not applied_actors:
                            # The rig-wide Gaussian gather happens on sector 0
                            # and is reused for the rest of the revolution, so
                            # actors move once per revolution, not per sector.
                            self._apply_actor_poses(latest_actors)
                            applied_actors = latest_actors
                        outputs.append(sector_hooks.render_sector(start, end, k, state))
                        k += 1
                        if k == n_sectors:
                            sector_hooks.publish_revolution(outputs, end, sector_end_ns)
                            frames_rendered += 1
                            if frames_rendered <= 3 or frames_rendered % 100 == 0:
                                logger.info(
                                    "Sector revolution #%d published "
                                    "(t=%d ns, %d sectors)",
                                    frames_rendered,
                                    sector_end_ns,
                                    n_sectors,
                                )
                            pose_buffer.trim_before(sector_end_ns)
                            rev_start_ns = sector_end_ns
                            k = 0
                            outputs = []
                            state = {}
            except Exception:
                logger.exception("Sector render loop failed")
                render_failed.set()

        render_thread = threading.Thread(
            target=_sector_render_loop if sector_hooks is not None else _render_loop,
            daemon=True,
        )
        render_thread.start()

        try:
            for data in request_iterator:
                stamp = data.stamp
                time_ns = stamp.sec * 1_000_000_000 + stamp.nanosec

                p = data.pose.position
                r = data.pose.rotation
                pose = TimestampedPose(
                    time_ns=time_ns,
                    position=(p.x, p.y, p.z),
                    rotation=(r.w, r.x, r.y, r.z),
                )
                pose_buffer.append(pose)
                poses_received += 1
                if data.actors:
                    latest_actors = tuple(data.actors)

                if poses_received <= 3 or poses_received % 100 == 0:
                    logger.info(
                        "Received pose #%d: t=%d ns pos=(%.4f, %.4f, %.4f)",
                        poses_received,
                        time_ns,
                        p.x,
                        p.y,
                        p.z,
                    )

                if render_failed.is_set():
                    logger.error("Render thread died, stopping stream reader")
                    break
        finally:
            stream_done.set()
            render_thread.join(timeout=10.0)

        logger.info(
            "Stream finished: poses_received=%d, frames_rendered=%d",
            poses_received,
            frames_rendered,
        )
        return pb2.StreamSummary(
            frames_rendered=frames_rendered,
            poses_received=poses_received,
        )

    def _render_and_publish(
        self,
        pose: TimestampedPose,
        render_time_ns: int,
        sweep_start: TimestampedPose | None = None,
    ) -> None:
        """Render a single frame at the interpolated pose and publish via DDS.

        ``sweep_start`` is accepted for signature parity with the LiDAR/rig
        callbacks; the camera renderer has no motion model, so it is unused.
        """
        del sweep_start
        assert self._renderer is not None  # noqa: S101
        assert self._K is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        t0 = time.monotonic()
        viewmat = build_viewmat_from_pose(pose.position, pose.rotation, self._device)
        t_viewmat = time.monotonic()

        logger.debug(
            "Render pose: pos=(%.4f, %.4f, %.4f) rot_wxyz=(%.4f, %.4f, %.4f, %.4f)",
            *pose.position,
            *pose.rotation,
        )
        logger.debug("Viewmat:\n%s", viewmat.cpu().numpy())

        with torch.no_grad():
            rgb = self._renderer.render(viewmat, self._K, scene=self._scene)
        t_render = time.monotonic()

        # float32 RGB [H, W, 3] → uint8 BGR [H, W, 3] (flip on GPU before transfer)
        bgr_np = (rgb.clamp(0.0, 1.0) * 255).byte()[:, :, [2, 1, 0]].cpu().numpy()
        t_transfer = time.monotonic()

        sec, nanosec = divmod(render_time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)

        if self._image_pub is not None:
            self._image_pub.publish(bgr_np, stamp=stamp)
        if self._camera_info_pub is not None:
            self._camera_info_pub.publish(stamp=stamp)
        t_publish = time.monotonic()

        total_ms = (t_publish - t0) * 1000
        self._render_count += 1
        if self._render_count <= 5 or self._render_count % 100 == 0:
            logger.info(
                "Render timing #%d: total=%.1fms "
                "(viewmat=%.1f render=%.1f transfer=%.1f publish=%.1f)",
                self._render_count,
                total_ms,
                (t_viewmat - t0) * 1000,
                (t_render - t_viewmat) * 1000,
                (t_transfer - t_render) * 1000,
                (t_publish - t_transfer) * 1000,
            )

    # ── LiDAR ────────────────────────────────────────────────────────────

    def InitializeLidar(
        self,
        request: pb2.InitializeLidarRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InitializeResponse:
        """Add a LiDAR sensor to the already-loaded scene.

        ``Initialize`` must have been called first — the LiDAR renderer shares
        the scene and DomainParticipant created there.
        """
        with self._lock:
            try:
                if not self._initialized or self._scene is None:
                    return pb2.InitializeResponse(
                        success=False,
                        message="Scene not initialized. Call Initialize first.",
                    )
                assert self._device is not None  # noqa: S101
                assert self._dp is not None  # noqa: S101

                # Rig form takes precedence; the single-sensor field stays
                # supported for existing clients.
                sensor_msgs = list(request.sensors) or [request.sensor]
                if len(sensor_msgs) > 1 or request.sensors:
                    return self._init_lidar_rig(sensor_msgs)

                s = sensor_msgs[0]
                ext = s.extrinsic
                pos = ext.position
                rot = ext.rotation  # wxyz
                elevation = tuple(s.elevation_deg) or None

                # Faithful hardware defaults for the requested model fill in
                # any field the client left at 0 (e.g. Velodyne HDL-64E gets
                # 64 beams / 2083 azimuth samples / 120 m range). Unknown
                # models fall back to the baseline literals below.
                d = sensor_defaults(s.sensor_type)
                cfg = LidarConfig(
                    name=s.name or "lidar",
                    sensor_type=s.sensor_type,
                    n_rows=int(s.n_rows) or int(d.get("n_rows", 128)),
                    n_columns=int(s.n_columns) or int(d.get("n_columns", 2048)),
                    fps=s.fps or d.get("fps", 10.0),
                    min_range_m=s.min_range_m or d.get("min_range_m", 0.3),
                    max_range_m=s.max_range_m or d.get("max_range_m", 120.0),
                    position=(pos.x, pos.y, pos.z),
                    rotation=(rot.w, rot.x, rot.y, rot.z),
                    elevation_deg=elevation,
                    pointcloud_topic=(
                        s.pointcloud_topic or "/splatsim/lidar/pointcloud"
                    ),
                    frame_id=s.frame_id or "splatsim_lidar",
                    drop_threshold=s.drop_threshold or 0.5,
                    alpha_threshold=s.alpha_threshold or 0.1,
                )

                spec = build_lidar_sensors_from_config([cfg])[0]
                lidar_renderer = LidarRenderer(
                    spec,
                    device=self._device,
                    min_range_m=cfg.min_range_m,
                    max_range_m=cfg.max_range_m,
                )
                self._lidar_renderer = lidar_renderer
                self._pointcloud_pub = PointCloud2Publisher(
                    self._dp,
                    topic_name=cfg.pointcloud_topic,
                    frame_id=cfg.frame_id,
                )
                self._lidar_frame_rate = cfg.fps
                self._lidar_drop_threshold = cfg.drop_threshold
                self._lidar_alpha_threshold = cfg.alpha_threshold

                logger.info(
                    "LiDAR initialized: name=%s rows=%d cols=%d %.1ffps topic=%s",
                    cfg.name,
                    self._lidar_renderer.n_rows,
                    self._lidar_renderer.n_columns,
                    cfg.fps,
                    cfg.pointcloud_topic,
                )

                lidar_scene = self._scene
                # Warm the same kernel path streaming will use: rolling
                # shutter stages per-Gaussian velocities, a different code
                # path from the static render (zero motion here, so the
                # output is unchanged).
                warm_end = (
                    torch.eye(4, device=self._device)
                    if _sweep_time_ns(lidar_renderer.sensor_spec.spinning_frequency_hz)
                    else None
                )
                self._warmup(
                    f"lidar {cfg.name}",
                    lambda: lidar_renderer.panorama_to_pointcloud2_data(
                        lidar_renderer.render(
                            torch.eye(4, device=self._device),
                            scene=lidar_scene,
                            base_to_world_end=warm_end,
                        ),
                        drop_threshold=cfg.drop_threshold,
                        alpha_threshold=cfg.alpha_threshold,
                    ),
                )
                return pb2.InitializeResponse(success=True)

            except Exception as exc:
                logger.exception("InitializeLidar failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    def StreamLidarData(
        self,
        request_iterator: Iterator[pb2.LidarData],
        context: grpc.ServicerContext,
    ) -> pb2.StreamSummary:
        """Consume timestamped base_link poses, render LiDAR, and publish via DDS.

        Mirrors :meth:`StreamCameraData`: the gRPC ingestion thread and the GPU
        render loop run separately, and the render loop always uses the latest
        buffered pose at the sensor's spin rate.
        """
        if not self._initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Service not initialized. Call Initialize first.",
            )
        if self._lidar_renderer is None or self._pointcloud_pub is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "LiDAR not initialized. Call InitializeLidar first.",
            )

        assert self._scene is not None  # noqa: S101
        assert self._device is not None  # noqa: S101
        assert self._lidar_renderer is not None  # noqa: S101

        return self._run_pose_stream(
            request_iterator,
            frame_rate=self._lidar_frame_rate,
            render_and_publish=self._render_and_publish_lidar,
            sweep_time_ns=_sweep_time_ns(
                self._lidar_renderer.sensor_spec.spinning_frequency_hz
            ),
            sector_hooks=self._lidar_sector_hooks(),
        )

    def _lidar_sector_hooks(self) -> "_SectorHooks | None":
        """Sector-streaming callbacks for the single-LiDAR pose stream, or
        ``None`` when ``SPLATSIM_LIDAR_SECTORS`` <= 1."""
        n_sectors = _lidar_sector_count()
        if n_sectors <= 1 or self._lidar_renderer is None:
            return None
        renderer = self._lidar_renderer
        scene = self._scene
        device = self._device
        assert scene is not None and device is not None  # noqa: S101
        publisher = self._pointcloud_pub
        assert publisher is not None  # noqa: S101
        # Rolling shutter (per-sector pose pair) unless disabled via env;
        # static sectors still render at each window's END pose.
        rolling = _sweep_time_ns(renderer.sensor_spec.spinning_frequency_hz) is not None

        def render_sector(
            start: TimestampedPose, end: TimestampedPose, k: int, state: dict
        ) -> dict:
            b2w_end = build_base_to_world_from_pose(end.position, end.rotation, device)
            if rolling:
                b2w = build_base_to_world_from_pose(
                    start.position, start.rotation, device
                )
                b2w_pair = (b2w, b2w_end)
            else:
                b2w_pair = (b2w_end, None)
            with torch.no_grad():
                if "shared" not in state:
                    # One LOD gather per revolution, reused by every sector.
                    state["shared"] = renderer.gather(
                        b2w_pair[0], scene, base_to_world_end=b2w_pair[1]
                    )
                shared = state["shared"]
                if shared is None:
                    return renderer._empty_panorama((k, n_sectors))
                return renderer.render(
                    b2w_pair[0],
                    scene=scene,
                    base_to_world_end=b2w_pair[1],
                    shared=shared,
                    sector=(k, n_sectors),
                )

        def publish_revolution(
            outputs: list, end_pose: TimestampedPose, stamp_ns: int
        ) -> None:
            del end_pose
            t0 = time.monotonic()
            with torch.no_grad():
                panorama = {
                    key: torch.cat([o[key] for o in outputs], dim=1)
                    for key in outputs[0]
                }
                records, n_points = renderer.panorama_to_pointcloud2_data(
                    panorama,
                    drop_threshold=self._lidar_drop_threshold,
                    alpha_threshold=self._lidar_alpha_threshold,
                )
            sec, nanosec = divmod(stamp_ns, 1_000_000_000)
            publisher.publish_packed(
                records, n_points, stamp=Time(sec=sec, nanosec=nanosec)
            )
            self._lidar_render_count += 1
            if self._lidar_render_count <= 5 or self._lidar_render_count % 100 == 0:
                logger.info(
                    "LiDAR sector revolution #%d: %d points, publish=%.1fms",
                    self._lidar_render_count,
                    n_points,
                    (time.monotonic() - t0) * 1000,
                )

        return _SectorHooks(
            n_sectors=n_sectors,
            spin_period_ns=_spin_period_ns(renderer.sensor_spec.spinning_frequency_hz),
            render_sector=render_sector,
            publish_revolution=publish_revolution,
        )

    # ── Rig: many LiDARs / cameras off one shared gather ─────────────────

    def _lidar_config_from_msg(self, s) -> LidarConfig:
        """Build a LidarConfig from a proto LidarSensorConfig (defaults filled)."""
        ext = s.extrinsic
        pos, rot = ext.position, ext.rotation  # rot is wxyz
        d = sensor_defaults(s.sensor_type)
        return LidarConfig(
            name=s.name or "lidar",
            sensor_type=s.sensor_type,
            n_rows=int(s.n_rows) or int(d.get("n_rows", 128)),
            n_columns=int(s.n_columns) or int(d.get("n_columns", 2048)),
            fps=s.fps or d.get("fps", 10.0),
            min_range_m=s.min_range_m or d.get("min_range_m", 0.3),
            max_range_m=s.max_range_m or d.get("max_range_m", 120.0),
            position=(pos.x, pos.y, pos.z),
            rotation=(rot.w, rot.x, rot.y, rot.z),
            elevation_deg=tuple(s.elevation_deg) or None,
            pointcloud_topic=(
                s.pointcloud_topic or f"/splatsim/{s.name or 'lidar'}/pointcloud"
            ),
            frame_id=s.frame_id or (s.name or "splatsim_lidar"),
            drop_threshold=s.drop_threshold or 0.5,
            alpha_threshold=s.alpha_threshold or 0.1,
        )

    def _init_lidar_rig(self, sensor_msgs) -> pb2.InitializeResponse:
        """Register several LiDARs that render together off one gather."""
        assert self._device is not None  # noqa: S101
        assert self._dp is not None  # noqa: S101

        cfgs = [self._lidar_config_from_msg(m) for m in sensor_msgs]
        names = [c.name for c in cfgs]
        if len(set(names)) != len(names):
            return pb2.InitializeResponse(
                success=False, message=f"duplicate LiDAR names: {names}"
            )

        specs = build_lidar_sensors_from_config(cfgs)
        rig: list[_RigLidar] = []
        for cfg, spec in zip(cfgs, specs):
            rig.append(
                _RigLidar(
                    name=cfg.name,
                    renderer=LidarRenderer(
                        spec,
                        device=self._device,
                        min_range_m=cfg.min_range_m,
                        max_range_m=cfg.max_range_m,
                    ),
                    publisher=PointCloud2Publisher(
                        self._dp,
                        topic_name=cfg.pointcloud_topic,
                        frame_id=cfg.frame_id,
                    ),
                    drop_threshold=cfg.drop_threshold,
                    alpha_threshold=cfg.alpha_threshold,
                )
            )
        self._rig_lidars = rig
        # The rig renders on one cadence; take the fastest sensor's rate so no
        # sensor is starved (slower ones simply repeat the latest pose).
        self._rig_frame_rate = max(c.fps for c in cfgs)
        logger.info(
            "LiDAR rig initialized: %d sensors (%s) at %.1f Hz",
            len(rig),
            ", ".join(names),
            self._rig_frame_rate,
        )

        assert self._scene is not None  # noqa: S101
        rig_scene = self._scene

        def _rig_once() -> None:
            # Zero-motion end pose so the warmup exercises the rolling-shutter
            # kernel path the pose stream will use.
            warm_end = (
                torch.eye(4, device=self._device) if self._rig_sweep_time_ns() else None
            )
            panoramas = render_lidars_concurrent(
                [rl.renderer for rl in rig],
                torch.eye(4, device=self._device),
                rig_scene,
                base_to_world_end=warm_end,
            )
            for rl, pano in zip(rig, panoramas):
                rl.renderer.panorama_to_pointcloud2_data(
                    pano,
                    drop_threshold=rl.drop_threshold,
                    alpha_threshold=rl.alpha_threshold,
                )

        self._warmup("lidar rig", _rig_once)
        return pb2.InitializeResponse(success=True)

    def InitializeCameraRig(
        self,
        request: pb2.InitializeCameraRigRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InitializeResponse:
        """Register cameras driven off the shared base_link pose stream."""
        with self._lock:
            try:
                if not self._initialized or self._scene is None:
                    return pb2.InitializeResponse(
                        success=False,
                        message="Scene not initialized. Call Initialize first.",
                    )
                assert self._device is not None  # noqa: S101
                assert self._dp is not None  # noqa: S101

                names = [c.name for c in request.cameras]
                if len(set(names)) != len(names):
                    return pb2.InitializeResponse(
                        success=False, message=f"duplicate camera names: {names}"
                    )

                rig: list[_RigCamera] = []
                for c in request.cameras:
                    intr = c.intrinsics
                    ext = c.extrinsic
                    cam_to_base = build_base_to_world_from_pose(
                        (ext.position.x, ext.position.y, ext.position.z),
                        (
                            ext.rotation.w,
                            ext.rotation.x,
                            ext.rotation.y,
                            ext.rotation.z,
                        ),
                        self._device,
                    )
                    rig.append(
                        _RigCamera(
                            name=c.name or "camera",
                            renderer=Renderer(
                                width=intr.width,
                                height=intr.height,
                                device=self._device,
                            ),
                            K=build_intrinsics(
                                intr.fx, intr.fy, intr.cx, intr.cy, self._device
                            ),
                            cam_to_base=cam_to_base,
                            image_pub=ImagePublisher(
                                self._dp,
                                topic_name=(
                                    c.image_topic
                                    or f"/splatsim/{c.name or 'camera'}/image"
                                ),
                                frame_id=c.frame_id or (c.name or "splatsim_camera"),
                                compress_format=c.compress_format or "",
                            ),
                            info_pub=CameraInfoPublisher(
                                self._dp,
                                topic_name=(
                                    c.camera_info_topic
                                    or f"/splatsim/{c.name or 'camera'}/camera_info"
                                ),
                                frame_id=c.frame_id or (c.name or "splatsim_camera"),
                                config=_PinholeConfig(
                                    fx=intr.fx,
                                    fy=intr.fy,
                                    cx=intr.cx,
                                    cy=intr.cy,
                                    image_width=intr.width,
                                    image_height=intr.height,
                                ),
                            ),
                        )
                    )
                self._rig_cameras = rig
                if request.frame_rate:
                    self._rig_frame_rate = request.frame_rate
                logger.info(
                    "Camera rig initialized: %d cameras (%s)",
                    len(rig),
                    ", ".join(names),
                )

                def _cam_rig_once() -> None:
                    render_cameras_concurrent(
                        [rc.renderer for rc in rig],
                        [torch.linalg.inv(rc.cam_to_base) for rc in rig],
                        [rc.K for rc in rig],
                        scene=self._scene,
                        camera_names=[rc.name for rc in rig],
                    )

                self._warmup("camera rig", _cam_rig_once)
                return pb2.InitializeResponse(success=True)
            except Exception as exc:
                logger.exception("InitializeCameraRig failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    def _rig_sweep_time_ns(self) -> int | None:
        """Sweep duration driving the rig's rolling shutter, or ``None``.

        The rig shares one (start, end) pose pair per frame, so one sweep
        duration has to serve every LiDAR. With mixed spin rates the fastest
        sensor's sweep is used — exact for it, under-compensating (never
        over-shooting) the slower ones.
        """
        if not self._rig_lidars:
            return None
        spins = {
            float(rl.renderer.sensor_spec.spinning_frequency_hz)
            for rl in self._rig_lidars
        }
        if len(spins) > 1:
            logger.warning(
                "Rig LiDARs spin at different rates (%s Hz); rolling shutter "
                "uses the fastest sweep, under-compensating slower sensors.",
                sorted(spins),
            )
        return _sweep_time_ns(max(spins))

    def StreamRigData(
        self,
        request_iterator: Iterator[pb2.RigData],
        context: grpc.ServicerContext,
    ) -> pb2.RigSummary:
        """Render every registered LiDAR and camera per streamed base_link pose."""
        if not self._initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Service not initialized. Call Initialize first.",
            )
        if not self._rig_lidars and not self._rig_cameras:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "No rig registered. Call InitializeLidar (sensors) or "
                "InitializeCameraRig first.",
            )

        summary = self._run_pose_stream(
            request_iterator,
            frame_rate=self._rig_frame_rate,
            render_and_publish=self._render_and_publish_rig,
            sweep_time_ns=self._rig_sweep_time_ns(),
            sector_hooks=self._rig_sector_hooks(),
        )
        return pb2.RigSummary(
            frames_rendered=summary.frames_rendered,
            poses_received=summary.poses_received,
            lidars_rendered=summary.frames_rendered * len(self._rig_lidars),
            cameras_rendered=summary.frames_rendered * len(self._rig_cameras),
        )

    def _rig_sector_hooks(self) -> "_SectorHooks | None":
        """Sector-streaming callbacks for the rig pose stream, or ``None``.

        LiDARs render per sector off ONE Gaussian gather per revolution;
        cameras render once per revolution at the sweep-end pose (their
        existing cadence — a camera has no sweep to slice).
        """
        n_sectors = _lidar_sector_count()
        if n_sectors <= 1 or not self._rig_lidars:
            return None
        scene = self._scene
        device = self._device
        assert scene is not None and device is not None  # noqa: S101
        renderers = [rl.renderer for rl in self._rig_lidars]
        rolling = self._rig_sweep_time_ns() is not None
        spins = {
            float(rl.renderer.sensor_spec.spinning_frequency_hz)
            for rl in self._rig_lidars
        }

        def render_sector(
            start: TimestampedPose, end: TimestampedPose, k: int, state: dict
        ) -> list[dict]:
            b2w_end = build_base_to_world_from_pose(end.position, end.rotation, device)
            if rolling:
                b2w = build_base_to_world_from_pose(
                    start.position, start.rotation, device
                )
                b2w_pair = (b2w, b2w_end)
            else:
                b2w_pair = (b2w_end, None)
            with torch.no_grad():
                if "shared" not in state:
                    # One rig-wide LOD gather per revolution, shared by every
                    # sensor and every sector.
                    state["shared"] = gather_lidar_rig(
                        renderers,
                        b2w_pair[0],
                        scene,
                        base_to_world_end=b2w_pair[1],
                    )
                shared = state["shared"]
                if shared is None:
                    return [r._empty_panorama((k, n_sectors)) for r in renderers]
                # sync=False: outputs stay stream-ordered; the one host sync
                # per revolution happens in publish (the .cpu() copy).
                return render_lidars_concurrent(
                    renderers,
                    b2w_pair[0],
                    scene,
                    base_to_world_end=b2w_pair[1],
                    shared=shared,
                    sector=(k, n_sectors),
                    sync=False,
                )

        def publish_revolution(
            outputs: list, end_pose: TimestampedPose, stamp_ns: int
        ) -> None:
            t0 = time.monotonic()
            sec, nanosec = divmod(stamp_ns, 1_000_000_000)
            stamp = Time(sec=sec, nanosec=nanosec)
            base_to_world = build_base_to_world_from_pose(
                end_pose.position, end_pose.rotation, device
            )
            n_points = 0
            with torch.no_grad():
                for i, rl in enumerate(self._rig_lidars):
                    panorama = {
                        key: torch.cat(
                            [outputs[s][i][key] for s in range(len(outputs))],
                            dim=1,
                        )
                        for key in outputs[0][i]
                    }
                    records, n = rl.renderer.panorama_to_pointcloud2_data(
                        panorama,
                        drop_threshold=rl.drop_threshold,
                        alpha_threshold=rl.alpha_threshold,
                    )
                    rl.publisher.publish_packed(records, n, stamp=stamp)
                    n_points += n
                self._render_rig_cameras(base_to_world, stamp)
            self._rig_render_count += 1
            if self._rig_render_count <= 5 or self._rig_render_count % 100 == 0:
                logger.info(
                    "Rig sector revolution #%d: %d LiDARs (%d pts) + %d "
                    "cameras, publish+cameras=%.1fms",
                    self._rig_render_count,
                    len(self._rig_lidars),
                    n_points,
                    len(self._rig_cameras),
                    (time.monotonic() - t0) * 1000,
                )

        return _SectorHooks(
            n_sectors=n_sectors,
            spin_period_ns=_spin_period_ns(max(spins)),
            render_sector=render_sector,
            publish_revolution=publish_revolution,
        )

    def _render_rig_cameras(self, base_to_world: torch.Tensor, stamp: Time) -> None:
        """Render + publish every rig camera at one ego pose (shared gather)."""
        if not self._rig_cameras:
            return
        # world→camera = inv(base_to_world @ cam_to_base)
        viewmats = [
            torch.linalg.inv(base_to_world @ rc.cam_to_base) for rc in self._rig_cameras
        ]
        images = render_cameras_concurrent(
            [rc.renderer for rc in self._rig_cameras],
            viewmats,
            [rc.K for rc in self._rig_cameras],
            scene=self._scene,
            camera_names=[rc.name for rc in self._rig_cameras],
        )
        for rc, rgb in zip(self._rig_cameras, images):
            bgr = (rgb.clamp(0.0, 1.0) * 255).byte()[:, :, [2, 1, 0]]
            if rc.image_pub is not None:
                rc.image_pub.publish(bgr.cpu().numpy(), stamp=stamp)
            if rc.info_pub is not None:
                rc.info_pub.publish(stamp=stamp)

    def _render_and_publish_rig(
        self,
        pose: TimestampedPose,
        render_time_ns: int,
        sweep_start: TimestampedPose | None = None,
    ) -> None:
        """One frame for the whole rig: one gather per modality, one stream each.

        ``pose`` is the sweep-END pose (the frame's stamp). Cameras render at
        it directly; the LiDARs additionally get ``sweep_start`` (one spin
        earlier) so their panoramas are motion-compensated over the sweep.
        """
        assert self._scene is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        t0 = time.monotonic()
        base_to_world = build_base_to_world_from_pose(
            pose.position, pose.rotation, self._device
        )
        sec, nanosec = divmod(render_time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)

        n_points = 0
        with torch.no_grad():
            if self._rig_lidars:
                if sweep_start is not None:
                    lidar_start = build_base_to_world_from_pose(
                        sweep_start.position, sweep_start.rotation, self._device
                    )
                    lidar_end = base_to_world
                else:
                    lidar_start, lidar_end = base_to_world, None
                panoramas = render_lidars_concurrent(
                    [rl.renderer for rl in self._rig_lidars],
                    lidar_start,
                    self._scene,
                    base_to_world_end=lidar_end,
                )
                for rl, pano in zip(self._rig_lidars, panoramas):
                    records, n = rl.renderer.panorama_to_pointcloud2_data(
                        pano,
                        drop_threshold=rl.drop_threshold,
                        alpha_threshold=rl.alpha_threshold,
                    )
                    rl.publisher.publish_packed(records, n, stamp=stamp)
                    n_points += n
            t_lidar = time.monotonic()

            self._render_rig_cameras(base_to_world, stamp)
        t_end = time.monotonic()

        self._rig_render_count += 1
        if self._rig_render_count <= 5 or self._rig_render_count % 100 == 0:
            logger.info(
                "Rig frame #%d: %d LiDARs (%d pts) + %d cameras "
                "total=%.1fms (lidar=%.1f camera=%.1f)",
                self._rig_render_count,
                len(self._rig_lidars),
                n_points,
                len(self._rig_cameras),
                (t_end - t0) * 1000,
                (t_lidar - t0) * 1000,
                (t_end - t_lidar) * 1000,
            )

    def _render_and_publish_lidar(
        self,
        pose: TimestampedPose,
        render_time_ns: int,
        sweep_start: TimestampedPose | None = None,
    ) -> None:
        """Render one LiDAR panorama and publish a PointCloud2.

        ``pose`` is the sweep-END pose (the cloud's stamp); ``sweep_start``,
        when given, is the pose one spin earlier and turns on the renderer's
        motion-during-sweep compensation (rolling shutter).
        """
        assert self._lidar_renderer is not None  # noqa: S101
        assert self._pointcloud_pub is not None  # noqa: S101
        assert self._scene is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        t0 = time.monotonic()
        base_to_world_end = build_base_to_world_from_pose(
            pose.position, pose.rotation, self._device
        )
        if sweep_start is not None:
            base_to_world = build_base_to_world_from_pose(
                sweep_start.position, sweep_start.rotation, self._device
            )
        else:
            base_to_world, base_to_world_end = base_to_world_end, None
        with torch.no_grad():
            panorama = self._lidar_renderer.render(
                base_to_world,
                scene=self._scene,
                base_to_world_end=base_to_world_end,
            )
            # Fast path: the point records are packed on the GPU and cross to
            # the host as one contiguous buffer (see panorama_to_pointcloud2_data).
            records, n_points = self._lidar_renderer.panorama_to_pointcloud2_data(
                panorama,
                drop_threshold=self._lidar_drop_threshold,
                alpha_threshold=self._lidar_alpha_threshold,
            )
        t_render = time.monotonic()

        sec, nanosec = divmod(render_time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)
        self._pointcloud_pub.publish_packed(records, n_points, stamp=stamp)
        t_publish = time.monotonic()

        self._lidar_render_count += 1
        if self._lidar_render_count <= 5 or self._lidar_render_count % 100 == 0:
            logger.info(
                "LiDAR render #%d: %d points total=%.1fms (render=%.1f publish=%.1f)",
                self._lidar_render_count,
                n_points,
                (t_publish - t0) * 1000,
                (t_render - t0) * 1000,
                (t_publish - t_render) * 1000,
            )
