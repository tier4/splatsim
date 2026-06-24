"""Base scenario class that combines CARLA APIs with SplatSim scene access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

import carla

from autoware_carla_scenario.scenario_base import BaseScenario, EgoConfig

from splatsim.carla_integration.attach_splatsim_camera import (
    AttachSplatSimCameraAction,
)
from splatsim.carla_integration.geo_transform import GeoTransform, parse_geo_reference
from splatsim.carla_integration.splatsim_camera import (
    SplatSimCameraSensor,
    SplatSimCameraSensorConfig,
)
from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher
from splatsim.cyclonedds.msg_types import Time
from splatsim.scene import Scene

if TYPE_CHECKING:
    from cyclonedds.domain import DomainParticipant

    from autoware_carla_scenario.actions import TickTiming
    from autoware_carla_scenario.conditions.base import BaseCondition
    from autoware_carla_scenario.coordinate import (
        GroundProjectionConfig,
        Lanelet2Pose,
    )
    from autoware_carla_scenario.entity import AutowareEntity
    from autoware_carla_scenario.entity_role import EntityRole
    from splatsim.dataclass import SceneConfig

logger = logging.getLogger(__name__)


class SplatSimScenario(BaseScenario):
    """Abstract base class for CARLA scenarios with SplatSim rendering.

    Extends :class:`BaseScenario` so that subclasses have full access to
    the CARLA client, world, ego vehicle helpers **and** a SplatSim
    :class:`~splatsim.scene.Scene` for Gaussian Splatting rendering.

    The geographic transform between CARLA world coordinates and the
    3DGS tile-local frame is computed automatically from the xodr
    ``<geoReference>`` and the tileset.json ECEF transform.

    Parameters
    ----------
    ego_config:
        Ego vehicle spawn configuration (forwarded to ``BaseScenario``).
    scene:
        A :class:`~splatsim.scene.Scene` instance, a
        :class:`~splatsim.dataclass.SceneConfig`, or a path to a YAML
        scene file.  Configs and paths are built into a ``Scene``
        automatically.
    spawn_pose:
        Optional Lanelet2 pose for ego spawn.
    ground_projection:
        Ground-projection settings for snapping to the CARLA road surface.
    random_seed:
        Seed for the CARLA TrafficManager random device.
    sensor_config:
        Default camera / renderer configuration used by
        :meth:`attach_splatsim_camera`.  Falls back to
        :class:`SplatSimCameraSensorConfig` defaults when *None*.
    """

    def __init__(
        self,
        ego_config: EgoConfig,
        scene: Scene | SceneConfig | str | Path,
        *,
        spawn_pose: Lanelet2Pose | None = None,
        ground_projection: GroundProjectionConfig | None = None,
        random_seed: int = BaseScenario.DEFAULT_RANDOM_SEED,
        sensor_config: SplatSimCameraSensorConfig | None = None,
        attach_to_existing_ego: bool = False,
    ) -> None:
        super().__init__(
            ego_config,
            spawn_pose=spawn_pose,
            ground_projection=ground_projection,
            random_seed=random_seed,
            attach_to_existing_ego=attach_to_existing_ego,
        )

        if isinstance(scene, Scene):
            self._scene = scene
        else:
            self._scene = Scene.from_config(scene)

        self._sensor_config = sensor_config
        self._geo_transform: GeoTransform | None = None

        # Registered camera actions and their publishers.
        self._camera_entries: list[_CameraEntry] = []

        # Capture the ego entity reference when ScenarioRunner calls
        # ego_type().  This happens before setup(), so ego_entity is
        # available in setup_autoware() and subclass setup() methods.
        self._ego_entity: AutowareEntity | None = None
        _original_type = self.ego_type

        def _capture_factory() -> AutowareEntity:
            entity: AutowareEntity = _original_type()  # ty: ignore[invalid-assignment]
            self._ego_entity = entity
            return entity

        self.ego_type = _capture_factory  # ty: ignore[invalid-assignment]

    # -- public properties ---------------------------------------------------

    @property
    def ego_entity(self) -> AutowareEntity | None:
        """The captured ego entity, available after ScenarioRunner calls ``ego_type()``."""
        return self._ego_entity

    @property
    def scene(self) -> Scene:
        """The Gaussian Splatting scene used for rendering."""
        return self._scene

    @property
    def geo_transform(self) -> GeoTransform:
        """The CARLA <-> tile-local coordinate transform.

        Created lazily on first access from the CARLA map's xodr
        ``<geoReference>`` and the scene's tileset ECEF transform.
        """
        if self._geo_transform is None:
            self._geo_transform = self._build_geo_transform()
        return self._geo_transform

    # -- Autoware integration ------------------------------------------------

    def setup_autoware(self) -> None:
        """Initialize Autoware integration for the ego entity.

        Call this in subclass ``setup()`` after ``_setup_ego_spawn()``.
        Performs the following:

        * Publishes ``/initialpose3d`` from the spawn pose.
        * Publishes ``localization_initialization_state=INITIALIZED``
          (bypasses NDT/EKF since CARLA bridge publishes pose directly).
        * Patches the hand-brake release for PARK -> DRIVE transitions.
        * Registers an auto-engage action that keeps publishing
          ``engage=True`` until the entity confirms engagement.
        """
        from autoware_carla_scenario.actions import EngageAction  # noqa: PLC0415
        from autoware_carla_scenario.conditions import (  # noqa: PLC0415
            AutowareStateCondition,
            AutowareStateField,
        )

        from splatsim.carla_integration.autoware_bridge import (  # noqa: PLC0415
            patch_motion_control,
            publish_initialpose,
            publish_localization_initialized,
        )

        assert self._spawn_pose is not None  # noqa: S101
        publish_initialpose(self.dds_participant, self._spawn_pose)
        publish_localization_initialized(self.dds_participant)

        assert self._ego_entity is not None  # noqa: S101
        patch_motion_control(self._ego_entity)

        not_engaged = AutowareStateCondition(
            entity=self._ego_entity,
            field=AutowareStateField.ENGAGED,
            expected=False,
            label="ego_not_engaged",
        )
        self.register_post_tick(
            EngageAction(
                self._ego_entity,
                value=True,
                condition=not_engaged,
                once=False,
                label="auto_engage",
            )
        )

        logger.info(
            "Autoware integration initialized (initialpose, localization, engage)"
        )

    # -- convenience helpers -------------------------------------------------

    def attach_splatsim_camera(
        self,
        entity_name: Union[EntityRole, str],
        *,
        sensor_config: SplatSimCameraSensorConfig | None = None,
        condition: BaseCondition | None = None,
        timing: TickTiming | None = None,
        label: str = "attach_splatsim_camera",
        once: bool = True,
        dds_participant: DomainParticipant | None = None,
        image_topic: str = "/splatsim/image_raw",
        camera_info_topic: str = "/splatsim/camera_info",
        frame_id: str = "splatsim_camera",
        compress_format: str = "",
    ) -> AttachSplatSimCameraAction:
        """Create and register an action that attaches a SplatSim camera.

        The action is automatically registered via :meth:`register_post_tick`.
        If *dds_participant* is provided, :class:`ImagePublisher` and
        :class:`CameraInfoPublisher` are created and the camera is included
        in :meth:`publish_ros_topics`.

        Parameters
        ----------
        entity_name:
            Role name of the CARLA actor to attach the camera to
            (e.g. ``EntityRole.ego()``).
        sensor_config:
            Per-call override.  Falls back to the instance-level
            ``sensor_config`` passed at construction, then to
            :class:`SplatSimCameraSensorConfig` defaults.
        condition:
            Optional trigger condition for the action.
        timing:
            When the action fires (pre-tick or post-tick).  Defaults to
            ``TickTiming.POST_TICK``.
        label:
            Human-readable action identifier.
        once:
            If *True* (default), the action fires at most once.
        dds_participant:
            CycloneDDS domain participant for ROS 2 publishing.  When
            *None*, no publishers are created (render-only mode).
        image_topic:
            ROS 2 topic name for ``sensor_msgs/Image``.
        camera_info_topic:
            ROS 2 topic name for ``sensor_msgs/CameraInfo``.
        frame_id:
            ``frame_id`` written into each published message header.

        Returns
        -------
        AttachSplatSimCameraAction
            The registered action (useful for accessing the sensor after
            attachment via ``action.sensor``).
        """
        from autoware_carla_scenario.actions import TickTiming as _TickTiming

        if timing is None:
            timing = _TickTiming.POST_TICK

        config = sensor_config or self._sensor_config

        action = AttachSplatSimCameraAction(
            entity_name,
            self._scene,
            self.geo_transform,
            sensor_config=config,
            condition=condition,
            timing=timing,
            label=label,
            once=once,
        )
        self.register_post_tick(action)

        # Create publishers if a DDS participant was provided.
        image_pub: ImagePublisher | None = None
        if dds_participant is not None:
            image_pub = ImagePublisher(
                dds_participant,
                topic_name=image_topic,
                frame_id=frame_id,
                compress_format=compress_format,
            )

        self._camera_entries.append(
            _CameraEntry(
                action=action,
                image_pub=image_pub,
                camera_info_pub=None,
                dds_participant=dds_participant,
                camera_info_topic=camera_info_topic,
                frame_id=frame_id,
            )
        )
        return action

    def publish_ros_topics(self, world: carla.World) -> None:
        """Render all attached SplatSim cameras and publish to ROS 2.

        This should be registered as a post-tick callback via
        :meth:`register_post_tick`.  Image and CameraInfo messages
        share the same timestamp derived from the CARLA world snapshot.

        Parameters
        ----------
        world:
            The current CARLA world (passed by the tick loop).
        """
        # CARLA simulation timestamp → ROS 2 Time
        snapshot = world.get_snapshot()
        stamp = _carla_timestamp_to_ros(snapshot.timestamp)

        for entry in self._camera_entries:
            sensor = entry.action.sensor
            if sensor is None:
                continue
            assert isinstance(sensor, SplatSimCameraSensor)  # noqa: S101

            image = sensor.get_image()

            if entry.image_pub is not None:
                entry.image_pub.publish(image, stamp=stamp)

            # Lazily create CameraInfo publisher (needs intrinsics from
            # the sensor config, available only after attachment).
            if entry.camera_info_pub is None and entry.dds_participant is not None:
                entry.camera_info_pub = CameraInfoPublisher(
                    entry.dds_participant,
                    sensor.config,
                    topic_name=entry.camera_info_topic,
                    frame_id=entry.frame_id,
                )

            if entry.camera_info_pub is not None:
                entry.camera_info_pub.publish(stamp=stamp)

    # -- internals -----------------------------------------------------------

    def _build_geo_transform(self) -> GeoTransform:
        """Build :class:`GeoTransform` from the CARLA map and scene data."""
        import numpy as _np  # noqa: PLC0415

        # 1. Parse (lat_0, lon_0) from CARLA xodr GeoReference
        xodr_xml = self.world.get_map().to_opendrive()
        proj_origin = parse_geo_reference(xodr_xml)

        # 2. Get tileset ECEF rotation/translation from Background
        bg = self._scene.background
        if bg is None:
            raise RuntimeError(
                "Cannot build GeoTransform: scene has no Background "
                "(no tileset.json ECEF data)"
            )

        # 3. Get tile-local centroid (torch Tensor on GPU → numpy float64)
        tile_origin = bg.tile_local_centroid.detach().cpu().numpy().astype(_np.float64)

        return GeoTransform(
            proj_origin=proj_origin,
            ecef_rotation=bg.ecef_rotation,
            ecef_translation=bg.ecef_translation,
            tile_origin=tile_origin,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _CameraEntry:
    """Bookkeeping for one attached SplatSim camera + its publishers."""

    __slots__ = (
        "action",
        "camera_info_pub",
        "camera_info_topic",
        "dds_participant",
        "frame_id",
        "image_pub",
    )

    def __init__(
        self,
        *,
        action: AttachSplatSimCameraAction,
        image_pub: ImagePublisher | None,
        camera_info_pub: CameraInfoPublisher | None,
        dds_participant: DomainParticipant | None,
        camera_info_topic: str,
        frame_id: str,
    ) -> None:
        self.action = action
        self.image_pub = image_pub
        self.camera_info_pub = camera_info_pub
        self.dds_participant = dds_participant
        self.camera_info_topic = camera_info_topic
        self.frame_id = frame_id


def _carla_timestamp_to_ros(ts: carla.Timestamp) -> Time:
    """Convert a CARLA simulation timestamp to a ROS 2 ``Time``.

    Uses ``elapsed_seconds`` (simulation time since episode start) so
    that all sensors referencing the same tick produce identical stamps.
    """
    elapsed_ns = int(ts.elapsed_seconds * 1e9)
    sec, nanosec = divmod(elapsed_ns, 10**9)
    return Time(sec=sec, nanosec=nanosec)
