"""Attach-only scenario with SplatSim camera on an externally-spawned ego.

The ego vehicle is expected to be spawned and driven externally (e.g. by
``autoware_carla_interface``). This scenario binds to the existing ego
actor via ``role_name``, attaches a SplatSim Gaussian Splatting camera,
and publishes rendered images to ROS 2 topics via CycloneDDS.

Usage::

    source .env
    uv run attach-scenario splatsim.scene_yaml=/path/to/scene.yaml
"""

import logging
from dataclasses import dataclass

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    EgoConfig,
    TimeoutCondition,
)

from splatsim.carla_integration import (
    SplatSimCameraSensorConfig,
    SplatSimConfig,
    SplatSimScenario,
)

logger = logging.getLogger(__name__)


@dataclass
class AttachOnlyConfig:
    """Parameters for the attach-only scenario."""

    name: str = "attach_only"
    timeout_seconds: float = 3600.0


class AttachOnlyScenario(SplatSimScenario):
    """Attach a SplatSim camera to an externally-spawned ego, publish to ROS 2."""

    def __init__(
        self,
        ego_config: EgoConfig,
        *,
        config: AttachOnlyConfig,
        splatsim_config: SplatSimConfig,
    ) -> None:
        super().__init__(
            ego_config,
            splatsim_config.scene_yaml,
            attach_to_existing_ego=True,
        )
        self._config = config
        self._splatsim_cfg = splatsim_config

    def setup(self) -> None:
        """Attach SplatSim camera to the externally-spawned ego and publish frames."""
        scfg = self._splatsim_cfg

        # Attach SplatSim camera to ego vehicle (1.5m above base_link)
        self.attach_splatsim_camera(
            EGO_ROLE_NAME,
            sensor_config=SplatSimCameraSensorConfig(position_z=1.5),
            label="ego_splatsim_camera",
            dds_participant=self.dds_participant,
            image_topic=scfg.image_topic,
            camera_info_topic=scfg.camera_info_topic,
            frame_id=scfg.frame_id,
            compress_format=scfg.compress_format,
        )

        # Register the shared publish callback (camera frames).
        # Initial-pose / engage / IMU / vehicle control are owned by
        # autoware_carla_interface and intentionally not handled here.
        self.register_post_tick(self.publish_ros_topics)

        # Suppress verbose position/tick logging from base scenario
        logging.getLogger("autoware_carla_scenario.scenario_base").setLevel(
            logging.WARNING
        )
        logging.getLogger("autoware_carla_scenario.scenario_runner").setLevel(
            logging.WARNING
        )

        logger.info(
            "Attached to externally-spawned ego (role_name='%s'). "
            "SplatSim camera attached -> publishing to %s. "
            "Holding for %.1f s ...",
            EGO_ROLE_NAME,
            scfg.image_topic,
            self._config.timeout_seconds,
        )

        self.register_pass_condition(
            TimeoutCondition(self._config.timeout_seconds, label="attach_hold_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` -- termination is driven by the timeout condition."""
        return False
