"""Entry point for the odaibatest attach-only scenario.

Registers :class:`AttachOnlyScenario` via the upstream scenario registry
and delegates all orchestration to
:func:`autoware_carla_scenario.examples.run.run_scenario`.

Usage::

    source .env
    uv run attach-scenario
    uv run attach-scenario scenario.timeout_seconds=60.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import hydra

from autoware_carla_scenario import BaseScenario, EgoConfig
from autoware_carla_scenario.examples.run import build_ego_and_spawn, run_scenario

from splatsim.carla_integration import SplatSimConfig

from .attach_only import AttachOnlyConfig, AttachOnlyScenario

logger = logging.getLogger(__name__)


def _find_env_file() -> Path | None:
    """Locate .env file, checking ROS2 share dir then source-tree fallback."""
    try:
        from ament_index_python.packages import (
            get_package_share_directory,
        )

        share = Path(get_package_share_directory("splatsim"))
        candidate = share / ".env"
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    # Fallback for editable / uv-run usage
    candidate = Path(__file__).resolve().parents[3] / ".env"
    return candidate if candidate.is_file() else None


def _resolve_relative_paths(base_dir: Path) -> None:
    """Resolve relative file paths in env vars to absolute using base_dir."""
    import os

    for key in ("ODAIBATEST_LANELET2_PATH", "ODAIBATEST_XODR_PATH"):
        val = os.environ.get(key)
        if val and not Path(val).is_absolute() and (base_dir / val).is_file():
            os.environ[key] = str(base_dir / val)


_env_file = _find_env_file()
if _env_file is not None:
    load_dotenv(_env_file, override=True)
    _resolve_relative_paths(_env_file.parent)


def _build_scenario_from_cfg(
    cfg: DictConfig,
) -> tuple[EgoConfig, BaseScenario]:
    """Build EgoConfig + AttachOnlyScenario from the full Hydra config."""
    ego, _spawn_pose, _ground_projection = build_ego_and_spawn(cfg)

    scenario_raw = OmegaConf.to_container(cfg.scenario, resolve=True)
    assert isinstance(scenario_raw, dict)  # noqa: S101
    config = AttachOnlyConfig(**{str(k): v for k, v in scenario_raw.items()})

    splatsim_raw = OmegaConf.to_container(cfg.splatsim, resolve=True)
    assert isinstance(splatsim_raw, dict)  # noqa: S101
    splatsim_config = SplatSimConfig(**{str(k): v for k, v in splatsim_raw.items()})

    return ego, AttachOnlyScenario(
        ego,
        config=config,
        splatsim_config=splatsim_config,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point -- delegates to the upstream run_scenario."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    result = run_scenario(cfg, build_scenario_fn=_build_scenario_from_cfg)
    sys.exit(0 if result.passed else 1)


def main() -> None:
    """CLI entry point."""
    _hydra_main()


if __name__ == "__main__":
    main()
