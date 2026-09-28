from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActorConfig:
    """One rigid dynamic-object instance spawned from the scene's asset bank.

    ``asset_id`` names an asset in the bundle's ``actor_assets.json``; ``name``
    is the key it gets in :attr:`splatsim.scene.Scene.rigid_bodies`, so a
    scenario can pose it by name. Several actors may share one ``asset_id`` —
    that is the point of the bank.

    The initial pose is a convenience for static props and spawn points; a
    scenario runner normally calls ``scene.set_pose(name, ...)`` every frame.
    ``rotation`` is ``wxyz``, matching every other pose in splatsim's config
    (scene-bundle poses are ``xyzw`` and are reordered on the way in).
    ``world_position`` selects the frame ``position`` is given in: ``True``
    means the bundle's ENU world frame and the background's tile-local centroid
    is subtracted on load; ``False`` means it is already tile-local.
    """

    asset_id: str
    name: str = ""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    world_position: bool = True

    def __post_init__(self) -> None:
        # An unknown or empty asset_id is reported by ActorAssetLibrary with the
        # available ids listed, which beats anything this could say.
        if not self.name:
            self.name = self.asset_id
