from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import spz

from splatsim._conversions import (
    MAX_NON_YAW_RAD,
    GaussianTensors,
    apply_rigid_transform,
    cloud_to_tensors,
    tilt_from_quat,
)
from splatsim.lod import LodIndex, LodManager


class RigidBody:
    """A rigid Gaussian object that can be positioned and rotated in the scene."""

    def __init__(
        self,
        source: str | Path | GaussianTensors,
        *,
        device: torch.device = torch.device("cuda"),
        use_sh: bool = False,
        lod_manager: LodManager | None = None,
        lod_index: LodIndex | None = None,
        rotate_sh: bool = False,
    ) -> None:
        """Load a rigid body from a file, or wrap tensors already on the device.

        Passing :class:`~splatsim._conversions.GaussianTensors` is how
        :class:`~splatsim.actor_assets.ActorAssetLibrary` spawns instances: the
        Gaussians come from a scene bundle's asset bank rather than a
        standalone file, and every instance of one asset shares them.
        ``use_sh`` applies to the file path only — tensors arrive already
        converted.

        ``lod_manager`` sorts the Gaussians by importance and builds the tier
        index, which REORDERS them: each instance that precomputes ends up
        holding its own copy of the whole cloud. Pass ``lod_index`` instead
        (with the already-sorted tensors ``precompute`` returned) to reuse one
        instance's work and keep sharing the upload — the sort is pose- and
        instance-independent, so one index is valid for every instance of an
        asset. Passing both is an error.
        """
        if lod_manager is not None and lod_index is not None:
            raise ValueError("pass either lod_manager or lod_index, not both")

        base_tensors = (
            source
            if isinstance(source, GaussianTensors)
            else cloud_to_tensors(_load_cloud(source), device, use_sh=use_sh)
        )

        # LOD: sort base tensors by importance and store tier boundaries.
        self._lod_index: LodIndex | None = lod_index
        if lod_manager is not None:
            base_tensors, self._lod_index = lod_manager.precompute(base_tensors)

        self._base_tensors = base_tensors
        self._device = device
        self._rotate_sh = rotate_sh
        self.position = torch.zeros(3, device=device, dtype=torch.float32)
        self.rotation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )  # identity quaternion (wxyz)

    def set_pose(
        self,
        position: tuple[float, float, float] | Tensor,
        rotation: tuple[float, float, float, float] | Tensor | None = None,
    ) -> None:
        """Set the world-space pose of this rigid body.

        Args:
            position: (x, y, z) translation.
            rotation: (w, x, y, z) quaternion. If None, keeps current rotation.
        """
        if isinstance(position, tuple):
            self.position = torch.tensor(
                position, device=self._device, dtype=torch.float32
            )
        else:
            self.position = position.to(device=self._device, dtype=torch.float32)

        if rotation is not None:
            if isinstance(rotation, tuple):
                self.rotation = torch.tensor(
                    rotation, device=self._device, dtype=torch.float32
                )
            else:
                self.rotation = rotation.to(device=self._device, dtype=torch.float32)

    @property
    def base_tensors(self) -> GaussianTensors:
        """Return the untransformed (local-space) base tensors."""
        return self._base_tensors

    @property
    def sh_rotation_tilt(self) -> float:
        """Radians this body's pose departs from a pure yaw.

        Colour SH is re-expressed by rotating about ``+Z`` only, which is exact
        for the yaw-only poses road vehicles have; past
        :data:`~splatsim._conversions.MAX_NON_YAW_RAD` the view-dependent
        colour is an approximation. ``0.0`` when the body does not rotate its
        SH at all, since nothing is being approximated.

        Reads a device scalar back to the host, so treat it as a setup-time or
        assertion-time check rather than something to poll every frame.
        """
        if not self._rotate_sh:
            return 0.0
        return float(tilt_from_quat(self.rotation))

    @property
    def sh_rotation_is_exact(self) -> bool:
        """True while the current pose is within :data:`MAX_NON_YAW_RAD` of a yaw."""
        return self.sh_rotation_tilt <= MAX_NON_YAW_RAD

    def posed(self, base: GaussianTensors | None = None) -> GaussianTensors:
        """Apply this body's current pose to ``base`` (default: its own tensors).

        The one place a body's transform is spelled out. ``base`` lets a caller
        that has already thinned the Gaussians — the LOD gather in
        :meth:`splatsim.scene.Scene.collect_tensors` — pose the subset without
        having to re-derive how this body wants to be transformed.
        """
        return apply_rigid_transform(
            self._base_tensors if base is None else base,
            self.position,
            self.rotation,
            rotate_sh=self._rotate_sh,
        )

    @property
    def tensors(self) -> GaussianTensors:
        """Return transformed tensors with current pose applied."""
        return self.posed()

    @property
    def lod_index(self) -> LodIndex | None:
        return self._lod_index

    @property
    def num_gaussians(self) -> int:
        return self._base_tensors.means.shape[0]


def _load_cloud(
    source: str | Path,
) -> spz.GaussianCloud:  # ty: ignore[unresolved-attribute]
    """Load a GaussianCloud from file, auto-detecting format by extension."""
    import importlib

    _3dgs_io = importlib.import_module("3dgs_io")
    load_gltf = _3dgs_io.load_gltf
    load_ply = _3dgs_io.load_ply
    load_spz = _3dgs_io.load_spz

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix == ".spz":
        return load_spz(str(path))
    elif suffix in (".glb", ".gltf"):
        return load_gltf(str(path))
    elif suffix == ".ply":
        return load_ply(str(path))
    else:
        msg = f"Unsupported file format: {suffix}. Use .spz, .glb, .gltf, or .ply."
        raise ValueError(msg)
