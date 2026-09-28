from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors
from splatsim.lod import LodIndex, LodManager


class Background:
    """Loads a v2 scene USDZ as GPU-ready Gaussian tensors."""

    def __init__(
        self,
        source_path: str | Path,
        *,
        device: torch.device = torch.device("cuda"),
        use_sh: bool = False,
        lod_manager: LodManager | None = None,
    ) -> None:
        path = Path(source_path)
        if path.suffix.lower() != ".usdz":
            raise ValueError(
                f"{path}: unsupported background source; only v2 scene USDZ "
                "(.usdz) is supported"
            )

        # 3dgs_io scene USDZ bundles scene.json + chunks/*.spz; load each
        # SPZ chunk (already baked in the ENU world frame) as a tensor chunk.
        # The scene's ecef_anchor is the ENU world→ECEF transform.
        from splatsim._usdz import load_spz_scene

        # Kept so consumers can re-open the same bundle for the sidecar data
        # the background itself does not need — e.g. the rigid actor asset bank
        # (see splatsim.actor_assets.ActorAssetLibrary).
        self._source_path = path
        self._use_sh = use_sh

        tensors, anchor = load_spz_scene(path, device, use_sh=use_sh)
        self._ecef_rotation = anchor[:3, :3].copy()
        self._ecef_translation = anchor[:3, 3].copy()

        # Re-center to the cloud's centroid for numerical stability. This
        # offset is the gaussians' own centroid in the ENU world frame, not
        # the ECEF origin (see ``ecef_translation``).
        self._tile_local_centroid = tensors.means.mean(dim=0).clone()
        tensors.means = tensors.means - self._tile_local_centroid

        # LOD: sort by importance and compute tier boundaries.
        self._lod_index: LodIndex | None = None
        if lod_manager is not None:
            tensors, self._lod_index = lod_manager.precompute(tensors)

        self._tensors = tensors

    @property
    def source_path(self) -> Path:
        """The scene USDZ this background was loaded from."""
        return self._source_path

    @property
    def use_sh(self) -> bool:
        """Whether view-dependent SH bands were kept when loading."""
        return self._use_sh

    @property
    def lod_index(self) -> LodIndex | None:
        return self._lod_index

    @property
    def tile_local_centroid(self) -> Tensor:
        """Tile-local centroid subtracted from gaussian means for numerical stability.

        This is ``means.mean(dim=0)`` in the Z-up ENU world frame of the loaded
        v2 USDZ gaussians. It is **not** an ECEF translation. Renderers that
        consume world-frame poses must add it back to the translation column of
        their world-to-camera matrices to align with the re-centered cloud.

        For the scene's ECEF translation, see :attr:`ecef_translation`.
        """
        return self._tile_local_centroid

    @property
    def ecef_translation(self) -> np.ndarray:
        """ECEF translation of the scene anchor, in meters (3,) ``float64``.

        Read from the USDZ scene's ``world.ecef_anchor``; never applied to the
        gaussians.
        """
        return self._ecef_translation

    @property
    def ecef_rotation(self) -> np.ndarray:
        """ECEF rotation of the scene anchor, 3x3 ``float64``.

        Read from the USDZ scene's ``world.ecef_anchor``; never applied to the
        gaussians.
        """
        return self._ecef_rotation

    @property
    def tensors(self) -> GaussianTensors:
        return self._tensors

    @property
    def num_gaussians(self) -> int:
        return self._tensors.means.shape[0]
