"""Level-of-Detail (LOD) system for Gaussian Splatting.

Pre-computes importance-based LOD tiers at load time and provides
per-frame tensor filtering at render time using density-adaptive
octree spatial partitioning.  Each leaf cell applies LOD independently
based on camera-to-cell distance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors
from splatsim.dataclass.lod_config import LodConfig

logger = logging.getLogger(__name__)


@dataclass
class LodIndex:
    """Pre-computed LOD metadata for a single GaussianTensors instance."""

    tier_counts: list[int]
    tier_max_distances: list[float]
    tier_max_distances_t: Tensor = field(repr=False)
    """[T] float32 GPU tensor of tier max distances (cached for filter)."""

    cell_centers: Tensor = field(repr=False)
    """[C, 3] center of each leaf cell."""
    cell_ranges: Tensor = field(repr=False)
    """[C, 2] int64 (start, end) into the sorted tensors."""
    cell_tier_counts: Tensor = field(repr=False)
    """[C, T] int64 per-cell Gaussian count for each tier."""

    cell_radius: Tensor = field(repr=False)
    """[C] float32 half-diagonal of each leaf cell's AABB (bounds every member
    Gaussian center's distance from the cell center)."""

    cell_max_scale: Tensor = field(repr=False)
    """[C] float32 max per-Gaussian ``scales.max(dim=1)`` within the cell (used
    to bound the 3-sigma cull margin for whole-cell max-range culling)."""


class LodManager:
    """Manages LOD pre-computation and per-frame tier selection."""

    def __init__(self, config: LodConfig) -> None:
        self._tiers = sorted(config.tiers, key=lambda t: t.max_distance)
        self._max_gpc = config.max_gaussians_per_cell
        self._prev_tier_idx: Tensor | None = None

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute(self, tensors: GaussianTensors) -> tuple[GaussianTensors, LodIndex]:
        """Sort Gaussians by importance and build octree LOD metadata."""
        n = tensors.means.shape[0]
        device = tensors.means.device

        max_scale = tensors.scales.max(dim=1).values  # [N]
        importance = max_scale * tensors.opacities  # [N]

        # --- octree subdivision (CPU) ---
        means_cpu = tensors.means.detach().cpu()
        aabb_min = means_cpu.min(dim=0).values
        aabb_max = means_cpu.max(dim=0).values
        all_indices = torch.arange(n)

        leaves: list[tuple[Tensor, Tensor, Tensor]] = []  # (indices, lo, hi)
        self._subdivide(means_cpu, all_indices, aabb_min, aabb_max, leaves, depth=0)

        # --- sort within each leaf by importance (descending) ---
        importance_cpu = importance.detach().cpu()
        ordered_indices: list[Tensor] = []
        cell_centers_list: list[Tensor] = []
        cell_ranges_list: list[tuple[int, int]] = []
        offset = 0

        max_scale_cpu = max_scale.detach().cpu()
        cell_radius_list: list[float] = []
        cell_max_scale_list: list[float] = []
        for leaf_indices, lo, hi in leaves:
            leaf_imp = importance_cpu[leaf_indices]
            local_order = leaf_imp.argsort(descending=True)
            ordered = leaf_indices[local_order]
            ordered_indices.append(ordered)

            count = len(ordered)
            cell_ranges_list.append((offset, offset + count))
            cell_centers_list.append((lo + hi) / 2.0)
            cell_radius_list.append(float(torch.norm((hi - lo) / 2.0)))
            cell_max_scale_list.append(float(max_scale_cpu[leaf_indices].max()))
            offset += count

        final_order = torch.cat(ordered_indices).to(device)
        sorted_tensors = tensors[final_order]

        # --- build cell metadata ---
        num_cells = len(leaves)
        num_tiers = len(self._tiers)

        cell_centers = torch.stack(cell_centers_list).to(
            device=device, dtype=torch.float32
        )
        cell_ranges = torch.tensor(cell_ranges_list, device=device, dtype=torch.int64)
        cell_counts = cell_ranges[:, 1] - cell_ranges[:, 0]

        cell_tier_counts = torch.zeros(
            num_cells, num_tiers, device=device, dtype=torch.int64
        )
        for j, tier in enumerate(self._tiers):
            cell_tier_counts[:, j] = (
                (cell_counts.float() * tier.fraction).clamp(min=1).long()
            )

        # Global aggregates used for logging
        tier_counts = [max(1, int(n * t.fraction)) for t in self._tiers]
        tier_max_distances = [t.max_distance for t in self._tiers]

        return sorted_tensors, LodIndex(
            tier_counts=tier_counts,
            tier_max_distances=tier_max_distances,
            tier_max_distances_t=torch.tensor(
                tier_max_distances, device=device, dtype=torch.float32
            ),
            cell_centers=cell_centers,
            cell_ranges=cell_ranges,
            cell_tier_counts=cell_tier_counts,
            cell_radius=torch.tensor(
                cell_radius_list, device=device, dtype=torch.float32
            ),
            cell_max_scale=torch.tensor(
                cell_max_scale_list, device=device, dtype=torch.float32
            ),
        )

    def _subdivide(
        self,
        means: Tensor,
        indices: Tensor,
        lo: Tensor,
        hi: Tensor,
        leaves: list[tuple[Tensor, Tensor, Tensor]],
        depth: int,
        max_depth: int = 10,
    ) -> None:
        """Recursively subdivide a cell into octants."""
        if len(indices) <= self._max_gpc or depth >= max_depth:
            if len(indices) > 0:
                leaves.append((indices, lo.clone(), hi.clone()))
            return

        mid = (lo + hi) / 2.0
        pts = means[indices]

        for octant in range(8):
            child_lo = lo.clone()
            child_hi = hi.clone()
            mask = torch.ones(len(indices), dtype=torch.bool)
            for axis in range(3):
                if octant & (1 << axis):
                    child_lo[axis] = mid[axis]
                    mask &= pts[:, axis] >= mid[axis]
                else:
                    child_hi[axis] = mid[axis]
                    mask &= pts[:, axis] < mid[axis]

            child_indices = indices[mask]
            if len(child_indices) > 0:
                self._subdivide(
                    means,
                    child_indices,
                    child_lo,
                    child_hi,
                    leaves,
                    depth + 1,
                    max_depth,
                )

    # ------------------------------------------------------------------
    # Per-frame filter
    # ------------------------------------------------------------------

    def filter(
        self,
        tensors: GaussianTensors,
        lod_index: LodIndex,
        camera_position: Tensor,
        count_scale: float = 1.0,
        lidar_view: bool = False,
        max_distance: float | None = None,
        max_distance_sigmas: float = 3.0,
    ) -> GaussianTensors:
        """Select the appropriate LOD tier per cell and return filtered tensors.

        Args:
            camera_position: ``[3]`` float32 tensor on the same device, or
                ``[S, 3]`` for a rig of S sensors. With several positions each
                cell takes its distance to the NEAREST of them, so the selected
                tier is never coarser than any individual sensor would have
                chosen — the gathered set is a superset of every sensor's own
                per-sensor selection. That is what lets one gather serve a whole
                rig without silently decimating a sensor's near field.
            count_scale: Extra per-cell decimation applied on top of the tier
                selection (``<1`` keeps that fraction of each cell's
                importance-sorted Gaussians, min 1). Lets a memory/throughput
                constrained consumer (e.g. the 360° LiDAR, which cannot azimuth-
                cull) thin the scene further than the camera-tuned tiers without
                changing the shared LOD config. ``1.0`` = no change.
            lidar_view: Gather a LiDAR-minimal view: the source's static
                ``lidar_mask`` is intersected with the LOD selection *before*
                the gather (one fused gather instead of LOD-gather followed by
                a second full mask-gather in the renderer), and the ``colors``
                SH block — untouched by the LiDAR path whenever per-Gaussian
                ``intensity_raw`` exists — is replaced by a zero-copy expanded
                placeholder. The returned tensors carry ``lidar_mask=None``
                (already applied).
            max_distance: When set, whole cells provably beyond this camera
                distance are dropped before the gather: a cell is culled when
                ``dist(cell_center) - cell_radius > max_distance +
                max_distance_sigmas * cell_max_scale`` — exactly the Gaussians
                the LiDAR renderer's per-Gaussian radial cull (same sigma
                margin) would discard afterwards, so the final render is
                unchanged while the gather shrinks to the in-range half of the
                scene.
            max_distance_sigmas: Sigma margin matching the downstream
                per-Gaussian cull (``cull_scale_sigmas``).
        """
        device = lod_index.cell_centers.device
        num_cells = lod_index.cell_centers.shape[0]

        # 1. Camera-to-cell distances. For a rig ([S, 3]) take the nearest
        # sensor per cell so no sensor ends up with a coarser tier than it would
        # have picked alone.
        if camera_position.dim() == 2:
            dists = torch.cdist(
                lod_index.cell_centers, camera_position.to(lod_index.cell_centers)
            ).amin(dim=1)  # [C]
        else:
            dists = torch.norm(
                lod_index.cell_centers - camera_position.reshape(1, 3), dim=1
            )  # [C]

        # 2. Tier selection per cell
        #
        # Gate each tier on the cell's NEAREST point (center distance minus the
        # cell's half-diagonal), not its center. Octree leaves are density-
        # adaptive, so a sparse region can stay one large cell (radius tens of
        # metres). Using the center distance made a large cell the sensor was
        # standing at the EDGE of read as "far" and drop to a coarser tier, then
        # snap to the fine tier the instant the moving ego crossed far enough
        # that the center fell under the threshold -- a step change in Gaussian
        # count (e.g. tier0 1.0 -> tier1 0.5 is a 2x density jump) right at the
        # cell boundary. The nearest-point convention matches the whole-cell
        # max-range cull below and removes that boundary pop.
        max_d = lod_index.tier_max_distances_t  # [T]
        near_dists = (dists - lod_index.cell_radius).clamp_min(0.0)  # [C]
        tier_idx = (
            (near_dists.unsqueeze(1) <= max_d.unsqueeze(0))
            .to(torch.int64)
            .argmax(dim=1)
        )  # [C]
        exceeds_all = near_dists > max_d[-1]
        tier_idx[exceeds_all] = len(lod_index.tier_max_distances) - 1

        # 3. Per-cell selected counts
        selected_counts = lod_index.cell_tier_counts[
            torch.arange(num_cells, device=device), tier_idx
        ]  # [C]

        # 3b. Optional extra decimation (keeps each cell's top-`count_scale`
        # fraction; cells stay populated via the min-1 floor).
        if count_scale < 1.0:
            selected_counts = (
                (selected_counts.float() * count_scale).clamp(min=1.0).long()
            )

        # 3c. Whole-cell max-range cull (see the ``max_distance`` docstring).
        if max_distance is not None:
            reachable = (dists - lod_index.cell_radius) <= (
                max_distance + max_distance_sigmas * lod_index.cell_max_scale
            )
            selected_counts = selected_counts * reachable.long()

        # Log tier distribution changes
        if logger.isEnabledFor(logging.INFO):
            changed = self._prev_tier_idx is None or not torch.equal(
                tier_idx, self._prev_tier_idx
            )
            if changed:
                num_tiers = len(lod_index.tier_max_distances)
                parts = []
                for t in range(num_tiers):
                    cnt = (tier_idx == t).sum().item()
                    if cnt > 0:
                        parts.append(f"T{t}:{cnt}")
                total_g = selected_counts.sum().item()
                logger.info(
                    "LOD octree: cells=[%s] total_gaussians=%d/%d",
                    " ".join(parts),
                    total_g,
                    tensors.means.shape[0],
                )
                self._prev_tier_idx = tier_idx.clone()

        # 4. Build the flat index tensor.
        #
        # Each cell contributes its first ``selected_counts[c]`` entries, i.e.
        # ``starts[c] .. starts[c] + selected_counts[c]``. Building that as a
        # dense [num_cells, max_count] matrix and masking it costs a
        # max_count-sized row for EVERY cell -- 406 MiB of int64 at
        # max_count=25k / 2137 cells on a driving scene, allocated, filled and
        # masked every frame, to keep ~3M entries. Instead lay the runs out
        # contiguously: repeat_interleave gives each output slot its cell's
        # start, and subtracting the run's own base offset turns a global arange
        # into a per-run 0,1,2,... ramp. Same values, same order, no dense
        # intermediate.
        starts = lod_index.cell_ranges[:, 0]  # [C]
        total = int(selected_counts.sum())
        if total == 0:
            selected_indices = torch.empty(0, dtype=torch.int64, device=device)
        else:
            run_ends = selected_counts.cumsum(0)  # [C] exclusive-end in output
            run_bases = run_ends - selected_counts  # [C] output offset per cell
            per_slot_start = torch.repeat_interleave(starts, selected_counts)
            per_slot_base = torch.repeat_interleave(run_bases, selected_counts)
            ramp = torch.arange(total, device=device) - per_slot_base
            selected_indices = per_slot_start + ramp

        if lidar_view:
            return _gather_lidar_view(tensors, selected_indices)
        if selected_indices.shape[0] >= tensors.means.shape[0]:
            return tensors
        return tensors[selected_indices]


def _gather_lidar_view(tensors: GaussianTensors, idx: Tensor) -> GaussianTensors:
    """One fused gather of the LiDAR-relevant fields at ``idx`` ∩ ``lidar_mask``.

    The renderer previously did tensors[lod_idx] (all fields, colors included)
    followed by tensors[lidar_mask] (all fields again). Intersecting the mask
    into the index first and skipping the colors block (unused by the LiDAR
    path when ``intensity_raw`` is present) roughly halves the per-frame gather
    bandwidth on SH scenes.
    """
    if tensors.lidar_mask is not None:
        idx = idx[tensors.lidar_mask[idx]]
    n = int(idx.shape[0])
    skip_colors = tensors.intensity_raw is not None
    if skip_colors:
        # Zero-copy placeholder with the right shape/dtype; the LiDAR path only
        # reads ``sh_degree`` and falls back to colors solely when
        # ``intensity_raw`` is absent, which is excluded here.
        colors = tensors.colors[:1].expand(n, *tensors.colors.shape[1:])
    else:
        colors = tensors.colors[idx]
    return GaussianTensors(
        means=tensors.means[idx],
        quats=tensors.quats[idx],
        scales=tensors.scales[idx],
        opacities=tensors.opacities[idx],
        colors=colors,
        sh_degree=tensors.sh_degree,
        intensity_raw=None
        if tensors.intensity_raw is None
        else tensors.intensity_raw[idx],
        raydrop_logit=None
        if tensors.raydrop_logit is None
        else tensors.raydrop_logit[idx],
        lidar_mask=None,
        raydrop_sh=None if tensors.raydrop_sh is None else tensors.raydrop_sh[idx],
    )
