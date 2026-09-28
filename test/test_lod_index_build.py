"""The ragged LOD index build must reproduce the dense-matrix construction.

``LodManager.filter`` used to materialise a ``[num_cells, max_count]`` index
matrix and mask it; it now lays the per-cell runs out contiguously. The two
must produce byte-identical index tensors, including for empty cells.
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _dense(starts: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """The original construction."""
    device = starts.device
    max_count = int(counts.max().item())
    offsets = (
        torch.arange(max_count, device=device).unsqueeze(0).expand(len(starts), -1)
    )
    abs_indices = starts.unsqueeze(1) + offsets
    mask = offsets < counts.unsqueeze(1)
    return abs_indices[mask]


def _ragged(starts: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """The construction now used by LodManager.filter."""
    device = starts.device
    total = int(counts.sum())
    if total == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    run_ends = counts.cumsum(0)
    run_bases = run_ends - counts
    per_slot_start = torch.repeat_interleave(starts, counts)
    per_slot_base = torch.repeat_interleave(run_bases, counts)
    return per_slot_start + (torch.arange(total, device=device) - per_slot_base)


def _case(device, seed, n_cells, zero):
    g = torch.Generator(device=device).manual_seed(seed)
    sizes = torch.randint(
        1, 400, (n_cells,), generator=g, device=device, dtype=torch.int64
    )
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.int64, device=device), sizes.cumsum(0)[:-1]]
    )
    counts = torch.randint(
        0, 200, (n_cells,), generator=g, device=device, dtype=torch.int64
    )
    counts = torch.minimum(counts, sizes)
    if zero == "all":
        counts = torch.zeros_like(counts)
    elif zero == "some":
        counts[::3] = 0
    return starts, counts


@cuda
@pytest.mark.parametrize("zero", [None, "some", "all"])
def test_ragged_matches_dense(zero) -> None:
    device = torch.device("cuda")
    for seed, n_cells in ((0, 2137), (1, 7), (2, 1)):
        starts, counts = _case(device, seed, n_cells, zero)
        ragged = _ragged(starts, counts)
        if int(counts.sum()) == 0:
            assert ragged.numel() == 0
            continue
        assert torch.equal(_dense(starts, counts), ragged)


@cuda
def test_filter_still_selects_the_top_of_each_cell() -> None:
    """Each cell must contribute its FIRST count entries (importance order)."""
    device = torch.device("cuda")
    starts = torch.tensor([0, 10, 25], device=device)
    counts = torch.tensor([3, 0, 4], device=device)
    expected = torch.tensor([0, 1, 2, 25, 26, 27, 28], device=device)
    assert torch.equal(_ragged(starts, counts), expected)
