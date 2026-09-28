#!/usr/bin/env python3
"""Re-sweep the SplatAD raster tile geometry on the current GPU.

docs/tuning_tips.md found 1x16 (tile_height=1 beam, tile_width=16 azimuth cols)
optimal on the RTX 3090, beating 1x4/1x8/1x12/1x32/2x8/2x16/4x8/4x64 etc. This
re-runs that sweep on this GPU (Ada / 4090) on the heaviest pose, to check
whether the 3090 optimum still holds.

Correctness across tilings is not re-checked here (tuning_tips: IoU >= 0.99998);
this is a pure timing sweep. Configs are interleaved and the GPU is warmed, per
the same protocol.

    PYTHONPATH=./src uv run python scripts/sweep_tiles_4090.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import splatsim.lidar_renderer as lr
from splatsim.dataclass.scene_config import SceneConfig
from splatsim.lidar_renderer import (
    build_lidar_sensors_from_config,
    LidarRenderer,
    gather_lidar_rig,
    render_lidars_concurrent,
)
from splatsim.scene import Scene, print_progress
from splatsim._usdz import load_rig_trajectories, _rig_in_world

USDZ = "/home/masaya/workspace/scene_unified_vad.usdz"

# (tile_height beams/tile, tile_width azimuth cols/tile). tile_height must divide
# the beam count (128). 1x16 is the shipped 3090 optimum.
CONFIGS = [
    (1, 4),
    (1, 8),
    (1, 12),
    (1, 16),
    (1, 24),
    (1, 32),
    (1, 64),
    (2, 8),
    (2, 16),
    (2, 32),
    (4, 8),
    (4, 16),
    (4, 32),
    (4, 64),
]


def build_rig(config, device):
    specs = build_lidar_sensors_from_config(config.lidar_sensors)
    rends = []
    for cfg_sensor, spec in zip(config.lidar_sensors, specs):
        rends.append(
            LidarRenderer(
                spec,
                device=device,
                min_range_m=float(cfg_sensor.min_range_m),
                max_range_m=float(cfg_sensor.max_range_m),
            )
        )
    return rends


def rig_poses(usdz, scene):
    # Subtract background.tile_local_centroid so the ego lands in the re-centered
    # Gaussian frame (see eval context.py:95); otherwise sensors render empty.
    cen = scene.background.tile_local_centroid.detach().cpu().numpy().astype(np.float64)
    for rig in load_rig_trajectories(usdz):
        poses = getattr(rig, "poses", None)
        if poses:
            out = []
            for p in poses:
                m = _rig_in_world(p).astype(np.float64)
                m[:3, 3] -= cen
                out.append(m.astype(np.float32))
            return out
    raise SystemExit("no rig trajectory poses")


def time_config(scene_config, scene, poses, heavy_idx, device, iters, warmup):
    rends = build_rig(scene_config, device)  # rebuilt so cached tile geometry is fresh
    b2w = torch.from_numpy(poses[heavy_idx]).to(device)
    for _ in range(warmup):
        render_lidars_concurrent(rends, b2w, scene)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(iters):
        start.record()
        render_lidars_concurrent(rends, b2w, scene)
        end.record()
        torch.cuda.synchronize()
        ms.append(start.elapsed_time(end))
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz", default=USDZ)
    ap.add_argument("--heavy", type=int, default=1200)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument(
        "--rounds", type=int, default=3, help="interleaved passes over the config list"
    )
    args = ap.parse_args()

    dev = torch.device("cuda")
    print(
        f"[gpu] {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}"
    )
    config = SceneConfig.from_source(args.usdz)
    scene = Scene.from_config(config, device=dev, progress=print_progress)
    poses = rig_poses(args.usdz, scene)
    b2w = torch.from_numpy(poses[args.heavy]).to(dev)
    shared = gather_lidar_rig(build_rig(config, dev), b2w, scene)
    n_union = int(shared.count) if shared is not None else 0
    print(
        f"[heavy] pose {args.heavy}: post-LOD union = "
        f"{n_union:,} gaussians, {len(config.lidar_sensors)} sensors"
    )

    # Warm the GPU generally before any timing.
    for _ in range(args.warmup):
        render_lidars_concurrent(build_rig(config, dev), b2w, scene)
    torch.cuda.synchronize()

    # Interleaved rounds: each round times every config once, so slow drift in
    # clock state spreads across configs instead of penalizing a few.
    acc: dict[tuple[int, int], list[float] | None] = {c: [] for c in CONFIGS}
    for r in range(args.rounds):
        for c in CONFIGS:
            samples = acc[c]
            if samples is None:  # marked infeasible on an earlier round
                continue
            lr._SPLATAD_TILE_HEIGHT, lr._SPLATAD_TILE_WIDTH = c
            try:
                ms = time_config(
                    config, scene, poses, args.heavy, dev, iters=args.iters, warmup=2
                )
            except (RuntimeError, ValueError) as e:
                shared = c[0] * c[1] * 16 * 32  # tw*th*BATCH_MULT*2*float4
                print(
                    f"[skip] {c[0]}x{c[1]} infeasible: shared={shared / 1024:.0f}KB "
                    f"> device cap ({str(e).splitlines()[0][:60]})"
                )
                acc[c] = None
                continue
            samples.extend(ms)
        done = (r + 1) * len(CONFIGS)
        print(f"[round {r + 1}/{args.rounds}] {done} configs timed")

    print("\n=== tile-geometry sweep on the heavy pose (ms/frame) ===")
    print(f"{'th x tw':>10} {'mean':>8} {'min':>8} {'p50':>8} {'vs 1x16':>9}")
    ref = float(np.mean(acc[(1, 16)] or [float("nan")]))
    rows = []
    for c in CONFIGS:
        samples = acc[c]
        if samples is None:
            print(
                f"{c[0]:>4} x{c[1]:>3}   infeasible (shared "
                f"{c[0] * c[1] * 16 * 32 / 1024:.0f}KB > cap)"
            )
            continue
        arr = np.array(samples)
        rows.append((c, arr.mean(), arr.min(), np.median(arr)))
    for c, mean, mn, p50 in sorted(rows, key=lambda x: x[1]):
        tag = "  <- 3090 optimum" if c == (1, 16) else ""
        best = "  *BEST*" if c == min(rows, key=lambda x: x[1])[0] else ""
        print(
            f"{c[0]:>4} x{c[1]:>3} {mean:>8.1f} {mn:>8.1f} {p50:>8.1f} "
            f"{mean / ref:>8.3f}x{best}{tag}"
        )


if __name__ == "__main__":
    main()
