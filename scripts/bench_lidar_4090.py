#!/usr/bin/env python3
"""Benchmark the tuned SplatAD 5-sensor LiDAR rig on the current GPU.

Reproduces the methodology recorded in docs/tuning_tips.md (which was measured
on an RTX 3090): scene_unified_vad.usdz, its own rig LiDAR calibration and rig
trajectory, rendered through render_lidars_concurrent. Warms the GPU, then times
each selected pose with CUDA events over interleaved rounds.

Everything is self-contained from the USDZ (rig.lidars for the mounts, rig.poses
for the base->world trajectory) -- no T4 dataset needed.

    PYTHONPATH=./src uv run python scripts/bench_lidar_4090.py

Env passthrough respected: SPLATSIM_LIDAR_LOD_SCALE (0.25), SPLATSIM_LIDAR_CONCURRENT (1).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

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
    # Gaussians are re-centered to background.tile_local_centroid on load; the USDZ
    # rig poses are in the original (uncentered) world frame, so subtract the same
    # centroid to place the ego in the Gaussian frame (else the ego sits ~40m below
    # the scene and vertical-mount sensors return nothing). Matches eval's re-center.
    cen = scene.background.tile_local_centroid.detach().cpu().numpy().astype(np.float64)
    rigs = load_rig_trajectories(usdz)
    for rig in rigs:
        poses = getattr(rig, "poses", None)
        if poses:
            out = []
            for p in poses:
                m = _rig_in_world(p).astype(np.float64)
                m[:3, 3] -= cen
                out.append(m.astype(np.float32))
            return out
    raise SystemExit("no rig trajectory poses in USDZ")


def timed(fn, iters):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms.append(start.elapsed_time(end))
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz", default=USDZ)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument(
        "--frames",
        type=str,
        default="",
        help="comma-separated pose indices; default = auto (heavy+light+spread)",
    )
    ap.add_argument(
        "--scan-step",
        type=int,
        default=0,
        help="post-LOD count scan stride (0 = ~24 grid points)",
    )
    ap.add_argument(
        "--sectors",
        type=int,
        default=1,
        help="time one revolution as S azimuth-sector renders sharing one "
        "gather (the gRPC sector-streaming path) instead of one full render",
    )
    ap.add_argument(
        "--rolling",
        action="store_true",
        help="exercise the rolling-shutter (velocity) kernel path by passing "
        "the next trajectory pose as the sweep end",
    )
    args = ap.parse_args()

    dev = torch.device("cuda")
    print(
        f"[gpu] {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}"
    )

    print("[load] scene config + gaussians ...")
    config = SceneConfig.from_source(args.usdz)
    scene = Scene.from_config(config, device=dev, progress=print_progress)
    assert scene.background is not None
    print(f"[load] background gaussians = {scene.background.num_gaussians:,}")

    rends = build_rig(config, dev)
    print(
        f"[rig] {len(rends)} sensors: "
        + ", ".join(f"{r.sensor_spec.name}(W={r.sensor_spec.n_columns})" for r in rends)
    )

    poses = rig_poses(args.usdz, scene)
    n = len(poses)
    print(f"[traj] {n} rig poses")

    # Post-LOD gaussian count per pose (heaviest = most work).
    step = args.scan_step if args.scan_step > 0 else max(1, n // 24)
    scan_idx = list(range(0, n, step))
    counts = {}
    for i in scan_idx:
        b2w = torch.from_numpy(poses[i]).to(dev)
        shared = gather_lidar_rig(rends, b2w, scene)
        counts[i] = int(shared.count) if shared is not None else 0
    heavy = max(counts, key=lambda i: counts[i])
    light = min(counts, key=lambda i: counts[i])
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print(
        f"[scan] {len(scan_idx)} poses (step={step}); top-5 heavy: "
        + ", ".join(f"{i}:{c:,}" for i, c in top)
    )
    print(
        f"[scan] heavy idx={heavy} ({counts[heavy]:,}), "
        f"light idx={light} ({counts[light]:,})"
    )

    if args.frames:
        sel = [int(x) for x in args.frames.split(",")]
    else:
        # Heaviest + lightest + 4 evenly spread poses (tuning_tips used 6).
        spread = [int(round(t * (n - 1))) for t in (0.15, 0.4, 0.6, 0.85)]
        sel = sorted(set([heavy, light] + spread))
    print(f"[sel] poses to time: {sel}")

    # Warm the GPU: idle SM clocks are ~210 MHz vs ~1965 warm (tuning_tips).
    print(f"[warm] {args.warmup} warmup renders on the heavy pose ...")
    b2w_h = torch.from_numpy(poses[heavy]).to(dev)
    for _ in range(args.warmup):
        render_lidars_concurrent(rends, b2w_h, scene)
    torch.cuda.synchronize()

    # Interleaved rounds: cycle poses so per-pose clock state is comparable.
    per_pose_ms = {i: [] for i in sel}
    b2ws = {i: torch.from_numpy(poses[i]).to(dev) for i in sel}
    ends = {
        i: torch.from_numpy(poses[min(i + 1, n - 1)]).to(dev) if args.rolling else None
        for i in sel
    }
    s_count = max(1, args.sectors)

    def one_revolution(i: int) -> None:
        if s_count == 1:
            render_lidars_concurrent(rends, b2ws[i], scene, base_to_world_end=ends[i])
            return
        # The gRPC sector loop's cost model: ONE rig-wide gather per
        # revolution, then S sector renders reusing it. (It interpolates a
        # pose pair per sector; pose arithmetic is noise next to the render,
        # so the same pair is reused here.)
        shared = gather_lidar_rig(rends, b2ws[i], scene, base_to_world_end=ends[i])
        for k in range(s_count):
            render_lidars_concurrent(
                rends,
                b2ws[i],
                scene,
                base_to_world_end=ends[i],
                shared=shared,
                sector=(k, s_count),
                sync=False,
            )

    if s_count > 1:
        print(
            f"[mode] sector streaming: {s_count} sectors/revolution"
            + (" + rolling shutter" if args.rolling else "")
        )
        for _ in range(2):  # prime per-sector geometry caches
            one_revolution(heavy)
        torch.cuda.synchronize()
    for _ in range(args.iters):
        for i in sel:
            ms = timed(lambda: one_revolution(i), 1)
            per_pose_ms[i].extend(ms)

    print("\n=== per-pose 5-sensor rig timing (ms/frame) ===")
    print(f"{'pose':>6} {'postLOD':>12} {'mean':>8} {'min':>8} {'p50':>8} {'fps':>6}")
    means = []
    for i in sel:
        arr = np.array(per_pose_ms[i])
        pc = counts.get(i)
        pc_s = f"{pc:,}" if pc is not None else "-"
        mean = arr.mean()
        means.append(mean)
        tag = "  <-heavy" if i == heavy else ("  <-light" if i == light else "")
        print(
            f"{i:>6} {pc_s:>12} {mean:>8.1f} {arr.min():>8.1f} "
            f"{np.median(arr):>8.1f} {1000 / mean:>6.1f}{tag}"
        )
    print(f"\n[mean over {len(sel)} poses] {np.mean(means):.1f} ms/frame")
    print(
        f"[heavy pose {heavy}] {np.mean(per_pose_ms[heavy]):.1f} ms/frame "
        f"({1000 / np.mean(per_pose_ms[heavy]):.1f} FPS, 5-LiDAR)"
    )
    print(f"[peak mem] {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
