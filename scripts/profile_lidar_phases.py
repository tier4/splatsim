#!/usr/bin/env python3
"""Break one LiDAR rig frame into its pipeline phases with CUDA events.

Answers "where does the frame time go?" by monkeypatching the SplatAD kernel's
three stages (projection, tile binning, rasterization) plus splatsim's own
gather / panorama layers with CUDA-event timers, then rendering selected poses
in SEQUENTIAL mode (SPLATSIM_LIDAR_CONCURRENT=0) so attribution is unambiguous.
The concurrent number from scripts/bench_lidar_4090.py is the end-to-end truth;
this script explains its composition.

    PYTHONPATH=./src uv run python scripts/profile_lidar_phases.py --poses 1160,1122
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

os.environ.setdefault("SPLATSIM_LIDAR_CONCURRENT", "0")

import splatad_kernel.rendering as sk_rendering
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

# phase -> list[(start_event, end_event)], flushed after a synchronize.
_EVENTS: dict[str, list] = defaultdict(list)


def _timed(phase: str, fn):
    def wrapper(*args, **kwargs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(*args, **kwargs)
        e.record()
        _EVENTS[phase].append((s, e))
        return out

    return wrapper


def install_probes():
    # SplatAD kernel stages (names bound at import time inside rendering.py).
    sk_rendering.fully_fused_lidar_projection = _timed(
        "2a projection", sk_rendering.fully_fused_lidar_projection
    )
    sk_rendering.isect_lidar_tiles = _timed(
        "2b isect_tiles", sk_rendering.isect_lidar_tiles
    )
    sk_rendering.isect_offset_encode = _timed(
        "2c isect_offset", sk_rendering.isect_offset_encode
    )
    sk_rendering.rasterize_to_points = _timed(
        "2d rasterize", sk_rendering.rasterize_to_points
    )
    sk_rendering.lidar_rasterization = _timed(
        "2k lidar_rast", sk_rendering.lidar_rasterization
    )
    # splatsim layers.
    lr.gather_lidar_rig = _timed("1 rig gather", lr.gather_lidar_rig)
    lr.render_lidar_panorama = _timed("2 panorama total", lr.render_lidar_panorama)
    lr._eval_view_dependent_raydrop = _timed(
        "2p raydrop_sh", lr._eval_view_dependent_raydrop
    )
    # lidar_renderer imported lidar_rasterization lazily via _splatad_lidar_rasterization,
    # which reads splatad_kernel.rendering.lidar_rasterization -> our patched stage fns
    # are already what it calls. Reset its cache in case it resolved earlier.
    lr._SPLATAD_RAST = None


def flush(reset=True):
    torch.cuda.synchronize()
    out = {}
    for phase, evs in _EVENTS.items():
        out[phase] = [s.elapsed_time(e) for s, e in evs]
    if reset:
        _EVENTS.clear()
    return out


def build_rig(config, device):
    specs = build_lidar_sensors_from_config(config.lidar_sensors)
    return [
        LidarRenderer(
            spec,
            device=device,
            min_range_m=float(c.min_range_m),
            max_range_m=float(c.max_range_m),
        )
        for c, spec in zip(config.lidar_sensors, specs)
    ]


def rig_poses(usdz, scene):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz", default=USDZ)
    ap.add_argument("--poses", type=str, default="1160,1122,0")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=6)
    args = ap.parse_args()

    dev = torch.device("cuda")
    print(f"[gpu] {torch.cuda.get_device_name(0)}")
    config = SceneConfig.from_source(args.usdz)
    scene = Scene.from_config(config, device=dev, progress=print_progress)
    rends = build_rig(config, dev)
    poses = rig_poses(args.usdz, scene)
    sel = [int(x) for x in args.poses.split(",")]

    install_probes()

    for idx in sel:
        b2w = torch.from_numpy(poses[idx]).to(dev)
        shared = gather_lidar_rig(rends, b2w, scene)
        n = int(shared.count) if shared is not None else 0
        for _ in range(args.warmup):
            render_lidars_concurrent(rends, b2w, scene)
        flush()  # discard warmup events

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        totals = []
        for _ in range(args.iters):
            s.record()
            render_lidars_concurrent(rends, b2w, scene)
            e.record()
            torch.cuda.synchronize()
            totals.append(s.elapsed_time(e))
        phases = flush()

        print(f"\n=== pose {idx}  postLOD={n:,}  sequential frame ===")
        print(f"{'phase':<18} {'calls/f':>8} {'ms/frame':>9} {'% frame':>8}")
        frame = float(np.mean(totals))
        accounted = 0.0
        for phase in sorted(phases):
            arr = np.array(phases[phase]).reshape(args.iters, -1)
            per_frame = arr.sum(axis=1).mean()
            calls = arr.shape[1]
            if phase.startswith(("1", "2 ")):
                accounted += per_frame if phase != "2 panorama total" else 0.0
            print(
                f"{phase:<18} {calls:>8} {per_frame:>9.2f} {per_frame / frame * 100:>7.1f}%"
            )

        def _pf(phase):
            if phase not in phases:
                return 0.0
            return float(np.array(phases[phase]).reshape(args.iters, -1).sum(1).mean())

        pano = _pf("2 panorama total")
        krast = _pf("2k lidar_rast")
        raydrop = _pf("2p raydrop_sh")
        inner = sum(
            _pf(p)
            for p in (
                "2a projection",
                "2b isect_tiles",
                "2c isect_offset",
                "2d rasterize",
            )
        )
        gather = _pf("1 rig gather")
        splatsim_glue = pano - krast - raydrop
        kernel_glue = krast - inner
        print(
            f"{'pano-side glue':<18} {'':>8} {splatsim_glue:>9.2f} {splatsim_glue / frame * 100:>7.1f}%  (stack/norm/flip/points)"
        )
        print(
            f"{'kernel-side glue':<18} {'':>8} {kernel_glue:>9.2f} {kernel_glue / frame * 100:>7.1f}%  (cat feats+depths etc)"
        )
        print(
            f"{'frame - gather - pano':<18} {'':>8} {frame - gather - pano:>9.2f} {(frame - gather - pano) / frame * 100:>7.1f}%  (other)"
        )
        print(f"{'FRAME TOTAL':<18} {'':>8} {frame:>9.2f}   100.0%")


if __name__ == "__main__":
    main()
