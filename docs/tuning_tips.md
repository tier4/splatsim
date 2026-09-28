# LiDAR rendering performance: what worked, what didn't

A record of the measured attempts at speeding up the SplatAD LiDAR path, kept
so the rejected ideas are not re-tried and the accepted ones are not undone
without knowing what they bought.

**Benchmark used throughout.** The 5-sensor rig of `scene_unified_vad.usdz`
(27.2M Gaussians) on an RTX 3090 (sm_86), rendered through
`render_lidars_concurrent` — the same path the gRPC `StreamRigData` loop uses.
Six poses spanning the trajectory, including its heaviest (frame 1125, 10.09M
Gaussians after LOD) and its lightest (frame 200, 3.29M). Every result below is
a mean over those poses, with every panorama compared cell-by-cell against a
reference render.

> **Measure on heavy frames.** Frame 200 was the original benchmark and turned
> out to be nearly the *lightest* pose on the trajectory — 40% of the median.
> Several early conclusions had to be redone because of it.

> **Warm the GPU and interleave configurations.** Idle SM clocks sit at 210 MHz
> versus ~1965 MHz warm. An early tile sweep produced a bimodal mess until the
> harness warmed up first and ran configurations in interleaved rounds.

---

## Accepted

| Change | Effect | Exact? |
|---|---|---|
| Reuse the rig's CUDA streams | 325 → 158 ms | bit-identical |
| Retune raster tiles (4x64 → 1x16) | 158 → 141 ms | IoU ≥ 0.99998 |
| Wider shared staging (`LIDAR_BATCH_MULT` 1 → 16) | 141 → 135 ms | bit-identical |
| Exponential-free rejection | 135 → 128 ms | bit-identical |
| Fold elevation for one-beam-row tiles | 128 → 124 ms | bit-identical |
| Default the pre-rasterizer frustum cull off | 124 → 117 ms | strictly more returns |
| Scalar raydrop-SH kernel | 1.80 → 0.18 ms/sensor | fp32 rounding |
| Ragged LOD index build | 1.38 → 0.15 ms/sensor | bit-identical |
| Shared Gaussian gather for the rig | see below | superset per sensor |
| LOD thinning default 0.5 → 0.25 | 148 → 77 ms (heavy) | 98.4% of returns |
| **Bin per beam, not per bounding box** | **70 → 48 ms** | bit-identical |

### The two that mattered most

**Reusing the CUDA streams.** Allocating fresh `torch.cuda.Stream` objects each
frame churns the caching allocator's per-stream blocks, and the cost arrives as
a *bimodal stall* rather than steady overhead: 25 consecutive identical renders
alternated 499 / 152 / 494 / 151 ms. The fast state was always there; the rig
only hit it half the time. Single-stream rendering is a steady 383 ms on that
frame, so the concurrent path only beats sequential once the churn stops.

**Binning per beam.** The tile binner took each Gaussian's widest azimuth extent
and applied it to every elevation row its bounding box spanned. At the shipped
1×W tiling a tile row *is* one beam, so the only elevation sampled in that row
is that beam's — and the reachable azimuth interval there is much narrower,
shrinking to nothing near the ellipse's poles. Solving the rasterizer's own
contribution test per row gives it exactly:

    0.5*cx*dx^2 + cy*dy*dx + 0.5*cz*dy^2 <= ln(255*opac)

The pair count drops only 11.3% but the frame drops 31%, because what is removed
are the *most expensive* pairs: tiles at the ellipse's elevation extremes, where
all 16 pixels test the Gaussian and none can be hit.

The intermediate version — taking `dy` to the nearest elevation in the row's
*band* rather than to the beam — is also exact but removes just 1.6% of pairs
and lands 1% slower than the bbox after its own arithmetic. Knowing that a row
is a single beam is what makes this pay.

---

## Rejected, with the measurement

### Radix-sort depth key quantization — 0–3%
Shortening the sort key (32-bit float depth → N-bit fixed point) does cut the
sort as intended: 8 → 6 kernel launches, 2.98 → 2.02 ms on lidar_top. But the
sort is only 2.5–7.7% of GPU time, and the rig's concurrency hides what remains
behind other sensors' rasterization. At 16 bits, 1.9% of cells change.

*Found while measuring:* `isect_offset_encode` decoded the tile id with a
hardcoded `>> 32`, so any change to the key layout silently corrupted every tile
offset (renders came out with IoU ~0 and erratic kernel times). Fixed on
`perf/lidar-sort-key-quantization`.

### Raising the rasterizer's transmittance cutoff — no gain, and it breaks the range
Ending a ray earlier gave nothing at any value from 1e-4 to 0.1, because most of
the inner loop is spent *testing* Gaussians that never contribute — work the
cutoff does not touch. Worse, it is not the free win it looks like: the loop
breaks **before** the median latch, so a ray whose transmittance crosses 0.5 and
the cutoff on the same Gaussian never records its range.

### `radius_clip` — no effect at all
Up to 0.3 the output is unchanged and so is the time: the clip requires *both*
extents below the threshold, and almost nothing in this scene is that small.

### Opacity-aware bounding boxes — more faithful, but slower
The rasterizer's real cutoff is `sqrt(2 ln(255*opac))` sigma, not the fixed 3
sigma used for binning — which is why a handful of boundary cells always differ
between tilings. But this scene's opacities are high (mean 0.94, median 0.99),
so 96.5% of Gaussians want 3.33 sigma and the tile lists would grow ~19%.

### Cutting `max_range` — 1.14x for 1.7% of returns
The return distribution does not saturate at 120 m (p99.9 = 114.5 m over 15
poses / 8.8M returns), and cost tracks near-field density rather than far
returns: rays stop at a median of 10.5 m. Even 50 m only reaches 1.27x. If you
need it anyway, 80 m is the knee.

### Column bitmask computed in the rasterizer — 13% slower
Computing, per staged Gaussian, the bitmask of tile columns it can reach, so
other threads skip it after a 4-byte read. Output identical, but the loader's
`sqrt` and the extra shared array cost more than the skipped inner-loop work.

### Column range packed into the flatten index — 7% slower
The same idea with the range computed once in the binner (where it is nearly
free) and carried in the top 8 bits of `flatten_ids`. Still slower: the range
lives in the second staged record, so a rejection reads it anyway and saves
nothing.

### Splitting the staged record hot/cold — no change
Folding `sigma_max` into the quadratic's constant term so a rejection needs only
`(xy.x, qa, qb, qc - sigma_max)`, moving opacity and the id to a record only
contributors read. 48.1 → 48.6 ms. **This is the useful null result:** halving
the shared read for ~90% of tests changed nothing, so the kernel is not
shared-bandwidth bound.

### Pixels per thread (register blocking) — 1.4x slower
Each thread owning several panorama columns so one staged record serves several
pixels: 48 → 68 ms at 2 columns, 82 ms at 4. Predicted by the null result above
— amortizing a read that was not the bottleneck cannot help, and the narrower
block costs occupancy.

---

## Where the time goes now

Frame 1125, LOD 0.25, 5 sensors, 126.8 ms of GPU work:

    RASTERIZE      94.7 ms  74.7%
    radix sort     12.4 ms   9.8%
    isect_tiles     4.5 ms   3.5%
    projection      2.7 ms   2.1%
    everything else ~12 ms   9.9%

The rasterizer's problem is arithmetic intensity, not bandwidth or occupancy:
each panorama cell tests ~2,200 Gaussians to produce one range sample. Tile
geometry is at a local optimum — 1x16 beat every alternative tried (1x4, 1x6,
1x8, 1x12, 1x32, 1x64, 1x128, 2x8, 2x16, 2x32, 2x64, 4x4, 4x8, 4x16, 4x32,
4x64, 8x4, 8x16, 8x32), and the two closest (2x8, 4x8) are within noise while
losing the one-beam-row fast path.

Getting substantially further needs a different spatial structure — per-beam
azimuth-sorted traversal, or a hierarchy that lets a pixel skip runs of the tile
list — not another parameter.

## Knobs

| Variable | Default | Notes |
|---|---|---|
| `SPLATSIM_LIDAR_LOD_SCALE` | 0.25 | 0.1 gives 4.45x for 95.8% of returns; return *rate* is what degrades (82.6% → 73.5% of cells), not range accuracy |
| `SPLATSIM_LIDAR_CONCURRENT` | 1 | 0 forces a single stream (the shared gather still applies) |
| `SPLATSIM_LIDAR_BATCH_MULT` | 16 | compile-time; Gaussians staged per thread per round. Swept 1/4/8/16/24/32; 32 is worse (shared-memory pressure) |

`SPLATSIM_LIDAR_PIX_PER_THREAD` appears in the rejected list but is **not** in
the tree — the pixels-per-thread work was reverted along with the measurement.

## Where the kernel lives

The rasterizer is no longer in this tree. It is
[`splatad_kernel`](https://github.com/hakuturu583/splatad_kernel), pinned by rev
in `[tool.uv.sources]`, and it holds only what gsplat has no equivalent for:
`rasterize_to_points` (the median first-return compositor), `isect_lidar_tiles`
(the per-beam binner) and `fully_fused_lidar_projection`. Everything the camera
path shares — `quat_to_rotmat`, `quat_scale_to_covar_preci`, `posW2C`,
`covarW2C`, `add_blur`, the glm headers, the `CHECK_*`/`CUB_WRAPPER` macros — is
`#include`d out of the installed gsplat (1.5.3), not copied.

Two things follow for anyone tuning it. The kernel and gsplat move together: the
JIT rebuilds when either changes, so a gsplat bump is a kernel rebuild. And
iterating means editing the kernel repo and rsyncing
`src/splatad_kernel/` over `.venv/lib/python3.10/site-packages/splatad_kernel/`;
`torch.utils.cpp_extension.load` notices the changed sources and rebuilds on the
next import. Re-pin the rev when the change is settled.
