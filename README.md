# splatsim

3D Gaussian Splatting based simulator with DDS and CARLA integration.

## Prerequisites

- Python 3.10+
- CUDA 12.8
- [uv](https://docs.astral.sh/uv/)
- (Docker builds) Docker with [BuildKit](https://docs.docker.com/build/buildkit/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Installation

### Local

```bash
# Core only (headless)
uv sync

# With GUI viewer
uv sync --extra gui

# With DDS support
uv sync --extra dds

# With CARLA support (includes DDS)
uv sync --extra carla

# All optional dependencies
uv sync --extra all
```

> CARLA extra requires system libraries for lanelet2. On Ubuntu 22.04:
>
> ```bash
> sudo apt-get install -y \
>   libboost-dev libboost-serialization-dev libboost-filesystem-dev \
>   libboost-program-options-dev libboost-python-dev libboost-system-dev \
>   libeigen3-dev libpugixml-dev libgeographic-dev librange-v3-dev python3-dev
> ```

### Docker

#### Pull a published image

Released images are published to the GitHub Container Registry. A **separate
image is built per GPU architecture** (one CUDA compute capability each), so
pick the tag that matches your GPU. Each tag is
`<version>-cuda<cuda>-sm<XX>`, plus a rolling `latest-sm<XX>`.

| GPU | Compute capability | Tag suffix |
| --- | --- | --- |
| RTX 20xx (Turing)     | 7.5  | `sm75`  |
| A100 (Ampere)         | 8.0  | `sm80`  |
| RTX 30xx (Ampere)     | 8.6  | `sm86`  |
| RTX 40xx (Ada)        | 8.9  | `sm89`  |
| H100 (Hopper)         | 9.0  | `sm90`  |
| RTX 50xx (Blackwell)  | 12.0 | `sm120` |

The tag suffix names the native SASS target (`sm_XX`). Every image also embeds
`compute_XX` PTX, so the driver can JIT-run it on GPU generations newer than
the one it was built for — PTX is uniform across images and is not a separate
tag dimension.

```bash
# A specific version for an RTX 40xx (Ada, sm_89), CUDA 12.8.1
docker pull ghcr.io/tier4/splatsim:1.0.0-cuda12.8.1-sm89

# Rolling latest for an RTX 50xx (Blackwell, sm_120)
docker pull ghcr.io/tier4/splatsim:latest-sm120

# Run (requires NVIDIA Container Toolkit)
docker run --rm -it --gpus all ghcr.io/tier4/splatsim:latest-sm89
```

#### Build locally

The Dockerfile uses a multi-stage build with the `dds` extra to keep the image minimal.

```bash
# Build
docker buildx build \
  -f docker/Dockerfile \
  -t splatsim .

# Run (requires NVIDIA Container Toolkit)
docker run --rm -it --gpus all splatsim
```

To customize CUDA or Ubuntu versions:

```bash
docker buildx build \
  -f docker/Dockerfile \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg UBUNTU_VERSION=22.04 \
  -t splatsim .
```

## Usage

```bash
# Launch the viewer
splatsim-viewer

# Render a configured LiDAR and publish sensor_msgs/PointCloud2 via DDS
splatsim-viewer scene.usdz --lidar top --dds

# Run a scenario
spawn-scenario
```

## Dynamic objects (actor assets)

A scene USDZ built with 3dgs_io >= v2.1.0 can carry a bank of **rigid
dynamic-object assets** (`splatsim.actor_assets/v1`): cars, trucks, trailers,
cones — anything whose shape does not change over time. Each is a Gaussian
cloud authored in a canonical object-local frame (`+x` forward, `+y` left,
`+z` up, origin at the box centre, metric scale), in the same SPZ container the
background chunks use, so an actor's Gaussians and its per-Gaussian LiDAR
attributes go through the readers splatsim already has.

**Poses come from outside.** splatsim hands you a `RigidBody` per instance and
the scenario drives it — a CARLA bridge, a scenario runner, the gRPC service.
The bundle's own `sequence_tracks.json` is metadata here, exposed for callers
that want to replay it, not a playback engine.

```python
from splatsim import Scene
from splatsim.actor_assets import pose_from_track_frame

scene = Scene.from_config("scene.usdz")

# What the bundle ships
library = scene.actor_library
print(library.asset_ids, library.info("sedan_0007").size)

# Spawn as many instances as the scenario needs — they share one upload
scene.spawn_actor("sedan_0007", name="car_01", position=(113.6, -58.5, 1.9))
scene.spawn_actor("sedan_0007", name="car_02", position=(120.0, -58.5, 1.9))

# ...then drive them
scene.set_pose("car_01", (114.1, -58.5, 1.9), (0.966, 0.0, 0.0, 0.259))
```

Two conventions worth getting right, both handled for you if you use the
helpers:

- **Frame.** `Background` re-centres the scene on its own Gaussian centroid for
  numerical stability, so an ENU world-frame pose (a track pose, a map
  waypoint) must have that centroid subtracted. `spawn_actor` /
  `ActorAssetLibrary.spawn` do this by default; pass `world_position=False`
  when your pose is already tile-local.
- **Quaternion order.** Everything inside a scene bundle is `xyzw`; every
  splatsim pose is `wxyz`. Route bundle poses through
  `splatsim.actor_assets.pose_from_track_frame`.

### Over gRPC

The rendering service places and drives the same objects, so a client that
already streams sensor poses does not need a second channel for the traffic
around them:

```python
stub.Initialize(pb2.InitializeRequest(scene_path="scene.usdz", ...))

# What the bundle ships
for asset in stub.ListActorAssets(pb2.ListActorAssetsRequest()).assets:
    print(asset.asset_id, asset.class_name, asset.size)

stub.SpawnActor(pb2.SpawnActorRequest(
    instance_id="car_01",
    asset_id="sedan_0007",          # or asset_path="/data/sedan.spz"
    pose=pb2.Pose(position=..., rotation=...),
))

def poses():
    for stamp, ego, actors in scenario:
        yield pb2.RigData(
            stamp=stamp,
            pose=ego,
            actors=[pb2.ActorPose(instance_id="car_01", pose=actors["car_01"])],
        )

stub.StreamRigData(poses())
```

`actors` rides on the pose messages of all three streams (`StreamCameraData`,
`StreamLidarData`, `StreamRigData`). The newest poses the stream has delivered
are applied to the scene right before the frame is rendered — so a frame's
objects are as fresh as its ego pose, and on a rig every sensor sees them at one
instant. Leaving `actors` out of a message keeps the previous poses, so a client
may move objects at a lower rate than it streams poses (the server then skips
the update entirely rather than re-uploading unchanged poses).

Two differences from the Python API, both because the wire is not the scenario:

- **Frame.** Camera / LiDAR / rig poses are streamed tile-local, so actors
  default to the same frame rather than to the world frame `spawn_actor` uses.
  Set `world_frame` on `SpawnActor` to send ENU world coordinates and have the
  server subtract the origin it reported in `InitializeResponse.scene_origin`;
  the choice is fixed per instance and applies to every `ActorPose` after it.
- **Zero quaternions.** proto3 has no unset scalar, so an all-zero `rotation`
  means "keep the current one" rather than collapsing the object — fill it in
  only when you mean to turn something.

`SpawnActor` also takes an `asset_path` to a standalone `.spz` / `.ply` /
`.glb`, for scenes whose bundle ships no bank; instances of one path share a
single upload, as bank assets do. `RemoveActor` takes an object back out — it
is gone from the next frame.

### View-dependent bands, camera and LiDAR

Actors re-express **both** their view-dependent band sets when posed, unlike the
pre-existing `RigidBodyConfig` bodies:

- **colour SH** — leaving the specular bands in the object frame makes
  highlights spin with the car;
- **`raydrop_sh`** — the LiDAR renderer evaluates drop probability along the
  world-space sensor→Gaussian ray, so unrotated bands sample a car facing east
  with the pattern it had facing north. The scalar `raydrop_logit` is the
  band-0 term and is rotation-invariant, so it needs nothing.

Both go through the same closed-form `+Z` rotation off a single yaw, which is
exact for the yaw-only poses road vehicles have.
`RigidBody.sh_rotation_tilt` / `.sh_rotation_is_exact` report when a pose is
tilted far enough out of the ground plane for it to be an approximation.

Everything else on the LiDAR path already treats an actor like any other
source: `lidar_mask` is applied per-source before the scene concat (so an asset
without one participates fully), assets that carry no raydrop bands are
zero-padded to the scene's SH width and contribute only their scalar logit, and
the LOD `lidar_view` gather thins the actor's object-frame Gaussians before the
pose is applied.

Actors can also be declared in a scene config, for static props and spawn
points:

```yaml
actors:
  - asset_id: sedan_0007
    name: parked_car
    position: [113.62, -58.55, 1.92]   # ENU world frame
    rotation: [0.966, 0.0, 0.0, 0.259] # wxyz
```

## LiDAR transport (DDS vs. HILS)

Each LiDAR sensor in a scene renders a point cloud that can be delivered two ways,
selected per sensor with the `communication` field:

- `dds` (default): publishes a `sensor_msgs/PointCloud2` over CycloneDDS on
  `pointcloud_topic`.
- `hils`: hardware-in-the-loop mode. Emits raw Hesai UDP data packets that mimic
  the physical sensor's wire format, so an unmodified LiDAR driver (e.g. Autoware
  `nebula`) can consume them. The `sensor_type` selects the packet format;
  currently supported models are **OT128** (Pandar OT128) and **XT32**
  (PandarXT-32).

```yaml
lidar_sensors:
  - name: top
    sensor_type: XT32        # OT128 | XT32
    communication: hils      # dds (default) | hils
    hils_host: 192.168.1.201 # UDP destination (unicast or broadcast)
    hils_port: 2368          # Hesai point-cloud port
    # hils_start_epoch: 1700000000  # optional: Unix time that sim-time 0
    #                               # maps to; omit to start "now".
```

Packet timestamps use the **simulation clock**: the date-time/timestamp written
into each packet is `hils_start_epoch + <simulation elapsed time>`. When
`hils_start_epoch` is omitted it defaults to the wall-clock time at which the
sensor is created (simulation starts "now").

Packets are encoded with `torch` directly from the rendered range image, so when
the renderer runs on CUDA the entire encode stays on the GPU and only the packed
byte buffer crosses to the host (once per frame) for the UDP send.

## LiDAR evaluation

`src/eval` scores the LiDAR the simulator renders from a reconstructed `.usdz`
scene against the ground-truth LiDAR of the matching WebAuto / T4 dataset, at the
same ego pose, and logs everything to a single [Rerun](https://rerun.io) `.rrd`
(both point clouds + every metric's time series on a shared timeline). It is a
small metric-plugin framework — each evaluation item is its own class under
`src/eval/metrics/` implementing `LidarEvalMetric`, so the per-frame render/mask
happens once and is shared across metrics:

- **`chamfer`** — symmetric Chamfer distance (raw + range-aware), in metres.
- **`bev`** — OnePlanner **BEV-encoder** feature similarity: both clouds are
  pushed through the BEV encoder and the resulting `[512, 180, 180]`
  bird's-eye-view feature maps are compared (per-cell / global cosine, relative
  L2). A shared-basis PCA(512→3) RGB view of each map and a cosine heatmap are
  logged as images so the learned representations can be compared by eye. This
  answers *"would a downstream planner perceive the reconstructed scene the same
  way it perceives the real one?"* rather than raw geometric distance.

```bash
uv sync --extra eval                       # chamfer metric (t4-devkit + rerun)
uv run python -m eval.eval_lidar \
    --scene scene.usdz --data-root ~/.webauto/datasets --dataset-id <id> \
    --metrics chamfer --output outputs/eval_lidar.rrd
```

### BEV-encoder metric

The BEV encoder is the LiDAR branch of OnePlanner's BEVFusion, shipped as
`oneplanner_bev_encoder.onnx`. Its ONNX embeds `autoware` sparse-convolution
custom ops (`GetIndicePairsImplicitGemm` / `ImplicitGemm`) that **plain
`onnxruntime` cannot execute** — their implementation lives only in the
`autoware_tensorrt_plugins` (which wrap [spconv](https://github.com/traveller59/spconv)).
Two backends run the ONNX:

- **`spconv` (default).** `onnx2torch` converts the whole graph to a
  `torch.nn.Module` and loads all weights from the ONNX; the two custom ops are
  supplied by `spconv` (the same library the plugins wrap) via lightweight
  converters. Pure-pip and CUDA-aligned with the project's torch — no TensorRT,
  no plugin `.so`, no version matching. This is the recommended path.
- **`tensorrt` (`--bev-backend tensorrt`).** Loads the `autoware_tensorrt_plugins`
  `.so`, builds (and disk-caches) a TensorRT engine, and runs inference using
  torch tensors as the CUDA buffers. The production path, but its `tensorrt`
  wheel + CUDA must match the prebuilt plugin's `libnvinfer` (here the plugin
  links `libnvinfer.so.10`, TensorRT 10.16.1), so install a matching `tensorrt`
  yourself and pass `--bev-plugins`.

```bash
uv sync --extra eval --extra bev           # onnx2torch + spconv (default backend)
export ONEPLANNER_BEV_ONNX=/path/to/oneplanner_bev_encoder.onnx
uv run python -m eval.eval_lidar \
    --scene scene.usdz --data-root ~/.webauto/datasets --dataset-id <id> \
    --metrics chamfer,bev --output outputs/eval_lidar.rrd
```

The encoder ONNX is an environment-specific artifact supplied via `--bev-onnx`
(or `$ONEPLANNER_BEV_ONNX`) rather than vendored. `spconv-cu126` pairs
`cumm-cu126` (CUDA 12.6 wheels, compatible with the CUDA 12.8 torch here).

## Development

```bash
uv sync --dev
pre-commit install
```
