ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=22.04

# =============================================================================
# Build stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python
ENV UV_LINK_MODE=copy

WORKDIR /workspace/splatsim

# gsplat is pinned to a git rev and compiles CUDA kernels during `uv sync`;
# TORCH_CUDA_ARCH_LIST must be set before the first sync or the build fails
# with an IndexError from torch's _get_cuda_arch_flags.
# Architectures cover consumer RTX 20xx (7.5, Turing) through RTX 50xx
# (12.0, Blackwell), plus datacenter Ampere (8.0) and Hopper (9.0). sm_120
# requires CUDA 12.8+ / torch 2.7+ (see pyproject.toml). The top arch carries
# +PTX so the driver can JIT this (default, fat) image onto GPU generations
# newer than Blackwell. The CI/release matrix overrides this with a single
# arch (+PTX) per image.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TORCH_EXTENSIONS_DIR=/workspace/splatsim/.torch_extensions

# Install dependencies first (layer cache optimization)
COPY pyproject.toml uv.lock .python-version README.md ./
COPY resource/ resource/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dds --no-dev --no-install-project

# Copy source, generate proto code, and install project
COPY src/ src/
COPY proto/ proto/
RUN uv run python src/splatsim/grpc_service/generate_proto.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dds --no-dev

# Pre-compile gsplat CUDA kernels (JIT) so the runtime image does not need nvcc.
# `mkdir -p` guarantees the extensions dir exists even when gsplat's wheel
# already includes a pre-compiled `.so` (in which case `_backend.py` imports it
# directly and never triggers JIT, leaving `TORCH_EXTENSIONS_DIR` uncreated
# and breaking the later COPY to the runtime stage).
RUN mkdir -p "${TORCH_EXTENSIONS_DIR}" \
 && uv run python -c "from gsplat.cuda._backend import _C; assert _C is not None, 'gsplat CUDA JIT compilation failed'"

# Pre-compile splatsim's fused LiDAR cull kernel too (same reasoning as above:
# the runtime image is a CUDA runtime container without nvcc, so we build the
# `.so` here and let ``_lidar_cull_ext._load_prebuilt`` pick it up at runtime).
RUN uv run python -c "from splatsim._lidar_cull_ext import _try_load; assert _try_load() is not None, 'splatsim CUDA cull JIT compilation failed'"

# Pre-compile the SplatAD spherical LiDAR rasterizer (splatad_kernel) as well --
# it is the sole LiDAR rasterization backend, with no gsplat fallback, and it
# JIT-builds on first import. Same reasoning as above: build the
# `splatad_kernel_cuda` .so here so the nvcc-less runtime image loads it from
# TORCH_EXTENSIONS_DIR.
RUN uv run python -c "from splatad_kernel.cuda._backend import _C; assert _C is not None, 'SplatAD LiDAR CUDA JIT compilation failed'"

# Patch gsplat _backend.py so it can load the pre-compiled .so without nvcc.
# The upstream code gates on cuda_toolkit_available() which requires nvcc even
# when a compiled extension already exists.  This patch adds a fallback that
# loads the pre-compiled .so directly when the CUDA toolkit is absent.
COPY <<'PATCH_EOF' /tmp/patch_gsplat.py
import pathlib

p = pathlib.Path(".venv/lib/python3.10/site-packages/gsplat/cuda/_backend.py")
src = p.read_text()

old = """\
    else:
        Console().print(
            "[yellow]gsplat: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )"""

new = """\
    else:
        # Try loading pre-compiled CUDA extension without toolkit
        name = "gsplat_cuda"
        build_dir = _get_build_directory(name, verbose=False)
        _so = os.path.join(build_dir, f"{name}.so")
        if os.path.exists(_so):
            _C = _import_module_from_library(name, build_dir, True)
        else:
            Console().print(
                "[yellow]gsplat: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
            )"""

assert old in src, "Patch target not found in _backend.py; gsplat version may have changed"
p.write_text(src.replace(old, new))
print("_backend.py patched successfully")
PATCH_EOF
RUN uv run python /tmp/patch_gsplat.py

# Remove packages unnecessary for inference/simulation
RUN rm -rf .venv/lib/python3.10/site-packages/triton \
    .venv/lib/python3.10/site-packages/nvidia/nccl \
    .venv/lib/python3.10/site-packages/nvidia/cusparse \
    .venv/lib/python3.10/site-packages/nvidia/cufft \
    .venv/lib/python3.10/site-packages/nvidia/cusolver

# =============================================================================
# Runtime stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Enable NVIDIA Container Toolkit GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/splatsim

COPY --from=builder /opt/python /opt/python
COPY --from=builder /workspace/splatsim/.venv .venv/
COPY --from=builder /workspace/splatsim/.torch_extensions .torch_extensions/
COPY --from=builder /workspace/splatsim/src/ src/
COPY --from=builder /workspace/splatsim/pyproject.toml .
COPY launch/ launch/
COPY package.xml ./

ENV VIRTUAL_ENV=/workspace/splatsim/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV TORCH_EXTENSIONS_DIR=/workspace/splatsim/.torch_extensions

EXPOSE 50051
CMD ["splatsim-grpc-server"]
