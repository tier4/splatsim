"""JIT-compiled CUDA extension for the fused LiDAR cull kernel.

The extension exposes a single function ``lidar_cull_mask`` that produces
the boolean keep mask combining the range shell test and (optionally) the
elevation-FOV band test in a single CUDA kernel launch. It replaces the
~10-op PyTorch chain in :func:`splatsim.lidar_renderer.render_lidar_panorama`.

Compilation is performed lazily on first use via ``torch.utils.cpp_extension.load``.
The Docker builder stage runs this once so the runtime image (no nvcc) just
imports the ``.so`` from ``TORCH_EXTENSIONS_DIR`` — see :func:`_load_prebuilt`.
Failures fall back to the pure-PyTorch path in the renderer.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import torch

# `Any` rather than a concrete Protocol: the loaded object is a Python
# module produced by pybind11 whose function signatures aren't visible to
# static analysers, and we call exactly one attribute (``lidar_cull_mask``).
_EXT: Any = None
_EXT_LOCK = threading.Lock()
_EXT_LOAD_FAILED = False
_EXT_LOAD_ERROR: Optional[BaseException] = None

_SRC_PATH = Path(__file__).resolve().parent / "csrc" / "lidar_cull_kernel.cu"

# Every binding the current kernel source exports. A pre-built ``.so`` missing
# any of these is STALE (built from an older lidar_cull_kernel.cu) and must not
# be used: callers probe with hasattr and silently fall back to a many-times
# slower PyTorch path, so a stale binary costs milliseconds per frame while
# looking perfectly healthy (observed: an .so predating raydrop_sh_eval kept
# the 5-sensor rig on the gsplat fallback at ~15 ms/frame vs ~2 ms).
_EXPECTED_SYMBOLS = ("lidar_cull_mask", "raydrop_sh_eval", "cull_ext_abi_version")

# lidar_cull_mask's signature changed when the azimuth-wedge test was added;
# an .so exporting an older ABI would raise TypeError on every call and send
# the renderer to the slow fallback. Reject it up front like a missing symbol.
_EXPECTED_ABI_VERSION = 3


def _try_load() -> Any:
    """Compile and load the CUDA extension, or return ``None`` on failure."""
    global _EXT, _EXT_LOAD_FAILED, _EXT_LOAD_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_LOAD_FAILED:
        return None
    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT
        if _EXT_LOAD_FAILED:
            return None
        try:
            # Prefer a pre-built `.so` (runtime image path — no nvcc and no
            # live GPU needed to *import* it). Fall back to JIT only when the
            # CUDA toolkit is present. We deliberately do NOT gate on
            # ``torch.cuda.is_available()``: the Docker builder stage has nvcc
            # but no GPU runtime, yet must still pre-compile the extension so
            # the runtime image can pick it up.
            _EXT = _load_prebuilt()
            if _EXT is None and _can_compile():
                _EXT = _load_via_jit()
            if _EXT is None:
                _EXT_LOAD_FAILED = True
                _EXT_LOAD_ERROR = RuntimeError(
                    "no pre-built .so found and CUDA toolkit unavailable for JIT"
                )
            return _EXT
        except Exception as exc:  # noqa: BLE001 — we want any failure to fall back
            _EXT_LOAD_FAILED = True
            _EXT_LOAD_ERROR = exc
            return None


_EXT_NAME = "splatsim_lidar_cull_ext"


def _can_compile() -> bool:
    """Return ``True`` iff this machine can JIT-compile the CUDA extension.

    Compilation needs the CUDA toolkit (``nvcc``), *not* a live GPU: the
    Docker builder stage has nvcc but no GPU runtime and must still build
    the ``.so``. ``CUDA_HOME`` is torch's own resolved toolkit root and is
    ``None`` when no toolkit is found — the same signal gsplat keys off.
    """
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        return CUDA_HOME is not None
    except Exception:
        return False


def _build_dir() -> Path:
    """Where the compiled ``.so`` lives — same convention torch itself uses.

    ``torch.utils.cpp_extension.load(name=X)`` writes the artifact to
    ``$TORCH_EXTENSIONS_DIR/X/X.so`` (or a per-python-and-cuda-version
    subdir of ``~/.cache`` if the env var is unset). We compute the same
    path so the "load pre-built .so" path picks up what the Docker
    builder stage produced.
    """
    from torch.utils.cpp_extension import _get_build_directory  # type: ignore[attr-defined]

    return Path(_get_build_directory(_EXT_NAME, verbose=False))


def _load_prebuilt() -> Any:
    """Try to load a previously-compiled ``.so`` without invoking nvcc.

    This is the path the runtime Docker image uses: the builder image
    JITs the extension into ``$TORCH_EXTENSIONS_DIR`` and the runtime
    image (which has no CUDA toolkit) imports the ``.so`` directly.
    """
    try:
        build_dir = _build_dir()
        so_path = build_dir / f"{_EXT_NAME}.so"
        if not so_path.exists():
            return None
        # Staleness must be decided BEFORE importing: CPython caches C
        # extensions by (name, path), so once a stale .so is imported, even a
        # JIT rebuild in this same process gets the old module handed back.
        # An .so older than the kernel source predates its latest edit; let
        # the JIT path rebuild instead. (In the Docker flow the builder JITs
        # after the sources are laid down, so the .so is always newer there
        # and this fast path keeps working without nvcc.)
        if so_path.stat().st_mtime < _SRC_PATH.stat().st_mtime:
            return None
        from torch.utils.cpp_extension import _import_module_from_library  # type: ignore[attr-defined]

        mod = _import_module_from_library(
            _EXT_NAME, str(build_dir), is_python_module=True
        )
        # Secondary guard for binaries whose mtime lies (e.g. copied around):
        # a build missing any current binding is stale. Rejecting here can't
        # rescue THIS process (the import above already pinned the module in
        # CPython's extension cache) but it does let callers' fallbacks run
        # correctly and the next process pick up the JIT-rebuilt .so.
        if any(not hasattr(mod, sym) for sym in _EXPECTED_SYMBOLS):
            return None
        if int(mod.cull_ext_abi_version()) != _EXPECTED_ABI_VERSION:
            return None
        return mod
    except Exception:
        return None


def _load_via_jit() -> Any:
    from torch.utils.cpp_extension import load

    # `load(name=X)` uses ``$TORCH_EXTENSIONS_DIR/X/`` automatically.
    return load(
        name=_EXT_NAME,
        sources=[str(_SRC_PATH)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def is_available() -> bool:
    """Return ``True`` iff the extension is loaded and usable."""
    return _try_load() is not None


def last_error() -> Optional[BaseException]:
    """Return the exception raised on the last failed load attempt, if any."""
    return _EXT_LOAD_ERROR


def lidar_cull_mask(
    means: torch.Tensor,
    scales: torch.Tensor,
    sensor_pos: torch.Tensor,
    up_world: torch.Tensor,
    *,
    min_range: float,
    max_range: Optional[float],
    cull_scale_sigmas: float,
    use_elev: bool,
    sin_min: float,
    cos_min: float,
    sin_max: float,
    cos_max: float,
    fwd_world: Optional[torch.Tensor] = None,
    left_world: Optional[torch.Tensor] = None,
    use_azim: bool = False,
    az_center: float = 0.0,
    az_halfwidth: float = 0.0,
    az_pad: float = 0.0,
) -> torch.Tensor:
    """Fused frustum + elevation-FOV (+ azimuth-wedge) cull → (N,) bool tensor.

    ``max_range=None`` disables the upper bound. The azimuth wedge (sector
    rendering) keeps Gaussians whose linearized angular extent can overlap
    ``[az_center - az_halfwidth, az_center + az_halfwidth]`` (radians, sensor
    frame, wrap-aware) with ``az_pad`` extra slack; it needs the sensor's
    +x/+y axes in world coordinates (``fwd_world`` / ``left_world``).
    """
    ext = _try_load()
    if ext is None:
        raise RuntimeError(
            "splatsim CUDA cull extension is not available; "
            f"last load error: {_EXT_LOAD_ERROR!r}"
        )
    if use_azim and (fwd_world is None or left_world is None):
        raise ValueError("use_azim=True requires fwd_world and left_world")
    axis_placeholder = sensor_pos  # any (3,) tensor; kernel ignores it
    return ext.lidar_cull_mask(
        means,
        scales,
        sensor_pos,
        up_world,
        fwd_world if fwd_world is not None else axis_placeholder,
        left_world if left_world is not None else axis_placeholder,
        float(min_range),
        float(max_range) if max_range is not None else -1.0,
        float(cull_scale_sigmas),
        bool(use_elev),
        float(sin_min),
        float(cos_min),
        float(sin_max),
        float(cos_max),
        bool(use_azim),
        float(az_center),
        float(az_halfwidth),
        float(az_pad),
    )
