from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from gsplat import rasterization

from splatsim._conversions import GaussianTensors

if TYPE_CHECKING:
    from splatsim.scene import Scene


class Renderer:
    """Renders a scene of Background + RigidBodies using gsplat."""

    def __init__(
        self,
        width: int = 960,
        height: int = 540,
        *,
        device: torch.device = torch.device("cuda"),
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
        radius_clip: float = 0.0,
        exposure: float = 1.0,
        ppisp_knn_k: int = 4,
    ) -> None:
        self.width = width
        self.height = height
        self.device = device
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.exposure = float(exposure)
        self.ppisp_knn_k = int(ppisp_knn_k)
        self._radius_clip = radius_clip
        self._bg_color = torch.tensor(
            [list(background_color)], device=device, dtype=torch.float32
        )  # [1, 3] — shape [C, D] where C=num_cameras
        # Optional equirectangular skybox: an (H, W, 3) [0, 1] panorama sampled
        # by world ray direction behind the Gaussians. A LiDAR-derived cloud has
        # no sky in it; a finite Gaussian dome shows parallax and does not fit
        # SPZ's position range, so the sky rides as a directional texture.
        self.skybox: Tensor | None = None

    def render(
        self,
        viewmat: Tensor,
        K: Tensor,
        *,
        scene: Scene | None = None,
        camera_name: str | None = None,
        shared: "CameraGaussians | None" = None,
    ) -> Tensor:
        """Render the scene and return an [H, W, 3] float32 RGB image (0-1).

        ``shared`` supplies a pre-gathered Gaussian set (see
        :func:`gather_camera_rig`) so a multi-camera rig pays one LOD gather and
        holds one transient buffer for the whole rig instead of one per camera.
        """
        camera_pos: Tensor | None = None
        if scene is not None and (scene.lod_enabled or scene.ppisp_tables is not None):
            # viewmat is world-to-camera: [R | t], camera_pos = -R^T @ t
            R = viewmat[:3, :3]
            t = viewmat[:3, 3]
            camera_pos = -(R.T @ t)

        if shared is None:
            shared = gather_camera(scene, camera_pos)

        if shared is None:
            return torch.zeros(
                self.height, self.width, 3, device=self.device, dtype=torch.float32
            )

        all_means = shared.means
        all_quats = shared.quats
        all_scales = shared.scales
        all_opacities = shared.opacities
        all_colors = shared.colors
        sh_degree = shared.sh_degree

        # Camera tensors: add batch dimension
        viewmats = viewmat.unsqueeze(0).to(self.device)  # [1, 4, 4]
        Ks = K.unsqueeze(0).to(self.device)  # [1, 3, 3]

        # An explicit renderer skybox wins; otherwise the one the scene carried
        # in from its bundle (see Scene.skybox / _usdz.load_skybox).
        skybox = (
            self.skybox if self.skybox is not None else getattr(scene, "skybox", None)
        )

        render_colors, _render_alphas, _meta = rasterization(
            means=all_means,
            quats=all_quats,
            scales=all_scales,
            opacities=all_opacities,
            colors=all_colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.width,
            height=self.height,
            sh_degree=sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            radius_clip=self._radius_clip,
            render_mode="RGB",
            packed=False,
            # With a skybox, the uncovered pixels are the skybox's, so render the
            # Gaussians over black — otherwise render_colors already holds
            # (1 - alpha) * bg_color and adding the skybox would double-count it.
            backgrounds=None if skybox is not None else self._bg_color,
        )

        rgb = render_colors[0]
        if skybox is not None:
            alpha = _render_alphas[0]  # [H, W, 1]
            rgb = rgb + (1.0 - alpha) * self._sample_skybox(viewmat, K, skybox)
        tables = scene.ppisp_tables if scene is not None else None
        if tables is not None and camera_name is not None and camera_pos is not None:
            from splatsim.ppisp import apply_ppisp

            rgb = apply_ppisp(
                tables,
                rgb,
                camera_name,
                camera_pos,
                k=self.ppisp_knn_k,
            )
        elif self.exposure != 1.0:
            rgb = rgb * self.exposure
        return rgb  # [H, W, 3]

    def _sample_skybox(self, viewmat: Tensor, K: Tensor, skybox: Tensor) -> Tensor:
        """Sample the equirect skybox along every pixel's world ray. [H, W, 3].

        The panorama convention matches the baker's: longitude ``atan2(y, x)``
        across the width, elevation ``asin(z)`` down the height with the zenith
        on the top row. The sky is at infinity, so only the camera rotation
        matters — pixel ``(u, v)`` maps to
        ``R_cam→world · normalize(K^-1 [u, v, 1])``.
        """
        import math

        import torch.nn.functional as functional

        dev = skybox.device
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        us = (torch.arange(self.width, device=dev, dtype=torch.float32) - cx) / fx
        vs = (torch.arange(self.height, device=dev, dtype=torch.float32) - cy) / fy
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        d_opt = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
        d_opt = d_opt / d_opt.norm(dim=-1, keepdim=True)
        dirs = d_opt @ viewmat[:3, :3].to(dev)  # R_cam→world = R^T, applied as d·R
        d = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-12)
        lon = torch.atan2(d[..., 1], d[..., 0])
        lat = torch.asin(d[..., 2].clamp(-1.0, 1.0))
        gx = ((lon / (2.0 * math.pi) + 0.5) % 1.0) * 2.0 - 1.0
        gy = (0.5 - lat / math.pi).clamp(0.0, 1.0) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        img = skybox.permute(2, 0, 1).unsqueeze(0)  # [1, 3, Hs, Ws]
        out = functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return out[0].permute(1, 2, 0)  # [H, W, 3]


@dataclass
class CameraGaussians:
    """A camera-ready Gaussian set, already LOD-filtered and concatenated.

    Produced by :func:`gather_camera` / :func:`gather_camera_rig` and consumed
    by :meth:`Renderer.render` via ``shared=``. Unlike the LiDAR equivalent this
    keeps ``colors`` (the appearance path needs the SH block).
    """

    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    sh_degree: int | None

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def _tensors(self) -> "list[Tensor]":
        return [self.means, self.quats, self.scales, self.opacities, self.colors]

    def nbytes(self) -> int:
        """Device bytes held by this set (for VRAM accounting / logging)."""
        return int(sum(t.numel() * t.element_size() for t in self._tensors()))

    def record_stream(self, stream) -> None:
        """Mark these buffers as in use by *stream* (see the LiDAR equivalent)."""
        for t in self._tensors():
            if t.is_cuda:
                t.record_stream(stream)


def _pack(tensor_list: "list[GaussianTensors]") -> "CameraGaussians | None":
    if not tensor_list:
        return None
    sh_degrees = {t.sh_degree for t in tensor_list}
    if len(sh_degrees) == 1 and sh_degrees.pop() > 0:
        sh_degree: int | None = tensor_list[0].sh_degree
    else:
        sh_degree = None
    return CameraGaussians(
        means=torch.cat([t.means for t in tensor_list], dim=0),
        quats=torch.cat([t.quats for t in tensor_list], dim=0),
        scales=torch.cat([t.scales for t in tensor_list], dim=0),
        opacities=torch.cat([t.opacities for t in tensor_list], dim=0),
        colors=torch.cat([t.colors for t in tensor_list], dim=0),
        sh_degree=sh_degree,
    )


def gather_camera(
    scene: "Scene | None", camera_pos: Tensor | None
) -> "CameraGaussians | None":
    """Collect the Gaussians a single camera at *camera_pos* would rasterize."""
    if scene is None:
        return None
    return _pack(scene.collect_tensors(camera_pos if scene.lod_enabled else None))


def gather_camera_rig(
    scene: "Scene | None", viewmats: "list[Tensor]"
) -> "CameraGaussians | None":
    """One LOD gather covering every camera on a rig.

    LOD tiers are picked per cell from the NEAREST camera (the filter accepts an
    ``[S, 3]`` position set), so the shared set is a superset of what any single
    camera would have selected -- no camera is handed a coarser tier than it
    asked for. In exchange an N-camera rig performs one gather and holds one
    transient Gaussian buffer per frame instead of N, which is the dominant VRAM
    term when a rig renders concurrently.
    """
    if scene is None or not viewmats:
        return None
    if not scene.lod_enabled:
        return _pack(scene.collect_tensors(None))
    positions = torch.stack(
        [-(vm[:3, :3].T @ vm[:3, 3]) for vm in viewmats], dim=0
    ).detach()  # [S, 3]
    return _pack(scene.collect_tensors(positions))


def render_cameras_concurrent(
    renderers: "list[Renderer]",
    viewmats: "list[Tensor]",
    Ks: "list[Tensor]",
    *,
    scene: "Scene | None" = None,
    camera_names: "list[str | None] | None" = None,
    shared_gather: bool = True,
) -> "list[Tensor]":
    """Render every camera on a rig for one frame off one shared gather.

    Mirrors :func:`splatsim.lidar_renderer.render_lidars_concurrent`: one gather
    for the rig (``shared_gather``) plus one CUDA stream per camera, so the
    per-camera setup/launch pipelines overlap without multiplying peak VRAM.
    ``SPLATSIM_CAMERA_CONCURRENT=0`` forces a single stream.
    """
    import os

    if not renderers:
        return []
    names = camera_names or [None] * len(renderers)
    shared = gather_camera_rig(scene, viewmats) if shared_gather else None

    def _one(i: int) -> Tensor:
        return renderers[i].render(
            viewmats[i], Ks[i], scene=scene, camera_name=names[i], shared=shared
        )

    if (
        len(renderers) <= 1
        or not torch.cuda.is_available()
        or os.environ.get("SPLATSIM_CAMERA_CONCURRENT", "1") == "0"
    ):
        return [_one(i) for i in range(len(renderers))]

    try:
        from splatsim.lidar_renderer import _side_streams

        current = torch.cuda.current_stream()
        # Cached side streams — see _side_streams: allocating them per frame
        # costs a bimodal multi-hundred-ms stall on large scenes.
        streams = _side_streams(len(renderers), renderers[0].device)
        # Side streams do not inherit the current stream's pending work, so they
        # must wait for the shared gather explicitly (see the LiDAR rig path for
        # what happens otherwise: the first camera reads a half-written buffer).
        for st in streams:
            st.wait_stream(current)
        outs: dict[int, Tensor] = {}
        for i, st in enumerate(streams):
            with torch.cuda.stream(st):
                outs[i] = _one(i)
            if shared is not None:
                shared.record_stream(st)
            outs[i].record_stream(current)
        for st in streams:
            current.wait_stream(st)
        torch.cuda.synchronize()
        return [outs[i] for i in range(len(renderers))]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return [_one(i) for i in range(len(renderers))]
