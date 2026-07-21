from __future__ import annotations

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
        packed: bool = True,
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
        self._packed = packed
        self._bg_color = torch.tensor(
            [list(background_color)], device=device, dtype=torch.float32
        )  # [1, 3] — shape [C, D] where C=num_cameras

    def render(
        self,
        viewmat: Tensor,
        K: Tensor,
        *,
        scene: Scene | None = None,
        camera_name: str | None = None,
    ) -> Tensor:
        """Render the scene and return an [H, W, 3] float32 RGB image (0-1)."""
        tensor_list: list[GaussianTensors] = []

        camera_pos: Tensor | None = None
        if scene is not None:
            if scene.lod_enabled or scene.ppisp_tables is not None:
                # viewmat is world-to-camera: [R | t], camera_pos = -R^T @ t
                R = viewmat[:3, :3]
                t = viewmat[:3, 3]
                camera_pos = -(R.T @ t)

            tensor_list = scene.collect_tensors(
                camera_pos if scene.lod_enabled else None
            )

        if not tensor_list:
            return torch.zeros(
                self.height, self.width, 3, device=self.device, dtype=torch.float32
            )

        # Concatenate all Gaussians
        all_means = torch.cat([t.means for t in tensor_list], dim=0)
        all_quats = torch.cat([t.quats for t in tensor_list], dim=0)
        all_scales = torch.cat([t.scales for t in tensor_list], dim=0)
        all_opacities = torch.cat([t.opacities for t in tensor_list], dim=0)
        all_colors = torch.cat([t.colors for t in tensor_list], dim=0)

        # Determine SH degree (use SH only if all sources agree)
        sh_degrees = {t.sh_degree for t in tensor_list}
        if len(sh_degrees) == 1 and sh_degrees.pop() > 0:
            sh_degree: int | None = tensor_list[0].sh_degree
        else:
            sh_degree = None

        # Camera tensors: add batch dimension
        viewmats = viewmat.unsqueeze(0).to(self.device)  # [1, 4, 4]
        Ks = K.unsqueeze(0).to(self.device)  # [1, 3, 3]

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
            packed=self._packed,
            backgrounds=self._bg_color,
        )

        rgb = render_colors[0]
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
