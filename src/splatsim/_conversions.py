from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

import spz

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


@dataclass
class GaussianTensors:
    """GPU-ready Gaussian parameters in gsplat convention."""

    means: Tensor  # [N, 3]
    quats: Tensor  # [N, 4] wxyz (gsplat convention)
    scales: Tensor  # [N, 3] actual scale (exp applied)
    opacities: Tensor  # [N] range [0, 1] (sigmoid applied)
    colors: Tensor  # [N, 3] RGB or [N, K, 3] SH coefficients
    sh_degree: int  # 0 = RGB only, 1-3 = SH degree
    # Optional per-Gaussian LiDAR attributes. Populated only for scenes trained
    # with tier4/gaussian_factory (see PLY properties `lidar_intensity_raw`,
    # `lidar_raydrop_logit`). Standard 3DGS scenes leave these as None and the
    # LiDAR renderer falls back to SH-derived intensity and a fixed low raydrop.
    intensity_raw: Tensor | None = None  # [N] unbounded, sigmoid at render
    raydrop_logit: Tensor | None = None  # [N] unbounded logit
    # Optional per-Gaussian LiDAR participation mask. Emitted by 3dgs_io >=
    # v1.1.0 as the sidecar's `lidar_mask` channel: True (1) = the Gaussian
    # participates in LiDAR / near-field geometry, False (0) = appearance-only
    # / far-field (excluded from the LiDAR geometry pass). ``None`` means the
    # channel is absent, i.e. all Gaussians participate (backward compatible
    # with old 2-channel sidecars). Consumed only by the LiDAR renderer; the
    # RGB/camera path ignores it.
    lidar_mask: Tensor | None = None  # [N] bool; True = participates in LiDAR
    # Optional per-Gaussian view-dependent (spherical-harmonics) raydrop bands.
    # Emitted by 3dgs_io >= v1.2.0 as the sidecar's `raydrop_sh` trailing block:
    # the *higher-order* SH bands only (`[N, (deg+1)**2 - 1]`), while the band-0
    # (DC) term stays in the scalar `raydrop_logit`. The LiDAR renderer evaluates
    # these at each Gaussian's sensor-view direction (exactly like colour SH) to
    # get a view-dependent raydrop logit. ``None`` means no higher bands, i.e.
    # the scalar `raydrop_logit` is used directly (backward compatible).
    raydrop_sh: Tensor | None = None  # [N, (deg+1)**2 - 1] higher-order SH bands

    def __getitem__(self, idx: Tensor | slice) -> GaussianTensors:
        """Return a new instance with all tensor fields indexed/sliced by *idx*."""
        return GaussianTensors(
            means=self.means[idx],
            quats=self.quats[idx],
            scales=self.scales[idx],
            opacities=self.opacities[idx],
            colors=self.colors[idx],
            sh_degree=self.sh_degree,
            intensity_raw=None
            if self.intensity_raw is None
            else self.intensity_raw[idx],
            raydrop_logit=None
            if self.raydrop_logit is None
            else self.raydrop_logit[idx],
            lidar_mask=None if self.lidar_mask is None else self.lidar_mask[idx],
            raydrop_sh=None if self.raydrop_sh is None else self.raydrop_sh[idx],
        )


def cloud_to_tensors(
    cloud: spz.GaussianCloud,  # ty: ignore[unresolved-attribute]
    device: torch.device,
    *,
    use_sh: bool = False,
) -> GaussianTensors:
    """Convert an spz GaussianCloud to gsplat-ready GPU tensors.

    Handles all convention differences:
    - Quaternion order: spz (x,y,z,w) -> gsplat (w,x,y,z)
    - Scales: log-scale -> actual scale via exp()
    - Opacities: logit -> [0,1] via sigmoid()
    - Colors: SH DC pre-activation -> RGB via dc * SH_C0 + 0.5
    """
    n = cloud.num_points

    # Positions: flat (N*3,) -> (N, 3)
    positions = np.array(cloud.positions, dtype=np.float32).reshape(n, 3)
    means = torch.from_numpy(positions).to(device)

    # Quaternions: spz (x,y,z,w) -> gsplat (w,x,y,z)
    quats_xyzw = np.array(cloud.rotations, dtype=np.float32).reshape(n, 4)
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
    quats = torch.from_numpy(quats_wxyz).to(device)

    # Scales: log-scale -> actual scale
    log_scales = np.array(cloud.scales, dtype=np.float32).reshape(n, 3)
    scales = torch.from_numpy(log_scales).to(device).exp()

    # Opacities: logit -> sigmoid
    alpha_logits = np.array(cloud.alphas, dtype=np.float32)
    opacities = torch.from_numpy(alpha_logits).to(device).sigmoid()

    # Colors
    sh_degree = cloud.sh_degree
    raw_colors = np.array(cloud.colors, dtype=np.float32).reshape(n, 3)

    if use_sh and sh_degree > 0:
        # Build full SH coefficient tensor [N, K, 3]
        # DC component: raw spz colors value (gsplat applies SH basis internally)
        dc = torch.from_numpy(raw_colors).to(device).unsqueeze(1)  # [N, 1, 3]

        # Higher-order SH coefficients
        raw_sh = np.array(cloud.sh, dtype=np.float32)
        num_higher_coeffs = len(raw_sh) // (n * 3)
        higher_sh = torch.from_numpy(raw_sh.reshape(n, num_higher_coeffs, 3)).to(
            device
        )  # [N, K-1, 3]

        colors = torch.cat([dc, higher_sh], dim=1)  # [N, K, 3]
    else:
        # RGB mode: convert SH DC to actual RGB
        colors_rgb = raw_colors * SH_C0 + 0.5
        colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
        colors = torch.from_numpy(colors_rgb).to(device)  # [N, 3]
        sh_degree = 0

    return GaussianTensors(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh_degree=sh_degree,
    )


def attach_lidar_attrs(
    tensors: GaussianTensors,
    attrs: dict[str, np.ndarray],
    num_points: int,
    device: torch.device,
    *,
    source: str,
) -> None:
    """Move a decoded LiDAR attribute dict onto a cloud's tensors.

    Shared by the two readers that produce Gaussians from a scene bundle —
    background chunks (:mod:`splatsim._usdz`) and rigid actor assets
    (:mod:`splatsim.actor_assets`) — because an actor's per-Gaussian LiDAR
    attributes are the same payload a chunk's are. ``source`` labels the
    payload in error messages.
    """
    intensity = attrs.get("lidar_intensity_raw")
    raydrop = attrs.get("lidar_raydrop_logit")
    if intensity is None or raydrop is None:
        raise ValueError(f"{source}: incomplete LiDAR attributes")
    if len(intensity) != num_points or len(raydrop) != num_points:
        raise ValueError(f"{source}: LiDAR attribute count does not match")
    tensors.intensity_raw = torch.from_numpy(intensity).to(device)
    tensors.raydrop_logit = torch.from_numpy(raydrop).to(device)
    # Optional per-Gaussian LiDAR participation mask (3dgs_io >= v1.1.0).
    # Absent for old 2-channel payloads → leave None (all Gaussians
    # participate). Stored {0.0, 1.0} floats; threshold to a bool tensor on
    # the render device.
    mask = attrs.get("lidar_mask")
    if mask is not None:
        if len(mask) != num_points:
            raise ValueError(f"{source}: LiDAR attribute count does not match")
        tensors.lidar_mask = torch.as_tensor(mask, device=device) > 0.5
    # Optional view-dependent (SH) raydrop bands (3dgs_io >= v1.2.0,
    # version-2 payload). Shape (num_points, (deg+1)**2 - 1): the
    # higher-order bands only; the DC term is in lidar_raydrop_logit.
    # Absent for version-1 payloads → leave None (scalar raydrop).
    raydrop_sh = attrs.get("raydrop_sh")
    if raydrop_sh is not None:
        if raydrop_sh.shape[0] != num_points:
            raise ValueError(f"{source}: LiDAR attribute count does not match")
        tensors.raydrop_sh = torch.from_numpy(np.ascontiguousarray(raydrop_sh)).to(
            device
        )


def quat_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert a (w,x,y,z) quaternion to a 3x3 rotation matrix.

    Differentiable torch counterpart of
    :func:`splatsim._geometry.quat_to_matrix` (same wxyz convention); kept here
    because it stays on-device and preserves gradients for the render path.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]
            ),
            torch.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]
            ),
            torch.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
            ),
        ]
    )  # [3, 3]


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Quaternion multiplication in (w,x,y,z) convention.

    q1: [4] single quaternion
    q2: [N, 4] batch of quaternions
    Returns: [N, 4]
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )  # [N, 4]


#: Highest SH band the 3DGS / SPZ colour layout carries.
MAX_SH_DEGREE = 3

#: Beyond this much pitch/roll, rotating the colour SH about ``+Z`` alone is no
#: longer a faithful re-expression of the bands (see :func:`rotate_sh_about_z`).
MAX_NON_YAW_RAD = 0.035  # ~2 degrees

#: ``{(coefs, device, dtype, dc_offset): (order, partner, sign)}`` — see
#: :func:`_sh_yaw_mixing_tables`. Keyed on everything the tables depend on, so a
#: scene's actors share one set no matter how many instances are posed per frame.
_SH_TABLE_CACHE: dict[
    tuple[int, str, torch.dtype, int], tuple[Tensor, Tensor, Tensor]
] = {}


def yaw_from_quat(rotation: Tensor) -> Tensor:
    """The ``+Z`` heading of a wxyz quaternion, as a 0-dim tensor on its device.

    This is a *body* heading — zero along ``+X``, the frame every scene-bundle
    pose uses. Not to be confused with the viewer yaw in
    :mod:`splatsim.viewer` / :mod:`splatsim._usdz`, which is measured off a
    camera forward vector with zero along ``-Y``.

    Read straight off the quaternion rather than through
    :func:`quat_to_rotation_matrix`: this sits on the per-frame transform path
    and only two matrix entries are wanted.
    """
    _w, x, y, z = rotation.unbind()
    return torch.atan2(2.0 * (x * y + _w * z), 1.0 - 2.0 * (y * y + z * z))


def tilt_from_quat(rotation: Tensor) -> Tensor:
    """How far a wxyz quaternion departs from a pure ``+Z`` yaw, in radians.

    The angle between the body's own ``+Z`` and world ``+Z``: zero for a pure
    yaw, ``pi`` for an upside-down body. 0-dim tensor on the input's device.
    """
    _w, x, y, _z = rotation.unbind()
    return torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0))


def _sh_yaw_mixing_tables(
    coefs: int, device: torch.device, dtype: torch.dtype, dc_offset: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-coefficient ``(order, partner index, sine sign)`` for a ``+Z`` rotation.

    Rotating real SH about the polar axis mixes only the ``(m, -m)`` pair
    within each band, so the whole rotation is one gather plus one blend once
    these three tables are known.

    ``dc_offset`` is ``1`` when band 0 occupies index 0 (the colour layout) and
    ``0`` when the DC term is stored outside the array (the raydrop layout,
    whose band-0 value lives in the scalar ``raydrop_logit``). Everything else
    about the two layouts is identical, which is why one table builder serves
    both.
    """
    key = (coefs, str(device), dtype, dc_offset)
    cached = _SH_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    order = [0] * coefs  # |m| per coefficient; 0 leaves the coefficient alone
    partner = list(range(coefs))  # the (m, -m) partner; self when m == 0
    sign = [0.0] * coefs  # +1 on the m > 0 slot, -1 on the m < 0 slot
    for degree in range(1, MAX_SH_DEGREE + 1):
        base = degree * degree - 1 + dc_offset
        if base + 2 * degree >= coefs:
            break
        for m in range(1, degree + 1):
            i_pos, i_neg = base + degree + m, base + degree - m
            order[i_pos] = order[i_neg] = m
            partner[i_pos], partner[i_neg] = i_neg, i_pos
            sign[i_pos], sign[i_neg] = 1.0, -1.0
    tables = (
        # In the coefficients' own dtype: a float32 table would quietly round a
        # float64 rotation down to single precision.
        torch.tensor(order, device=device, dtype=dtype),
        torch.tensor(partner, device=device, dtype=torch.long),
        torch.tensor(sign, device=device, dtype=dtype),
    )
    _SH_TABLE_CACHE[key] = tables
    return tables


def _rotate_sh_bands(coeffs: Tensor, yaw: Tensor, *, dc_offset: int) -> Tensor:
    """Mix each ``(m, -m)`` pair of ``coeffs`` for a ``+Z`` rotation by ``yaw``.

    Accepts the two coefficient layouts splatsim carries — ``[N, K, 3]`` colour
    bands and ``[N, K]`` scalar raydrop bands — since the rotation touches the
    band axis only.
    """
    order, partner, sign = _sh_yaw_mixing_tables(
        coeffs.shape[1], coeffs.device, coeffs.dtype, dc_offset
    )
    angle = order * yaw
    cos_m = torch.cos(angle)
    sin_m = torch.sin(angle) * sign
    if coeffs.dim() == 3:
        return (
            coeffs * cos_m[None, :, None] + coeffs[:, partner, :] * sin_m[None, :, None]
        )
    return coeffs * cos_m[None, :] + coeffs[:, partner] * sin_m[None, :]


def rotate_sh_about_z(colors: Tensor, yaw: Tensor) -> Tensor:
    """Rotate real-SH colour coefficients about ``+Z`` by ``yaw``.

    ``colors`` is ``[N, K, 3]`` in the 3DGS / SPZ layout: index 0 is the DC
    band and the rest cover bands ``1..degree`` with ``m`` ascending from
    ``-l`` to ``+l`` inside each band. Rotation about the SH polar axis mixes
    only the ``(m, -m)`` pair, so this is exact and closed-form — no Wigner-D
    machinery — and the DC band is rotation-invariant.

    This is the torch counterpart of ``3dgs_io.rotate_sh_about_z``: the
    returned coefficients evaluate at direction ``d`` to what the input
    evaluated at ``R_z(yaw) @ d``, which is exactly what instancing an
    object-local asset through a world pose needs.

    Runs on the per-frame transform path, so it is a whole-tensor blend rather
    than a loop over bands: ``cos`` is ``1`` and ``sin`` is ``0`` on the DC and
    ``m == 0`` slots, which leaves them untouched for free.
    """
    if colors.dim() != 3:
        raise ValueError(
            f"rotate_sh_about_z expects [N, K, 3] colors, got {tuple(colors.shape)}"
        )
    return _rotate_sh_bands(colors, yaw, dc_offset=1)


def rotate_raydrop_sh_about_z(raydrop_sh: Tensor, yaw: Tensor) -> Tensor:
    """Rotate view-dependent raydrop bands about ``+Z`` by ``yaw``.

    ``raydrop_sh`` is ``[N, (degree + 1)**2 - 1]``: a single scalar channel
    holding bands ``1..degree`` only, because the band-0 term lives in the
    scalar ``raydrop_logit`` (which is rotation-invariant and needs nothing).
    Same closed form as :func:`rotate_sh_about_z`, one index shift apart.

    The LiDAR renderer evaluates these along the world-space sensor-to-Gaussian
    ray, exactly as gsplat evaluates colour SH along the camera ray. So a posed
    actor's bands have to be re-expressed in the world frame for the same reason
    its colours do — otherwise the drop probability of a car facing east is
    sampled with the angular pattern it had facing north.
    """
    if raydrop_sh.dim() != 2:
        raise ValueError(
            "rotate_raydrop_sh_about_z expects [N, K] bands, got "
            f"{tuple(raydrop_sh.shape)}"
        )
    return _rotate_sh_bands(raydrop_sh, yaw, dc_offset=0)


def apply_rigid_transform(
    tensors: GaussianTensors,
    position: Tensor,
    rotation: Tensor,
    *,
    rotate_sh: bool = False,
) -> GaussianTensors:
    """Apply a rigid body transform (translation + rotation) to Gaussian tensors.

    Args:
        tensors: Base Gaussian tensors to transform.
        position: [3] world-space translation.
        rotation: [4] wxyz quaternion for orientation.
        rotate_sh: Re-express the view-dependent bands — colour SH *and*
            ``raydrop_sh`` — in the world frame. Off by default so existing
            rigid bodies keep their behaviour;
            :class:`~splatsim.actor_assets.ActorAssetLibrary` turns it on,
            because a dynamic actor's heading changes every frame: leaving its
            specular bands in the object frame makes highlights spin with the
            car, and leaving its raydrop bands there samples the drop
            probability of a car facing east with the pattern it had facing
            north. Only the yaw component is applied (see
            :func:`rotate_sh_about_z`); a pose tilted more than
            :data:`MAX_NON_YAW_RAD` out of the ground plane is reported by
            :meth:`~splatsim.rigid_body.RigidBody.sh_rotation_tilt` rather than
            silently approximated.

    Returns:
        New GaussianTensors with transformed means and quats.
    """
    rot_mat = quat_to_rotation_matrix(rotation)  # [3, 3]

    # Transform positions: R @ p + t
    new_means = tensors.means @ rot_mat.T + position.unsqueeze(0)

    # Compose quaternions
    new_quats = quat_multiply(rotation, tensors.quats)

    colors = tensors.colors
    raydrop_sh = tensors.raydrop_sh
    # The colour bands and the raydrop bands are the same kind of quantity —
    # a per-Gaussian angular pattern evaluated along a world-space ray — so
    # they rotate together, off one yaw. `raydrop_sh` is independent of
    # `sh_degree`: an RGB-mode asset can still carry view-dependent raydrop.
    rotates_colors = rotate_sh and tensors.sh_degree > 0 and colors.dim() == 3
    rotates_raydrop = rotate_sh and raydrop_sh is not None
    if rotates_colors or rotates_raydrop:
        # NEGATIVE yaw. The rotators re-express coefficients so that
        # ``f_out(d) == f_in(R_z(angle) @ d)``; instancing needs the world-frame
        # bands to answer for the object frame, ``f_world(d) == f_object(R^-1 d)``
        # — the inverse. (3dgs_io's extract_actor_asset goes the other way,
        # world -> object, and passes +yaw.) Easy to get backwards, and only a
        # test that evaluates the bands along a real ray catches it.
        yaw = -yaw_from_quat(rotation)
        if rotates_colors:
            colors = rotate_sh_about_z(colors, yaw)
        if rotates_raydrop:
            assert raydrop_sh is not None  # noqa: S101 - implied by rotates_raydrop
            raydrop_sh = rotate_raydrop_sh_about_z(raydrop_sh, yaw)

    return GaussianTensors(
        means=new_means,
        quats=new_quats,
        scales=tensors.scales,
        opacities=tensors.opacities,
        colors=colors,
        sh_degree=tensors.sh_degree,
        intensity_raw=tensors.intensity_raw,
        raydrop_logit=tensors.raydrop_logit,
        lidar_mask=tensors.lidar_mask,
        raydrop_sh=raydrop_sh,
    )
