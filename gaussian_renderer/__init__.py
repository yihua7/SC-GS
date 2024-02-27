#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous


# def quaternion_multiply(q1, q2):
#     w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
#     w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

#     return torch.stack((w, x, y, z), dim=-1)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, d_opacity=None, d_color=None, scaling_modifier=1.0, override_color=None, random_bg_color=False, render_motion=False, detach_xyz=False, detach_scale=False, detach_rot=False, detach_opacity=False, d_rot_as_res=True, scale_const=None, d_rotation_bias=None, force_visible=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # if torch.is_tensor(d_xyz) is False:
    #     means3D = pc.get_xyz
    # else:
    #     means3D = from_homogenous(
    #         torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)
    else:
        opacity = pc.get_opacity if d_opacity is None else pc.get_opacity + d_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier, d_rotation=None if type(d_rotation) is float else d_rotation, gs_rot_bias=d_rotation_bias)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation_bias(d_rotation)
        if d_rotation_bias is not None:
            rotations = quaternion_multiply(d_rotation_bias, rotations)

    if render_motion:
        shs = None
        colors_precomp = torch.zeros_like(pc.get_xyz)
        colors_precomp[..., :1] = pc.motion_mask
        colors_precomp[..., -1:] = 1 - pc.motion_mask
    else:
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            sh_features = torch.cat([pc.get_features[:, :1] + d_color[:, None], pc.get_features[:, 1:]], dim=1) if d_color is not None and type(d_color) is not float else pc.get_features
            if pipe.convert_SHs_python:
                shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = sh_features
        else:
            colors_precomp = override_color

    if detach_xyz:
        means3D = means3D.detach()
    if detach_rot or detach_scale:
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp.detach()
        else:
            rotations = rotations.detach() if detach_rot else rotations
            scales = scales.detach() if detach_scale else scales
    if detach_opacity:
        opacity = opacity.detach()

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "bg_color": bg}


def render_flow(
    pc: GaussianModel,
    viewpoint_camera1,
    viewpoint_camera2,
    d_xyz1, d_xyz2,
    d_rotation1, d_scaling1,
    scaling_modifier=1.0,
    compute_cov3D_python=False,
    scale_const=None,
    d_rot_as_res=True,
    **kwargs
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera1.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera1.FoVy * 0.5)

    # About Motion
    carnonical_xyz = pc.get_xyz.clone()
    xyz_at_t1 = xyz_at_t2 = carnonical_xyz.detach()  # Detach coordinates of Gaussians here
    xyz_at_t1 = xyz_at_t1 + d_xyz1
    xyz_at_t2 = xyz_at_t2 + d_xyz2
    gaussians_homogeneous_coor_t2 = torch.cat([xyz_at_t2, torch.ones_like(xyz_at_t2[..., :1])], dim=-1)
    full_proj_transform = viewpoint_camera2.full_proj_transform if viewpoint_camera2 is not None else viewpoint_camera1.full_proj_transform
    gaussians_uvz_coor_at_cam2 = gaussians_homogeneous_coor_t2 @ full_proj_transform
    gaussians_uvz_coor_at_cam2 = gaussians_uvz_coor_at_cam2[..., :3] / gaussians_uvz_coor_at_cam2[..., -1:]

    gaussians_homogeneous_coor_t1 = torch.cat([xyz_at_t1, torch.ones_like(xyz_at_t1[..., :1])], dim=-1)
    gaussians_uvz_coor_at_cam1 = gaussians_homogeneous_coor_t1 @ viewpoint_camera1.full_proj_transform
    gaussians_uvz_coor_at_cam1 = gaussians_uvz_coor_at_cam1[..., :3] / gaussians_uvz_coor_at_cam1[..., -1:]

    flow_uvz_1to2 = gaussians_uvz_coor_at_cam2 - gaussians_uvz_coor_at_cam1
    
    # Rendering motion mask
    flow_uvz_1to2[..., -1:] = pc.motion_mask

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera1.image_height),
        image_width=int(viewpoint_camera1.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(flow_uvz_1to2[0]),  # Background set as 0
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera1.world_view_transform,
        projmatrix=viewpoint_camera1.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera1.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz + d_xyz1  # About Motion
    means2D = screenspace_points
    opacity = pc.get_opacity

    if scale_const is not None:
        # If providing scale_const, directly use scale_const
        scales = torch.ones_like(pc.get_scaling) * scale_const
        if d_rot_as_res:
            rotations = pc.get_rotation + d_rotation1
        else:
            rotations = pc.get_rotation if type(d_rotation1) is float else quaternion_multiply(d_rotation1, pc.get_rotation)
        cov3D_precomp = None
    else:
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier, d_rotation=None if type(d_rotation1) is float else d_rotation1)
        else:
            scales = pc.get_scaling + d_scaling1
            if d_rot_as_res:
                rotations = pc.get_rotation + d_rotation1
            else:
                rotations = pc.get_rotation if type(d_rotation1) is float else quaternion_multiply(d_rotation1, pc.get_rotation)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=flow_uvz_1to2,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
