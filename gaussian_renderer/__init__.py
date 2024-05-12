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

import math

import torch
from torch.nn import functional as F
from gsplat import project_gaussians, rasterize_gaussians, spherical_harmonics
from scene.gaussian_model import GaussianModel

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    cx = viewpoint_camera.image_width / 2.0
    cy = viewpoint_camera.image_height / 2.0

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)

    tile_size = 16
    means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
        means3D,
        scales,
        scaling_modifier,
        rotations,
        viewmat,
        focal_length_x,
        focal_length_y,
        cx,
        cy,
        height,
        width,
        tile_size,
    )

    if colors.dim() == 3:
        c2w = viewmat.inverse()
        viewdirs = means3D - c2w[:3, 3]
        viewdirs = F.normalize(viewdirs, dim=-1).detach()
        if sh_degree is None:
            sh_degree = int(math.sqrt(colors.shape[1]) - 1)
        colors = spherical_harmonics(sh_degree, viewdirs, colors)

    render_colors, render_alphas = rasterize_gaussians(
        means2d,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        height,
        width,
        tile_size,
        background=bg_color,
        return_alpha=True,
    )
    # [H, W, 3] -> [3, H, W]
    rendered_image = render_colors.permute(2, 0, 1)
    try:
        means2d.retain_grad() # [N, 2]
    except:
        pass
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": means2d,
            "visibility_filter" : radii > 0,
            "radii": radii}
