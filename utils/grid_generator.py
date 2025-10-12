#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:29:58 2025

@author: arash
"""

import torch

class GridGenerator:
    def __init__(self, grid_size, grid_unc, H_offset):
        """
        grid_size: tuple (width, height, depth) in meters
        grid_unc: tuple (dx, dy, dz) voxel size per axis (resolution)

        The camera is placed at:
        - X (width): center → spans [-width/2, +width/2]
        - Y (height): offset → spans [-bottom_offset, +top_offset]
        - Z (depth): starts from 0 → spans [0, depth]
        """
        self.grid_size = grid_size    # (width, height, depth)
        self.grid_unc = grid_unc      # (dx, dy, dz)
        self.H_offset = H_offset
        self.grid_resolution = tuple(
            int(round(size / res)) for size, res in zip(grid_size, grid_unc)
        )

        self.grid = self._generate_grid()

    def _generate_grid(self):
        """
        Generates a voxel grid with the camera at (0, 0, 0),
        positioned at:
        - Horizontal center of width (X axis)
        - Height offset: from -bottom to +top (Y axis)
        - Starting depth = 0 (Z axis)
        """

        # Dimensions
        width, height, depth = self.grid_size
        dx, dy, dz = self.grid_unc
        nx, ny, nz = self.grid_resolution
        
        num_points = nx * ny * nz

        # Define the physical ranges
        x_range = torch.linspace(-width/2 + dx/2, width/2 - dx/2, steps=nx)                        # Width
        y_range = torch.linspace(-height/2 + dy/2 + self.H_offset,  height/2 - dy/2 + self.H_offset, steps=ny)             # Height
        z_range = torch.linspace(0.0 + dz/2, depth - dz/2, steps=nz)                               # Depth

        # Create meshgrid in (W, H, D) order (i.e., X, Y, Z)
        x, y, z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

        
        grid = torch.stack((x, y, z), dim=0) 
        # order of points from X --> Y --> Depth
        # for each point first: lef&right, elevation, depth
        return {'grid': grid,
                'num_points': num_points,
                'info_1': "In dim = 0 order is width, height, depth",
                'info_2': "In other dms: Width resolution, height resoluion, and depth resolution"}

    def get_grid(self):
        """Returns the voxel grid of shape (3, W, H, D) in world meters."""
        return self.grid


def cam_to_img(grid, P):
    """
    Projects 3D voxel grid to 2D image coordinates.

    Args:
        grid (Tensor): shape (3, W, H, D) with coordinates (X, Y, Z) in meters.
        P (Tensor): shape (3, 4), camera projection matrix.

    Returns:
        pixel_coords (Tensor): shape (2, W, H, D), values are (y, x) in image space.
    """
    device = grid.device
    dtype = grid.dtype

    # Flatten the grid to (N, 3), where N = H*W*D
    X, Y, Z = grid[0], grid[1], grid[2]  # each: (W, H, D)
    W, H, D = X.shape
    xyz = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # shape: (N, 3)

    # Convert to homogeneous coordinates (N, 4)
    ones = torch.ones((xyz.shape[0], 1), dtype=dtype, device=device)
    xyz_h = torch.cat([xyz, ones], dim=1)  # shape: (N, 4)

    # Apply projection matrix: (N, 3) = (N, 4) x (4, 3)^T
    uvw = xyz_h @ P.T  # shape: (N, 3)

    # Normalize to get pixel coordinates (divide by w)
    u = uvw[:, 0] / (uvw[:, 2] + 1e-6)  # pixel x
    v = uvw[:, 1] / (uvw[:, 2] + 1e-6)  # pixel y

    # Stack and reshape to (2, W, H, D)
    pixel_coords = torch.stack([v, u], dim=0).reshape(2, W, H, D)

    return pixel_coords

def grid_for_sample(pixel_coords, img_shape):
    """
    Convert pixel coordinates (2, W, H, D) to normalized grid_sample format
    of shape (1, W, H, D, 2), where 2 = (x=col, y=row) in normalized range [-1, 1].
    
    Args:
        pixel_coords (Tensor): shape (2, W, H, D), pixel coordinates [v, u]
        img_shape (tuple): (H_img, W_img) - original image shape

    Returns:
        Tensor: shape (1, H, W*D, 2), normalized and ordered for grid_sample
    """
    assert pixel_coords.shape[0] == 2, "Input must be (2, H, W, D)"
    
    # Unpack image dimensions
    H_img, W_img = img_shape

    # Unpack voxel grid dimensions
    _, W, H, D = pixel_coords.shape

    # pixel_coords[0] = v (row), pixel_coords[1] = u (col)
    # Need to convert to (H, W*D, 2)
    v = pixel_coords[0]  # (H, W, D)
    u = pixel_coords[1]  # (H, W, D)

    # Normalize u (x direction)
    norm_u = (u / (W_img - 1)) * 2 - 1  # range [-1, 1]
    # Normalize v (y direction)
    norm_v = (v / (H_img - 1)) * 2 - 1  # range [-1, 1]

    # Stack and reorder into shape (H, W*D, 2)
    norm_grid = torch.stack([norm_u, norm_v], dim=-1)  # (W, H, D, 2)
    norm_grid = norm_grid.reshape(1, W*H*D, 1, 2)
    # 1, N, 1, 2 in VU
    return norm_grid


def oob_voxels(pixel_coords, img_size):
    # to find voxels out of the image box
    # grid_cam.shape is (2, res_w, res_h, res_d)
    
    H, W = img_size
    
    v = pixel_coords[0]
    u = pixel_coords[1]
    
    # Compute out-of-bounds mask
    out_of_bounds_mask = (u < 0) | (u >= W) | (v < 0) | (v >= H)
    
    return out_of_bounds_mask