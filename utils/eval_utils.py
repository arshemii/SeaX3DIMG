#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 11:48:47 2025

@author: arash
"""
import torch
from shapely.geometry import Polygon
import numpy as np


def grid3d_to_grid2d(grid):
    bev_grid = grid[:, 0, :, :][:, :, [0, 2]]
    # bev_grid will be 100, 70, 2
    return bev_grid


def get_corners(cx, cz, w, l, yaw):
    """
    Compute the (x,z) coordinates of the 4 corners of the rotated box.
    Returns np.array of shape (4,2).
    
    ChatGPT generated function!
    """
    # half-dimensions
    w2, l2 = w / 2.0, l / 2.0

    # corners in box local frame (before rotation)
    # order: [front-left, front-right, back-right, back-left]
    corners = np.array([
        [ w2,  l2],
        [-w2,  l2],
        [-w2, -l2],
        [ w2, -l2]
    ])

    # rotation matrix around yaw (z-axis rotation in BEV plane)
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # rotate + translate
    rotated = corners @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cz

    return rotated

def bev_iou(box1, box2):
    """
    Compute the IoU of two oriented 2D bounding boxes in BEV (x-z plane).

    Args:
        box1: list [cx, cz, w, l, yaw]
              cx, cz = box center coordinates
              w, l   = width (x-axis extent), length (z-axis extent)
              yaw    = rotation angle in radians (counter-clockwise, from x-axis)
        box2: same format as box1

    Returns:
        iou: float, intersection over union (0..1)
    
    ChatGPT generated function!
    """

    # convert both boxes to polygons
    poly1 = Polygon(get_corners(*box1))
    poly2 = Polygon(get_corners(*box2))

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    # intersection & union
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter

    if union <= 0:
        return 0.0

    return inter / union



def compress_tensor(tensor):
    """
    tensor: [B, 10, W, D]
    returns: [B, 8, W, D]
    
    ChatGPT generated function
    """
    B, C, W, D = tensor.shape
    assert C == 10, "Input tensor must have 10 channels"

    # Objectness
    objectness = tensor[:, 0:1, :, :]  # [B,1,W,D]

    # Class predictions
    class_logits = tensor[:, 1:5, :, :]                # [B,4,W,D]
    class_prob, class_idx = torch.max(class_logits, 1) # [B,W,D], [B,W,D]

    # Expand dims for stacking
    class_idx = class_idx.unsqueeze(1).float()   # [B,1,W,D]
    class_prob = class_prob.unsqueeze(1)         # [B,1,W,D]

    # Regression heads
    regressed = tensor[:, 5:10, :, :]  # [B,5,W,D]

    # Concatenate all parts
    out = torch.cat([objectness, class_idx, class_prob, regressed], dim=1)  # [B,8,W,D]

    return out


def add_voxel_centers(pred_tensor, grid):
    """
    Add voxel centers (x, z) to predicted offsets.

    Args:
        pred_tensor: [B, 8, W, D]
        grid: [W, D, 2] with voxel centers (x, z)

    Returns:
        pred_with_centers: [B, 8, W, D]
            where channel 3 and 4 now contain absolute (x, z) centers
    """
    B, C, W, D = pred_tensor.shape
    assert C == 8, "Expected pred_tensor with 8 channels"
    assert grid.shape == (W, D, 2), "Grid must be [W, D, 2]"

    pred = pred_tensor.clone()

    # Broadcast grid to [1, W, D, 2]
    grid_exp = grid.unsqueeze(0)

    # Add voxel centers to offsets
    pred[:, 3, :, :] = pred[:, 3, :, :] + grid_exp[..., 0]  # cx_abs
    pred[:, 4, :, :] = pred[:, 4, :, :] + grid_exp[..., 1]  # cz_abs

    return pred

def filter_detections(tensor, obj_thresh=0.5):
    """
    Filter detections based on objectness threshold.

    Args:
        tensor: [B, 8, W, D] tensor
        obj_thresh: float, threshold for objectness

    Returns:
        List of length B, where each element is a tensor of shape [N, 8].
        (N = number of detections kept for that batch sample)
    """
    B, C, W, D = tensor.shape
    assert C == 8, "Input tensor must have 8 channels"

    results = []
    for b in range(B):
        # [8, W, D] → [8, W*D]
        dets = tensor[b].reshape(C, -1).permute(1, 0)  # [W*D, 8]

        # Filter by objectness
        mask = dets[:, 0] > obj_thresh
        dets = dets[mask]

        results.append(dets)  # [N, 8]

    return results