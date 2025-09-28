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
    Compute corners of a single box in BEV (numpy).
    """
    w2, l2 = w / 2.0, l / 2.0
    corners = np.array([
        [ w2,  l2],
        [-w2,  l2],
        [-w2, -l2],
        [ w2, -l2]
    ])
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    rotated = corners @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cz
    return rotated

def bev_iou(dets: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between N detections and 1 gt box in BEV.

    Args:
        dets: Tensor [N, 5] -> cx, cz, w, l, yaw
        gt:   Tensor [5]    -> cx, cz, w, l, yaw
    
    Returns:
        Tensor [N] of IoUs
    """
    dets_np = dets.cpu().numpy()
    gt_np = gt.cpu().numpy()

    # Convert GT to polygon
    poly_gt = Polygon(get_corners(*gt_np))
    if not poly_gt.is_valid:
        return torch.zeros(dets.shape[0], device=dets.device)

    ious = []
    for det in dets_np:
        poly_det = Polygon(get_corners(*det))
        if not poly_det.is_valid:
            ious.append(0.0)
            continue

        inter = poly_det.intersection(poly_gt).area
        union = poly_det.area + poly_gt.area - inter
        if union <= 0:
            ious.append(0.0)
        else:
            ious.append(inter / union)

    return torch.tensor(ious, device=dets.device, dtype=torch.float32)



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

def drop_far_dets(dets_list, Z_threshold):
    """
    Drop detections whose z-center is larger than Z_threshold.

    Args:
        dets_list: list of length B, each element is [N, 8] tensor of detections
        Z_threshold: float, max allowed z coordinate

    Returns:
        new_dets_list: list of length B, each element is [M, 8] tensor (M <= N)
    """
    new_dets_list = []
    for dets in dets_list:
        if dets.numel() == 0:  
            # No detections for this batch element
            new_dets_list.append(dets)  
            continue

        # Keep only where z <= Z_threshold
        mask = dets[:, 4] <= Z_threshold
        filtered = dets[mask]

        new_dets_list.append(filtered)

    return new_dets_list

def drop_far_gts(gt_labels, Z_threshold):
    """
    Filter GT labels by Z threshold, and keep only bbox_bev and category.

    Args:
        gt_labels: list of length B
            Each element is a list of dicts with keys:
                'category', 'bbox3d', 'bbox_bev'
        Z_threshold: float, max allowed cz (forward distance)

    Returns:
        filtered_gt: list of length B
            Each element is a list of dicts with keys:
                'category', 'bbox_bev'
    """
    filtered_gt = []
    for batch_gts in gt_labels:
        batch_result = []
        for gt in batch_gts:
            cz = gt['bbox_bev'][3]  # bbox_bev = [w, l, cx, cz, yaw]
            if cz <= Z_threshold:
                batch_result.append({
                    'category': gt['category'],
                    'bbox_bev': gt['bbox_bev']
                })
        filtered_gt.append(batch_result)
    return filtered_gt