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



def compress_tensor(tensor, number_c):
    """
    tensor: [B, 10, W, D]
    returns: [B, 8, W, D]
    
    ChatGPT generated function
    """
    B, C, W, D = tensor.shape
    assert C == 10, "Input tensor must have 10 channels, same as the model output"

    # Objectness
    objectness = tensor[:, number_c:number_c+1, :, :]  # [B,1,W,D]

    # Class predictions
    class_logits = tensor[:, 0:number_c, :, :]                # [B,4,W,D]
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
    assert grid.shape == torch.Size([W, D, 2]), f"Grid must be [{W}, {D}, 2], but received {grid.shape}"

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

def mark_far_gts(gt_labels, Z_threshold):
    """
    Filter GT labels by Z threshold, and keep only bbox_bev and category.

    """
    B, max_objects, _ = gt_labels.shape
    
    for b in range(B):
        if gt_labels[b, :, 13].sum() == 0:
             continue
        else:
             for obj_idx in range(max_objects):
                 if int(gt_labels[b, obj_idx, 13]) == 1:
                     if gt_labels[b, obj_idx, 10] <= Z_threshold:
                         gt_labels[b, obj_idx, -1] = 0.0
                         gt_labels[b, obj_idx, 12] = -3.0
    return gt_labels



def mark_nonvalid_obj(grid, gtl, oob_mask, voxel_size, ignore_class_id=-2):
    """
    Mark all ignored objects and out of FOV ones by zeroing valid flag
    """
   
    B, max_objects, _ = gtl.shape
    W, H, D = grid.shape[:3]
    device = grid.device
    
    for b in range(B):
        if gtl[b, :, 13].sum() == 0:
             continue
        else:
            for obj_idx in range(max_objects):
                if int(gtl[b, obj_idx, 13]) == 1:
                    h, w, l = gtl[b, obj_idx, 0:3]
                    cx, cy, cz = gtl[b, obj_idx, 3:6]
                    cat = int(gtl[b, obj_idx, 12])
        
                    # if object is smaller than an edge of the voxel:
                    w = torch.maximum(w, torch.tensor(voxel_size[0] * 1.02, device=w.device, dtype=w.dtype))
                    h = torch.maximum(h, torch.tensor(voxel_size[1] * 1.02, device=h.device, dtype=h.dtype))
                    l = torch.maximum(l, torch.tensor(voxel_size[2] * 1.02, device=l.device, dtype=l.dtype))
        
                    # AABB
                    x_min, x_max = cx - w / 2, cx + w / 2
                    y_min, y_max = cy - h / 2, cy + h / 2
                    z_min, z_max = cz - l / 2, cz + l / 2
        
                    # mask voxels inside this object
                    xs, ys, zs = grid[..., 0], grid[..., 1], grid[..., 2]
                    inside = (xs >= x_min) & (xs <= x_max) & \
                             (ys >= y_min) & (ys <= y_max) & \
                             (zs >= z_min) & (zs <= z_max)
        
                    assert inside.sum() != 0
        
        
                    # find voxel closest to GT center
                    voxel_coords = grid[inside]
                    dists = torch.norm(voxel_coords - gtl[b, obj_idx, 3:6].to(device), dim=1)
                    min_idx = torch.argmin(dists)
                    idx_flat = torch.nonzero(inside, as_tuple=False)[min_idx]
                    i, j, k = idx_flat.tolist()
                    
                    if not oob_mask[i, j, k]:
                        gtl[b, obj_idx, -1] = 0.0
                        gtl[b, obj_idx, 12] = -3.0
                    
                    if cat == -2:
                        gtl[b, obj_idx, -1] = 0.0

    return gtl