#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 11:48:47 2025

@author: arash
"""
import torch
import torch.nn.functional as F
import math

def local_maximum_3d(
    objectness: torch.Tensor,
    kernel: int = 3,
    score_thr: float = 0.3,
    topk: int | None = None,
):
    """
    Simple 3D local-maxima NMS on an objectness heatmap.

    objectness: [B, 1, D, H, W] or [B, 1, W, H, D] (just be consistent downstream)
    Returns:
        indices: [N, 5]  (b, c, d, h, w)
        scores:  [N]
    """
    assert kernel % 2 == 1, "Kernel size should be odd (3, 5, ...)"

    # local max pooling
    pooled = F.max_pool3d(objectness, kernel_size=kernel, stride=1, padding=kernel // 2)
    local_max_mask = (objectness == pooled)

    # apply score threshold
    if score_thr is not None:
        local_max_mask = local_max_mask & (objectness >= score_thr)

    indices = torch.nonzero(local_max_mask, as_tuple=False)  # [N, 5]
    if indices.numel() == 0:
        empty_idx = torch.empty((0, 5), dtype=torch.long, device=objectness.device)
        empty_scores = torch.empty((0,), dtype=objectness.dtype, device=objectness.device)
        return empty_idx, empty_scores

    scores = objectness[local_max_mask]  # [N]

    # sort by score
    sorted_idx = torch.argsort(scores, descending=True)
    if topk is not None:
        sorted_idx = sorted_idx[:topk]

    indices_sorted = indices[sorted_idx]
    scores_sorted = scores[sorted_idx]

    return indices_sorted, scores_sorted

def create_prediction_line_est2d(center, dim, yaw, score, cls_id, P2, class_names):
    x, y, z = center.tolist()
    w3d, h3d, l3d = dim.tolist()
    ry = yaw.item()
    sc = score.item()

    # ---- 2D bbox: quick fake one via projecting center ----
    center_3d = torch.tensor([x, y, z, 1.0], dtype=torch.float32)
    uvd = P2 @ center_3d
    u = (uvd[0] / uvd[2]).item()
    v = (uvd[1] / uvd[2]).item()

    box_w = 20.0  # fixed 2D size in pixels (dummy)
    box_h = 20.0
    xmin = u - box_w / 2.0
    ymin = v - box_h / 2.0
    xmax = u + box_w / 2.0
    ymax = v + box_h / 2.0

    truncated = 0.0
    occluded = 0
    # alpha from x,z,ry (optional; can use 0.0 instead)
    ray = math.atan2(x, z)
    alpha = ry - ray

    # Map cls_id to KITTI type string
    kitti_type = class_names[int(cls_id)]

    line = f"{kitti_type} {truncated:.2f} {occluded:d} {alpha:.4f} " \
           f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} " \
           f"{h3d:.2f} {w3d:.2f} {l3d:.2f} " \
           f"{x:.2f} {y:.2f} {z:.2f} {ry:.4f} {sc:.4f}"
    
    
    return line

def compute_corners(dimensions, alpha):
    dtype = dimensions.dtype

    num_boxes = dimensions.shape[0]
    h, w, l = torch.split( dimensions.view(num_boxes, 1, 3), [1, 1, 1], dim=2)
    unrot = torch.cat([torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=2),
                       torch.cat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=2)], dim=1)
    alpha_r = alpha.view(num_boxes, 1)

    x_rot_vect = torch.cat([torch.cos(alpha_r), torch.sin(alpha_r)], dim=1).view(num_boxes, 2, 1)
    x_rot = (unrot * x_rot_vect).sum(dim=1, keepdim=True)

    z_rot_vect = torch.cat([-torch.sin(alpha_r), torch.cos(alpha_r)], dim=1).view(num_boxes, 2, 1)
    z_rot = (unrot * z_rot_vect).sum(dim=1, keepdim=True)

    zeros = torch.zeros((num_boxes, 1, 1), dtype=dtype)
    if dimensions.is_cuda:
        zeros = zeros.cuda()
    y_rot = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=2)

    corners_rot = torch.cat([x_rot, y_rot, z_rot], dim=1)
    return corners_rot

def create_prediction_line_acc2d(center, dim, yaw, score, cls_id, P2, class_names):
    x, y, z = center.tolist()
    w3d, h3d, l3d = dim.tolist()
    ry = yaw.item()
    sc = score.item()

    # build tensor for corners: compute_corners expects dims=(h,w,l)
    dims_hwl = torch.tensor([[h3d, w3d, l3d]], dtype=torch.float32)
    ry_t     = torch.tensor([ry], dtype=torch.float32)
    
    # compute corners relative to bottom centre (DSGN anchor)
    corners = compute_corners(dims_hwl.unsqueeze(0), ry_t)[0].T  # (8,3)
    
    # shift to actual centre; compute_corners uses bottom anchor (top y=0, bottom y=–h)
    # so adding y + h/2 moves the box so that its centre is at (x,y,z)
    corners[:, 0] += x
    corners[:, 1] += y + h3d / 2.0
    corners[:, 2] += z

    corners_h = torch.cat([corners, torch.ones((8, 1), dtype=torch.float32)], dim=1)  # homogeneous coords
    pts_2d    = (P2 @ corners_h.T).T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    # 2D bounding box from projected corners
    xmin = pts_2d[:, 0].min().item()
    ymin = pts_2d[:, 1].min().item()
    xmax = pts_2d[:, 0].max().item()
    ymax = pts_2d[:, 1].max().item()

    # observation angle alpha: difference between box yaw and camera ray
    alpha = ry - math.atan2(x, z)

    # map class id to class name (KITTI wants type strings)
    kitti_type = class_names[int(cls_id)]

    # write KITTI-format detection line: type, trunc, occlusion, alpha, bbox2D, h,w,l, centre, ry, score
    line = f"{kitti_type} 0.00 0 {alpha:.4f} " \
           f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} " \
           f"{h3d:.2f} {w3d:.2f} {l3d:.2f} " \
           f"{x:.2f} {y:.2f} {z:.2f} {ry:.4f} {sc:.4f}"

    return line
    

# TODO
def nms_python():
    raise NotImplementedError()