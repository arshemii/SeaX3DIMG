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
    # (replace with proper 3D-corner projection if you want accurate 2D AP)
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

def nms_python():
    raise NotImplementedError()