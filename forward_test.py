#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:14:36 2025

@author: arash
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.data_utils import collate_fn
from utils.kitti_sx3d import kitti_sx3d
from SX3DIMG import get_SX3D_model


def check_env():
    
    from model_cong import config_generator
    cfg = config_generator()
    
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
    
        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i} name:", torch.cuda.get_device_name(i))
            print("  Total memory (GB):", round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2))
            print("  Multiprocessors (SMs):", torch.cuda.get_device_properties(i).multi_processor_count)
            print("  CUDA capability:", torch.cuda.get_device_properties(i).major, ".", torch.cuda.get_device_properties(i).minor)

    
    return cfg


def tensor_to_image(tensor, mean, std):
    """
    Convert a normalized image tensor [1, 3, H, W] back to a NumPy RGB image [H, W, 3].
    
    Args:
        tensor: torch.Tensor of shape [1, 3, H, W] or [3, H, W]
        mean: list or np.array of length 3 (RGB)
        std: list or np.array of length 3 (RGB)
    Returns:
        img_rgb: np.ndarray of shape [H, W, 3], values in [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [3, H, W]
        
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)

    img = tensor * std + mean        # denormalize
    img = torch.clamp(img, 0, 1)     # keep valid range

    img_rgb = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return img_rgb 

def test_prep():
    cfg = check_env()
    dataset = kitti_sx3d(cfg)
        
    model = get_SX3D_model(cfg)
    model = model.to(cfg.device[0])
    #checkpoint = torch.load(latest_ckpt, map_location = cfg.device[0])
    #model.load_state_dict(checkpoint['model_state'])
    model = model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                     collate_fn=collate_fn, num_workers=cfg.num_worker)
    
    print("Model outputs in order:")
    print("First tuple:")
    print("------ Objectness:        Direct output of a conv layer with one channel")
    print("------ Classification:    Direct output of a conv layer with num_c channels.")
    print("------ Center Offsets:    Output of a tanh, should be scaled.")
    print("------ Dimension:         Output of a softplus.")
    print("------ Yaw:               Output of a tanh, should be scaled.")
    print("Second tuple:")
    print("------ Disparity:         Only used in training")
    
    return model, dataloader, cfg

def a_forward(model, dataloader, cfg):

    batch = next(iter(dataloader))
    
    with torch.no_grad():
        batch["left_img"] = batch["left_img"].to(cfg.device[0])
        batch["right_img"] = batch["right_img"].to(cfg.device[0])
                        
        outputs = model(batch["left_img"], batch["right_img"])
        
    disp = outputs[1]
    obj_out = outputs[0]
    # disparity = outputs[1].detach().cpu()
        
    img = tensor_to_image(batch["left_img"], cfg.data.mean[0], cfg.data.std[0])

    gtl = batch['label']
    
    return disp, obj_out, img, gtl

def out_preparation(obj_out, cfg):
    obj, classes, offset, dims, yaw = obj_out
    
    scores = F.sigmoid(obj).to('cpu')
    
    probs = F.softmax(classes, dim=1).to('cpu')
    max_probs, max_indices = torch.max(probs, dim=1)
    
    grid = cfg.grid_forward[0].permute(3, 0, 1, 2).unsqueeze(0)
    centers = offset.to('cpu')
    scales = torch.tensor(cfg.grid_unc, dtype=centers.dtype, device=centers.device).view(1, 3, 1, 1, 1)
    centers = centers * scales
    centers = centers + grid

    dims = dims.to('cpu')
    yaw = math.pi * yaw.to('cpu')

    return scores, max_probs, max_indices, centers, dims, yaw
    
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

def create_obj_list(indices_sorted, scores_sorted,
                    class_score, class_id,
                    centers, dimensions, orientation):
    """
    indices_sorted: [N, 5] -> (b, c, w, h, d) from nonzero on [B,1,W,H,D]
    scores_sorted:  [N]    -> objectness scores (already sorted desc)
    """

    # unpack indices
    b, _, w, h, d = indices_sorted.T  # each is [N]

    # gather scalar stuff
    cls_id = class_id[0, w, h, d].unsqueeze(0)       # [N]
    cls_sc = class_score[0, w, h, d].unsqueeze(0)    # [N]

    # gather vector stuff
    cnt   = centers[b, :, w, h, d]         # [N, 3]
    dim   = dimensions[b, :, w, h, d]      # [N, 3]
    yaw   = orientation[b, :, w, h, d]     # [N, C_yaw]

    objs = []
    N = indices_sorted.size(0)
    for i in range(N):
        objs.append({
            "voxel_idx": (int(w[i]), int(h[i]), int(d[i])),
            "obj_score": float(scores_sorted[i]),
            "class_id": int(cls_id.T[i]),
            "class_score": float(cls_sc.T[i]),
            "center": cnt[i].tolist(),          # [cx, cy, cz]
            "dimensions": dim[i].tolist(),      # [l, w, h] or whatever order you use
            "orientation": yaw[i].tolist(),     # angle or [sin, cos]
        })

    # objs is already sorted because scores_sorted / indices_sorted are sorted
    return objs

def heatmap_3d_v1(objectness_score, cfg, objectness_thr):
    obj_all = objectness_score[0, 0]  # [W, H, Z]
    assert cfg.oob_mask_valid[0].shape == obj_all.shape
    obj = obj_all * cfg.oob_mask_valid[0] 
    
    W, H, Z = obj.shape
    
    mask = obj > objectness_thr
    if mask.any():
        xs, ys, zs = torch.nonzero(mask, as_tuple=True)  # indices of active voxels
        vals = obj[mask]  # probabilities
    
        # Optional: subsample if too many points
        max_points = 20000
        if xs.numel() > max_points:
            idx = torch.randperm(xs.numel())[:max_points]
            xs, ys, zs, vals = xs[idx], ys[idx], zs[idx], vals[idx]
    
        xs = xs.cpu().numpy()
        ys = ys.cpu().numpy()
        zs = zs.cpu().numpy()
        vals = vals.cpu().numpy()
    
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        # scatter: coordinates vs values (as color)
        p = ax.scatter(xs, ys, zs, c=vals, s=5, alpha=0.7)
        fig.colorbar(p, ax=ax, label='Objectness prob')
    
        ax.set_xlabel('W (x)')
        ax.set_ylabel('H (y)')
        ax.set_zlabel('Z (depth)')
    
        ax.set_title('3D objectness scatter')
        plt.tight_layout()
        plt.show()
    else:
        print("No voxels above threshold.")
        
def heatmap_3d_v2(objectness_score, cfg, objectness_thr=0.3, max_points=20000):
    """
    Visualize 3D objectness as a scatter plot in metric space.

    objectness_score: [1, 1, W, H, Z] after sigmoid
    cfg.grid_size    : (X_size, Y_size, Z_size) in meters
    cfg.grid_unc     : (dx, dy, dz) in meters
    cfg.oob_mask_valid[0]: [W, H, Z] in-FOV mask (1/True = valid)
    cfg.H_min, cfg.H_max: vertical range in meters (with camera at ~H_off)
    """

    # [W, H, Z]
    obj_all = objectness_score[0, 0]

    # mask out-of-FOV voxels
    oob = cfg.oob_mask_valid[0].to(obj_all.device).bool()
    assert oob.shape == obj_all.shape
    obj = obj_all * oob  # outside FOV → 0
    #obj = obj_all
    W, H, Z = obj.shape
    x_size, y_size, z_size = cfg.grid_size
    dx, dy, dz = cfg.grid_unc

    # Sanity: grid_resolution should match tensor shape
    if hasattr(cfg, "grid_resolution"):
        assert cfg.grid_resolution == (W, H, Z), \
            f"grid_resolution {cfg.grid_resolution} vs tensor {obj.shape}"

    # threshold
    mask = obj > objectness_thr
    if not mask.any():
        print("No voxels above threshold.")
        return

    xs, ys, zs = torch.nonzero(mask, as_tuple=True)   # voxel indices
    vals = obj[mask]

    # Optional subsampling
    if xs.numel() > max_points:
        idx = torch.randperm(xs.numel(), device=xs.device)[:max_points]
        xs, ys, zs, vals = xs[idx], ys[idx], zs[idx], vals[idx]

    xs = xs.float()
    ys = ys.float()
    zs = zs.float()

    # --- Convert indices to metric coordinates ---
    # Center each voxel with +0.5, then map to meters

    # lateral (left/right), symmetric around 0
    # x in [-x_size/2, x_size/2]
    x_world = (xs + 0.5) * dx - x_size / 2.0

    # depth (forward), camera at z=0
    # z in [0, z_size]
    z_world = (zs + 0.5) * dz

    # height (up), using H_min / H_max with offset
    # y in [H_min, H_max]
    # assume cfg.H_min / cfg.H_max precomputed as you described
    y_world = (ys + 0.5) * dy + cfg.H_min

    x_world = x_world.cpu().numpy()
    y_world = y_world.cpu().numpy()
    z_world = z_world.cpu().numpy()
    vals = vals.cpu().numpy()

    # --- Plot ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Ground plane is XZ, height is Z axis in the plot
    p = ax.scatter(x_world, z_world, y_world,
                   c=vals, s=5, alpha=0.7)
    fig.colorbar(p, ax=ax, label='Objectness prob')

    ax.set_xlabel('X (left/right) [m]')
    ax.set_ylabel('Z (forward) [m]')
    ax.set_zlabel('Y (up) [m]')

    # Set full extents of the voxel grid, independent of active points
    ax.set_xlim(-x_size / 2.0, x_size / 2.0)      # lateral
    ax.set_ylim(0.0, z_size)                      # forward
    ax.set_zlim(cfg.H_min, cfg.H_max)             # vertical

    ax.set_title(f'3D objectness scatter (thr={objectness_thr})')
    plt.tight_layout()
    plt.show()
   

model, dataloader, cfg = test_prep()
_, out_tuple, img, GT = a_forward(model, dataloader, cfg)

objectness_score, class_score, class_id, centers, dimensions, orientation = out_preparation(out_tuple, cfg)

indices_sorted, scores_sorted = local_maximum_3d(objectness = objectness_score, kernel = 9,
                                                 score_thr = 0.3, topk = 10)

obj_list = create_obj_list(indices_sorted, scores_sorted,
                    class_score, class_id,
                    centers, dimensions, orientation)

##### --------------------------- Visualization
cv.imshow("original image", img)
heatmap_3d_v2(objectness_score, cfg, 0.20)


