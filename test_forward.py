#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:01:20 2025

@author: arash
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2 as cv
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
        



def test_forward(cfg, dataloader, model):

    #import random

    batch = next(iter(dataloader))

    
    with torch.no_grad():
        batch["left_img"] = batch["left_img"].to(cfg.device[0])
        #batch["left_img_previous"] = batch["left_img_previous"].to(cfg.device[0])
        batch["right_img"] = batch["right_img"].to(cfg.device[0])
        #batch["right_img_previous"] = batch["right_img_previous"].to(cfg.device[0])
                        
        #temporal_l = model(batch["left_img_previous"], batch["right_img_previous"], None, True)
        outputs = model(batch["left_img"], batch["right_img"])
        
    obj_out = outputs[0]
    # disparity = outputs[1].detach().cpu()
        
    img = tensor_to_image(batch["left_img"], cfg.data.mean[0], cfg.data.std[0])
    
    ids = batch["id"]
    gtl = batch['label']
    
    return obj_out, img, ids, gtl

def local_maximum(objectness: torch.Tensor, kernel: int = 3):

    assert kernel % 2 == 1, "Kernel size should be odd (3, 5, etc.)"

    # Compute local maxima
    pooled = F.max_pool3d(objectness, kernel_size=kernel, stride=1, padding=kernel // 2)
    local_max_mask = (objectness == pooled)


    # Get indices of local maxima
    indices = torch.nonzero(local_max_mask, as_tuple=False)  # [N, 5]
    
    if indices.numel() == 0:
        # no local maxima found
        return torch.empty((0, 5), dtype=torch.long), torch.empty((0,), dtype=objectness.dtype)

    # Extract corresponding scores
    scores = objectness[local_max_mask]  # [N]

    # Sort by descending score
    sorted_idx = torch.argsort(scores, descending=True)
    indices_sorted = indices[sorted_idx]

    return indices_sorted

def prepare_opt_pred(gt_boxes2):

    pred_opt = gt_boxes2.clone()
    #centers
    pred_opt[0, 2] = pred_opt[0, 2] + 0.17
    pred_opt[0, 3] = pred_opt[0, 3] - 0.12
    
    pred_opt[1, 2] = pred_opt[1, 2] + 0.88
    pred_opt[1, 3] = pred_opt[1, 3] - 0.92
    
    pred_opt[2, 2] = pred_opt[2, 2] - 2.97
    pred_opt[2, 3] = pred_opt[2, 3] + 3.55
    
    #dims
    pred_opt[0, 0] = pred_opt[0, 0] - 0.07
    pred_opt[0, 1] = pred_opt[0, 1] + 0.33
    
    pred_opt[1, 0] = pred_opt[1, 0] + 0.98
    pred_opt[1, 1] = pred_opt[1, 1] + 0.52
    
    pred_opt[2, 0] = pred_opt[2, 0] - 0.97
    pred_opt[2, 1] = pred_opt[2, 1] - 0.55
    
    #yaw
    pred_opt[0, 4] = pred_opt[0, 0] - 0.27
    
    pred_opt[1, 4] = pred_opt[1, 0] + 0.55
    
    pred_opt[2, 4] = pred_opt[2, 0] - 0.44
    
    return pred_opt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

def draw_bboxes(W, Z, boxes, title="", voxel_size=(0.62, 0.70)):
    """
    Interactive top-down visualization of 2D bounding boxes (bird's-eye view).
    
    Coordinate system:
        - Origin: bottom-center (vehicle position)
        - W axis (width): rightward (cw > 0 → right, cw < 0 → left)
        - Z axis (depth): upward (cz increases forward)
        - Each box = [bw, bz, cw, cz, yaw]
          where (bw, bz) = width/depth in meters,
                (cw, cz) = box center in meters,
                yaw = rotation in radians.
    """
    boxes = np.asarray(boxes)
    vw, vz = voxel_size  # voxel size in meters per cell

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(0, Z)

    # --- Convert from meters to voxel coordinates ---
    def meters_to_voxel_coords(cw, cz):
        x_vox = W / 2 + cw / vw   # origin at center horizontally
        z_vox = cz / vz           # origin at bottom vertically
        return x_vox, z_vox

    # --- Draw bounding boxes ---
    for box in boxes:
        bw, bz, cw, cz, yaw = box
        cx_vox, cz_vox = meters_to_voxel_coords(cw, cz)
        bw_vox, bz_vox = bw / vw, bz / vz

        rect = patches.Rectangle(
            (cx_vox - bw_vox / 2, cz_vox - bz_vox / 2),
            bw_vox, bz_vox,
            angle=np.degrees(-yaw),
            linewidth=1.5,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

    # --- Show vehicle origin ---
    ax.plot(W / 2, 0, 'ro', markersize=6, label="Vehicle (origin)")

    # --- Add zoom slider ---
    ax_zoom = plt.axes([0.25, 0.05, 0.5, 0.03])
    zoom_slider = Slider(ax_zoom, 'Zoom', 0.5, 4.0, valinit=1.0)

    def update_zoom(val):
        scale = zoom_slider.val
        ax.set_xlim(W / 2 - (W / 4) / scale, W / 2 + (W / 4) / scale)
        ax.set_ylim(0, (Z / 2) / scale)
        fig.canvas.draw_idle()

    zoom_slider.on_changed(update_zoom)

    # --- Axis labels (in meters) ---
    tick_spacing_w = 5  # meters
    tick_spacing_z = 5  # meters

    w_ticks_m = np.arange(-W/2*vw, W/2*vw + tick_spacing_w, tick_spacing_w)
    z_ticks_m = np.arange(0, Z*vz + tick_spacing_z, tick_spacing_z)

    ax.set_xticks(W/2 + w_ticks_m / vw)
    ax.set_xticklabels([f"{m:.0f}" for m in w_ticks_m])
    ax.set_yticks(z_ticks_m / vz)
    ax.set_yticklabels([f"{m:.0f}" for m in z_ticks_m])

    ax.set_xlabel("Lateral position cw (m)")
    ax.set_ylabel("Forward position cz (m)")

    ax.legend(loc="upper right")
    plt.show()
    
    
#######################################

cfg = check_env()
dataset = kitti_sx3d(cfg)

latest_ckpt = './checkpoints_exp/staged_training/fullhead.pth'
    
model = get_SX3D_model(cfg)
model = model.to(cfg.device[0])
checkpoint = torch.load(latest_ckpt, map_location = cfg.device[0])
model.load_state_dict(checkpoint['model_state'])
model = model.eval()

dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                 collate_fn=collate_fn, num_workers=cfg.num_worker)

output, img, ids, gtl = test_forward(cfg, dataloader, model)
obj, classes, offset, dims, yaw = output

scores = F.sigmoid(obj).to('cpu')
probs = F.softmax(classes, dim=1).to('cpu')
max_probs, max_indices = torch.max(probs, dim=1)
grid = cfg.grid_forward[0].permute(3, 0, 1, 2).unsqueeze(0)
centers = offset.to('cpu')
factors_tensor = torch.tensor(cfg.grid_unc, dtype=centers.dtype, device=centers.device).view(1, 3, 1, 1, 1)
centers = centers + grid

dims = dims.to('cpu')

import math
yaw = math.pi * yaw.to('cpu')

indices = local_maximum(obj, cfg.test.local_maxima_kernel).to('cpu')
b, c, w, h, z = tuple(indices.T)
values = scores[b, c, w, h, z]  # shape [N]

value_mask = values > 0.30

K = int(value_mask.sum())

objects = []

pred_boxes = torch.zeros(K, 5)

for i in range(K):
    obj = {}
    obj["class"] = max_indices[0, w[i], h[i], z[i]].item()
    obj["score"] = max_indices[0, w[i], h[i], z[i]].item() * values[i].item()
    obj["center"] = centers[0, : ,w[i], h[i], z[i]].numpy()
    obj['dimension'] = dims[0, : ,w[i], h[i], z[i]].numpy()
    obj['yaw'] = yaw[0, : ,w[i], h[i], z[i]].item()
    
    pred_boxes[i, 0] = dims[0, : ,w[i], h[i], z[i]][0]
    pred_boxes[i, 1] = dims[0, : ,w[i], h[i], z[i]][2]
    pred_boxes[i, 2] = centers[0, : ,w[i], h[i], z[i]][0]
    pred_boxes[i, 3] = centers[0, : ,w[i], h[i], z[i]][2]
    pred_boxes[i, 4] = obj['yaw']
    
    objects.append(obj)


realistic_mask = (pred_boxes[:, 0] > 0.2) & (pred_boxes[:, 1] > 0.2)
realistic_objects_pred = pred_boxes[realistic_mask]


gt_boxes = torch.zeros(18, 5)

gt_boxes[:, 0] = gtl[0][:, 0]
gt_boxes[:, 1] = gtl[0][:, 2]
gt_boxes[:, 2] = gtl[0][:, 3]
gt_boxes[:, 3] = gtl[0][:, 5]
gt_boxes[:, 4] = gtl[0][:, 6]


gt_boxes2 = gt_boxes[:1, :]
pred_box_ref = pred_boxes[3:, :]
pred_opt = prepare_opt_pred(gt_boxes2)





draw_bboxes(68, 128, gt_boxes2, "GT")
draw_bboxes(68, 128, realistic_objects_pred, "PRED")
cv.imshow("original image", img)