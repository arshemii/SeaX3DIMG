#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:01:20 2025

@author: arash
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import glob
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import collate_fn
from utils.kitti_sx3d import kitti_sx3d
from SX3DIMG import get_SX3D_model
from utils.grid_generator import GridGenerator
import utils.eval_utils as eu


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


def output_PP(prediction, bev_grid, score_th=0.65, num_classes=4):
    """
    Convert raw prediction map into structured detections per sample.

    Args:
        prediction: Tensor [B, out_ch, W, H]
        bev_grid: Tensor [W, H, 2] with voxel centers (x, y)
        score_th: float, threshold for objectness
        num_classes: int, number of classes

    Returns:
        results: list of length B
            Each element is a list of dictionaries, each dictionary has:
                {
                    "class_id": int,
                    "class_conf": float,
                    "objectness": float,
                    "center": (x, y, z),
                    "dims": (w, l),
                    "yaw": float
                }
    """
    B, out_ch, W, H = prediction.shape
    device = prediction.device

    # Split prediction channels
    class_logits = prediction[:, :num_classes, :, :]        # [B, C, W, H]
    objectness = torch.sigmoid(prediction[:, num_classes, :, :])  # [B, W, H]
    offsets = prediction[:, num_classes + 1:num_classes + 3, :, :]  # [B, 2, W, H]
    dims = prediction[:, num_classes + 3:num_classes + 5, :, :]     # [B, 2, W, H]
    yaw = prediction[:, -1, :, :]                                   # [B, W, H]

    # Apply softmax to class logits
    class_probs = F.softmax(class_logits, dim=1)  # [B, C, W, H]
    class_conf, class_id = torch.max(class_probs, dim=1)  # [B, W, H]

    # Objectness mask
    mask = objectness > score_th

    results = []
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False)  # [N, 2]
        sample_objs = []

        if idx.numel() == 0:
            results.append([])  # no detections for this sample
            continue

        w_idx = idx[:, 0]
        h_idx = idx[:, 1]

        # Extract relevant features
        scores = objectness[b, w_idx, h_idx]        # [N]
        classes = class_id[b, w_idx, h_idx]         # [N]
        class_scores = class_conf[b, w_idx, h_idx]  # [N]
        yaw_vals = yaw[b, w_idx, h_idx]             # [N]
        offset_vals = offsets[b, :, w_idx, h_idx]   # [2, N]
        dims_vals = dims[b, :, w_idx, h_idx]        # [2, N]

        # Add offsets to voxel grid centers
        centers = bev_grid[w_idx, h_idx, :].T + offset_vals  # [2, N]
        zeros = torch.zeros((1, centers.shape[1]), device=device)

        # Build list of dicts for this sample
        for i in range(centers.shape[1]):
            obj = {
                "class_id": int(classes[i].item()),
                "class_conf": float(class_scores[i].item()),
                "objectness": float(scores[i].item()),
                "center": (
                    float(centers[0, i].item()),
                    float(centers[1, i].item()),
                ),
                "dims": (
                    float(dims_vals[0, i].item()),
                    float(dims_vals[1, i].item())
                ),
                "yaw": float(yaw_vals[i].item())
            }
            sample_objs.append(obj)

        results.append(sample_objs)

    return results

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
        
def draw_bev_heatmap(prediction, bev_grid, num_classes = 4, batch_idx=0):
    """
    Draw a BEV heatmap of objectness scores from raw prediction.
    
    Args:
        prediction: Tensor of shape [B, out_ch, W, H] (raw model output)
        bev_grid: Tensor of shape [W, H, 2] (voxel centers in BEV)
        num_classes: int, number of classes in prediction
        batch_idx: int, which batch element to visualize
    """

    # Extract objectness and apply sigmoid
    objectness = torch.sigmoid(prediction[batch_idx, num_classes, :, :])  # [W, H]

    # Convert to numpy for matplotlib
    heatmap = objectness.cpu().numpy()
    return heatmap

def draw_bev_bboxes(prediction, bev_grid, num_classes = 4, score_th=0.65, batch_idx=0):
    """
    Draw BEV bounding boxes from raw prediction.
    
    Args:
        prediction: Tensor [B, out_ch, W, H] (raw output)
        bev_grid: Tensor [W, H, 2] voxel centers
        num_classes: int, number of classes
        score_th: float, threshold on objectness
        batch_idx: int, batch index to visualize
    """
    device = prediction.device
    W, H = bev_grid.shape[:2]

    # Extract objectness, offsets, dims
    objectness = torch.sigmoid(prediction[batch_idx, num_classes, :, :])  # [W, H]
    offsets = prediction[batch_idx, num_classes + 1:num_classes + 3, :, :]  # [2, W, H]
    dims = prediction[batch_idx, num_classes + 3:num_classes + 5, :, :]      # [2, W, H]

    # Mask high-confidence locations
    mask = objectness > score_th
    if mask.sum() == 0:
        print("No objects above threshold")
        return

    idx = mask.nonzero(as_tuple=False)  # [N, 2] -> (w_idx, h_idx)
    w_idx, h_idx = idx[:, 0], idx[:, 1]

    # Compute BEV centers (x, z) using grid + offsets
    centers = bev_grid[w_idx, h_idx, :].T + offsets[:, w_idx, h_idx]  # [2, N]

    # Extract dimensions
    widths = dims[0, w_idx, h_idx]  # [N]
    lengths = dims[1, w_idx, h_idx]  # [N]

    # Plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.title(f'BEV Bounding Boxes - Batch {batch_idx}')
    plt.xlabel('X (BEV)')
    plt.ylabel('Z (BEV)')

    # Draw boxes
    for i in range(centers.shape[1]):
        cx, cz = centers[0, i].item(), centers[1, i].item()
        w, l = widths[i].item(), lengths[i].item()
        # Rectangle expects bottom-left corner, so shift
        rect = patches.Rectangle(
            (cx - w/2, cz - l/2),
            w, l,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    # Optionally overlay the objectness as scatter points
    plt.scatter(centers[0].cpu(), centers[1].cpu(), c=objectness[w_idx, h_idx].cpu(), cmap='hot', s=20)
    plt.colorbar(label='Objectness Score')
    plt.xlim(bev_grid[...,0].min().item(), bev_grid[...,0].max().item())
    plt.ylim(bev_grid[...,1].min().item(), bev_grid[...,1].max().item())
    plt.show()

def labels_to_dict(batch_labels):
    """
    Convert padded label tensor to list of dictionaries per batch element.
    
    Args:
        batch_labels: Tensor of shape [B, 18, 14], last feature indicates valid object (1) or padding (0)
    
    Returns:
        labels_dict_list: List of length B, each element is a list of dictionaries (one per object)
    """
    B, max_objects, num_features = batch_labels.shape
    labels_dict_list = []

    for b in range(B):
        objects = []
        for obj_idx in range(max_objects):
            # Only keep valid objects
            if batch_labels[b, obj_idx, -1] == 0:
                continue

            obj_features = batch_labels[b, obj_idx]  # [14]
            obj_dict = {
                'bev': obj_features[7:12],     # bbox 2D: x_min, y_min, x_max, y_max
                'category': obj_features[-2].item()  # 1
            }
            objects.append(obj_dict)
        labels_dict_list.append(objects)

    return labels_dict_list

def test_forward():
    
    cfg = check_env()
    
    dataset = kitti_sx3d(cfg)
    
    # model preparation
    checkpoint_dir = cfg.model.sx3d.checkpoint_3d
    latest_ckpt = max(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")),
        key=lambda x: int(re.search(r"checkpoint_epoch_(\d+).pth", x).group(1)))
    
    model = get_SX3D_model(cfg)
    model = model.to(cfg.device[0])
        
    checkpoint = torch.load(latest_ckpt, map_location = cfg.device[0])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    grid_obj = GridGenerator(cfg.grid_size, cfg.grid_unc, cfg.H_off) # points in cam coordinates
    grid = grid_obj.get_grid()['grid'].to(dtype=torch.float32).permute(1,2,3,0)
    bev_grid = eu.grid3d_to_grid2d(grid)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                 collate_fn=collate_fn, num_workers=cfg.num_worker)
    

    batch = next(iter(dataloader))
    
    with torch.no_grad():
        batch["left_img"] = batch["left_img"].to(cfg.device[0])
        batch["left_img_previous"] = batch["left_img_previous"].to(cfg.device[0])
        batch["right_img"] = batch["right_img"].to(cfg.device[0])
        batch["label"] = batch["label"].to(cfg.device[0])
                        
        temporal_l = model.create_memory(batch["left_img_previous"])
        outputs = model(batch["left_img"], batch["right_img"], temporal_l)[0]
        outputs = outputs.detach().cpu()
        
    results = output_PP(outputs, bev_grid, score_th=0.65, num_classes=4)
    img = tensor_to_image(batch["left_img"], cfg.data.mean[0], cfg.data.std[0])
    heatmap = draw_bev_heatmap(outputs, bev_grid)
    gt = labels_to_dict(batch["label"])
    
    
    return results, bev_grid, img, heatmap, outputs, gt
    
    
    
# plt.figure(figsize=(10, 8))
# plt.imshow(hm.T, origin='lower', cmap='hot', interpolation='nearest')  # transpose to match BEV orientation
# plt.colorbar(label='Objectness Score')
# plt.title(f'BEV Heatmap - Batch {0}')
# plt.xlabel('W-axis (BEV)')
# plt.ylabel('H-axis (BEV)')
# plt.show()




def postprocess_bev(prediction, bev_grid, score_th=0.65, iou_th=0.35, num_classes=4):
    """
    Postprocess raw BEV prediction map into structured detections.
    
    Args:
        prediction: Tensor [B, num_f, W, H]
        bev_grid: Tensor [W, H, 2]  (x, y centers of voxels)
        score_th: float, minimum score to keep
        iou_th: float, IoU threshold for NMS
        num_classes: int
    Returns:
        detections: list[list[dict]]  # [batch][objects]
    """
    B, out_ch, W, H = prediction.shape
    device = prediction.device
    detections = []

    for b in range(B):
        pred = prediction[b]

        # Split raw output
        class_logits = pred[:num_classes]                 # [C, W, H]
        obj_score = torch.sigmoid(pred[num_classes])      # [W, H]
        offsets = pred[num_classes + 1:num_classes + 3]   # [2, W, H]
        dims = pred[num_classes + 3:num_classes + 5]      # [2, W, H]
        yaw = pred[-1]                                    # [W, H]

        # Class probabilities
        class_probs = F.softmax(class_logits, dim=0)      # [C, W, H]
        class_conf, class_id = torch.max(class_probs, dim=0)

        # Combined confidence
        scores = obj_score * class_conf

        # Keep high-confidence voxels
        mask = obj_score > score_th
        if not mask.any():
            detections.append([])
            continue

        # Extract voxel indices
        idx = mask.nonzero(as_tuple=False)
        w_idx, h_idx = idx[:, 0], idx[:, 1]

        # Gather data
        s = scores[w_idx, h_idx]
        c = class_id[w_idx, h_idx]
        yaw_vals = yaw[w_idx, h_idx]
        offset_vals = offsets[:, w_idx, h_idx].T  # [N, 2]
        dims_vals = dims[:, w_idx, h_idx].T        # [N, 2]
        centers = bev_grid[w_idx, h_idx] + offset_vals  # [N, 2]

        # Stack boxes [x, y, w, l, yaw, score, class_id]
        boxes = torch.cat([centers, dims_vals, yaw_vals.unsqueeze(1), s.unsqueeze(1), c.unsqueeze(1).float()], dim=1)

        # Sort by score
        boxes = boxes[boxes[:, 5].argsort(descending=True)]

        # Simple NMS (axis-aligned IoU)
        keep = []
        for box in boxes:
            if all(iou_bev(box, kept, use_yaw=False) < iou_th for kept in keep):
                keep.append(box)
        keep = torch.stack(keep) if keep else torch.zeros((0, 7), device=device)

        # Convert to list of dicts
        dets = []
        for bx in keep:
            dets.append({
                "class_id": int(bx[6].item()),
                "score": float(bx[5].item()),
                "center": bx[:2].tolist(),
                "dims": bx[2:4].tolist(),
                "yaw": float(bx[4].item())
            })
        detections.append(dets)
    return detections


def iou_bev(box1, box2, use_yaw=False):
    """Simple BEV IoU (approx, yaw ignored unless use_yaw=True)."""
    if not use_yaw:
        x1_min, y1_min = box1[0]-box1[2]/2, box1[1]-box1[3]/2
        x1_max, y1_max = box1[0]+box1[2]/2, box1[1]+box1[3]/2
        x2_min, y2_min = box2[0]-box2[2]/2, box2[1]-box2[3]/2
        x2_max, y2_max = box2[0]+box2[2]/2, box2[1]+box2[3]/2
        inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_x * inter_y
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        return inter_area / (area1 + area2 - inter_area + 1e-6)
    else:
        # TODO: yaw-aware IoU (requires polygon intersection)
        raise NotImplementedError("Rotated IoU not yet implemented")

