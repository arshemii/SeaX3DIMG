#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn
import torch

def assign_gt_to_voxels(grid, gtl, ignore_class_id=-1):
    """
    Assigns ground truth objects to the 3D grid.

    Args:
        grid: Tensor of shape [3, W, H, D] representing voxel centers (x, y, z)
        gtl: List of dicts, each with:
            - 'category': class ID or -1 for ignored objects
            - 'bbox3d': (h, w, l, cx, cy, cz, yaw)
        Returns:
            assignments: Tensor of shape [W, H, D] with:
                - -1: background
                - >= 0: index of gtl object
                - -2: ignored (like tram)
    """
    W, H, D = grid.shape[1], grid.shape[2], grid.shape[3]
    device = grid.device
    
    assignments = torch.full((W, H, D), fill_value=-1, dtype=torch.long, device=device)
    center_voxels = []

    for idx, gt in enumerate(gtl):
        cat = gt["category"]
        h, w, l, cx, cy, cz, yaw = gt["bbox3d"]

        # Compute voxel indices inside the box (simplified AABB logic)
        x_min, x_max = cx - w/2, cx + w/2
        y_min, y_max = cy - h/2, cy + h/2
        z_min, z_max = cz - l/2, cz + l/2

        xs, ys, zs = grid[0], grid[1], grid[2]
        inside = (xs >= x_min) & (xs <= x_max) & \
                 (ys >= y_min) & (ys <= y_max) & \
                 (zs >= z_min) & (zs <= z_max)

        if cat == ignore_class_id:
            assignments[inside] = -2  # Ignored class (e.g. Tram)
        else:
            assignments[inside] = idx  # Assign voxel to this gt
            
        # Find the voxel closest to GT center
        voxel_xyz = grid[:, inside].T  # [N, 3]
        gt_center = torch.tensor([cx, cy, cz], device=grid.device)
        dists = torch.norm(voxel_xyz - gt_center, dim=1)
        min_idx = torch.argmin(dists)
        idx_flat = torch.nonzero(inside, as_tuple=False)[min_idx]
        i, j, k = idx_flat.tolist()
        center_voxels.append((i, j, k, idx))  # voxel_i, voxel_j, voxel_k, gt_idx

    return assignments, center_voxels




class loss_3d(nn.Module):
    def __init__(self, cfg):
        super(loss_3d, self).__init__()
        self.cfg = cfg
        self.num_c = self.cfg.model.num_class
        self.loss_weights = self.cfg.loss.weght
        self.B = self.cfg.num_batch
        self.alpha = self.cfg.loss.alpha
        self.gamma = self.cfg.loss.gamma
        
        
    def object_conf(self, pred_obj_logits, voxel_assignments):
        """
        pred_obj_logits: [B, 1, W, H, D]
        voxel_assignments: a list of tensors with lenght = B and -1=bg, -2=ignored, >=0=object
        """
        loss = 0.0
        for b in range(self.B):
            target = (voxel_assignments[b] >= 0).float()
            valid = (voxel_assignments[b] != -2)
            pred = pred_obj_logits[b, 0][valid]
            tgt = target[valid]
    
            # Focal BCE
            bce = nn.functional.binary_cross_entropy_with_logits(pred, tgt, reduction='none')
            pt = torch.exp(-bce)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
            loss += focal_loss.mean()
        return loss / self.B

    
    def forward(self, prediction, gtl, grid):
        """
        prediction is:
            pred[:num_class] = class probabilities,
            pred[num_class] = objecness score,
            pred[num_class + 1 : num_class + 4] = offsets from voxel center,
            pred[num_class + 4 : num_class + 7] = object dimensions,
            pred[-1] = object box yaw angle
            
        gtl is is a list where for gt in gtl:
            gt['category'] = object class (zero to num_classes-1 and -1 for not important objects)
            gt['bbox3d'] = order is: h, w, l, cx, cy, cz, yaw
            
        grid is:
            the grid with x, y, z of each voxel to match prediction with gt objects
            
        *** Prediction comes like [n, out_ch, w_res, h_res, d_res]
        """
        assert self.B == prediction.shape[0]
        
        # first part: a function to match each gt detection to corresponding voxels and find which voxel is closest to the box center
        assignments = []
        c_voxels = []
        for i in range(self.B):
            ass, center_voxels = assign_gt_to_voxels(grid, gtl)  # shape: (res_w, res_h, res_d)
            assignments.append(ass)
            c_voxels.append(center_voxels)
                
        # Second part: objectness loss
        obj_conf = self.object_conf(prediction[:, self.num_c:self.num_c+1], assignments)
        
        # Third part: class loss (might be useful if focal loss is used)
        
        
        # Forth part: add bbox offset to the voxel center for bbox center position
        
        
        # Fifth part: bbox center loss
        
        
        # Sixth part: bbox dim loss
        
        
        # Seventh part: bbox orientation loss
        
        
        # Total loss: Sum of all loss considering their importance based on self.loss_weights
        
        
        """
        final_loss = {"obj_loss": objectness_loss,
                      "class_loss": class_loss,
                      "pose_loss": pose_loss,
                      "dim_loss": dim_loss,
                      "orn_loss": orn_loss,
                      "total_loss": total_loss}
        
        return final_loss
        """