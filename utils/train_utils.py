#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn
import torch

def assign_gt_to_voxels(grid, gtl, res_w, res_h, res_d, ignore_class_id=-1):
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
    W, H, D = res_w, res_h, res_d
    device = grid.device
    assignments = torch.full((W, H, D), fill_value=-1, dtype=torch.long, device=device)

    for i, gt in enumerate(gtl):
        cls = gt["category"]
        h, w, l, cx, cy, cz, yaw = gt["bbox3d"]

        # Compute voxel indices inside the box (simplified AABB logic)
        x_min, x_max = cx - w/2, cx + w/2
        y_min, y_max = cy - h/2, cy + h/2
        z_min, z_max = cz - l/2, cz + l/2

        xs, ys, zs = grid[0], grid[1], grid[2]
        inside = (xs >= x_min) & (xs <= x_max) & \
                 (ys >= y_min) & (ys <= y_max) & \
                 (zs >= z_min) & (zs <= z_max)

        if cls == ignore_class_id:
            assignments[inside] = -2  # Ignored class (e.g. Tram)
        else:
            assignments[inside] = i  # Assign voxel to this gt

    return assignments




class loss_3d(nn.Module):
    def __init__(self, cfg):
        super(loss_3d, self).__init__()
        self.cfg = cfg
        self.num_c = self.cfg.model.num_class
        self.loss_weights = self.cfg.loss.weght
        
        
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
        
        # first part: a function to match each gt detection to a voxel
        
        
        # Second part: objectness loss
        
        
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