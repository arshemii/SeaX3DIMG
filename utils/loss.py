#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn
import torch
import numpy as np
import math
    


def assign_gt_to_voxels(assignments, c_voxels,
                        grid, gtl, voxel_size,
                        debug, ignore_class_id=-1):
    """
    Assigns ground truth objects to the 3D grid.

    Args:
        assignments: Tensor of shape [B, W, H, D]
        c_voxels: Tensor of shape [B, 18, 4]
        
        grid: Tensor of shape [W, H, D, 3] representing voxel centers (x, y, z)
        gtl: Shape: (B, 18, 14)
        
        Returns:
            assignments: Tensor of shape [B, W, H, D] with:
                - -1: background
                - >= 0: index of gtl object
                - -2: ignored (like tram)
            c_voxels:Tensor of shape [B, 18, 4]:
                for each object (dim=1) the i, j, k of the center of the voxel related to that object
                as well as its object ID is written
    """
   
    B, max_objects, _ = gtl.shape
    W, H, D = grid.shape[:3]
    device = grid.device
    
    for b in range(B):
        if gtl[b, 0, 13] == 0:
             # no object
             continue
        else:
            for obj_idx in range(max_objects):
                if int(gtl[b, obj_idx, 13]) == 1:
                    h, w, l = gtl[b, obj_idx, 0:3]
                    cx, cy, cz = gtl[b, obj_idx, 3:6]
                    cat = int(gtl[b, obj_idx, 12])
        
                    # if object is smaller than an edge of the voxel:
                    w = max(w, voxel_size[0])
                    h = max(h, voxel_size[1])
                    l = max(l, voxel_size[2])
        
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
        
                    # assign voxel values
                    if cat == ignore_class_id:
                        assignments[b][inside] = -2
                    else:
                        assignments[b][inside] = obj_idx
        
                    # find voxel closest to GT center
                    voxel_coords = grid[inside]
                    dists = torch.norm(voxel_coords - gtl[b, obj_idx, 3:6].to(device), dim=1)
                    min_idx = torch.argmin(dists)
                    idx_flat = torch.nonzero(inside, as_tuple=False)[min_idx]
                    i, j, k = idx_flat.tolist()
                    c_voxels[b, obj_idx] = torch.tensor([i, j, k, obj_idx], device=device)

    return assignments, c_voxels


def aggregate_assignment(assignments: torch.Tensor) -> torch.Tensor:
    """
    Collapse [B, W, H, D] voxel assignments into [B, W, D] BEV map.

    Rules:
      1. If any object index (>=0) exists → pick one randomly.
      2. If mix of object and ignored (-2) → pick from object ones.
      3. If only ignored (-2) → pick one randomly from ignored.
      4. Else (all -1) → -1 background.
    """
    B, W, H, D = assignments.shape
    bev = torch.full((B, W, D), -1, dtype=assignments.dtype, device=assignments.device)

    for b in range(B):
        ass = assignments[b]  # [W, H, D]

        # Masks for presensce of an object
        obj_mask = ass >= 0
        ign_mask = ass == -2

        # pillars that have at least one object
        has_obj = obj_mask.any(dim=1)  # [W, D]
        has_ign = ign_mask.any(dim=1)  # [W, D]

        # planar coordinate of pillars with object
        xs, zs = torch.nonzero(has_obj | has_ign, as_tuple=True)

        for x, z in zip(xs.tolist(), zs.tolist()):
            pillar = ass[x, :, z]
            objs = pillar[pillar >= 0]
            if len(objs) > 0:
                idx = torch.randint(len(objs), (1,), device=ass.device)
                bev[b, x, z] = objs[idx]
            else:
                igs = pillar[pillar == -2]
                if len(igs) > 0:
                    idx = torch.randint(len(igs), (1,), device=ass.device)
                    bev[b, x, z] = igs[idx]
                else:
                    bev[b, x, z] = -1

    return bev

class loss_bev(nn.Module):
    def __init__(self, cfg, grid, oob_mask_valid):
        super(loss_bev, self).__init__()
        self.cfg = cfg
        
        self.grid = grid
        assert self.grid.shape[-1] == 3, f"Expected grid[..., 3] for (x,y,z), got shape {grid.shape}"
        
        self.grid_bev = self._grid3d_to_grid2d()
        
        self.oob_mask_valid = oob_mask_valid
        # oob_mask_valid is:
        #    shape 100, 30, 70 and where the voxel is out of boundary of image --> False otherwise, True
        
        self.num_c = self.cfg.model.num_class
        self.loss_weights = self.cfg.loss.weight
        self.alpha = self.cfg.loss.alpha
        self.gamma = self.cfg.loss.gamma
        self.beta = self.cfg.loss.beta
        self.object_threshold = self.cfg.loss.object_threshold_loss
        self.zeta = self.cfg.loss.zeta
        self.voxel_size = self.cfg.grid_unc
        self.lb = self.cfg.debug_loss  # local debug
        
    def _grid3d_to_grid2d(self):
        bev_grid = self.grid[:, 0, :, :][:, :, [0, 2]]
        # bev_grid will be 100, 70, 2
        return bev_grid
    
    def _mark_oob_dets(self, assignments, c_voxels, gtl):
        """
        Drops ground truth objects that fall outside the camera view.
    
        Args:
            assignments: Tensor [B, W, H, D] with voxel→object indices
            c_voxels: Tensor [B, 18, 4] (i, j, k, obj_idx)
            gtl: Tensor [B, 18, 14] (object parameters)
    
        Returns:
            All input arguments will be updated
        """
        B, max_objects, _ = gtl.shape
        
        for b in range(B):
            # No object condition:
            if gtl[b, :, 13].sum() == 0:
                continue
            
            for i, j, k, gt_idx in c_voxels[b]:
                if gt_idx >= 0:
                    if not self.oob_mask_valid[i, j, k]:
                        # i, j, k is out of the boundary
                        assignments[b][assignments[b] == gt_idx] = -1
                        c_voxels[b, c_voxels[b, :, 3] == gt_idx] = -1
                        gtl[b, gt_idx, -1] = 0.0
                    
        return assignments, c_voxels, gtl
    
    def object_conf_loss(self, pred_obj_logits, voxel_assignments):
        """
        pred_obj_logits: [B, 1, W, D]
        voxel_assignments: a tensor of shape B, W, H, D
        """
        loss = []
                
        for b in range(self.B):
            target = (voxel_assignments[b] >= 0).float()
            valid = (voxel_assignments[b] != -2)
            pred = pred_obj_logits[b, 0][valid]
            tgt = target[valid]
            
            if pred.numel() == 0:
                continue  # skip this batch if no valid voxels
        
            # Focal BCE
            bce = nn.functional.binary_cross_entropy_with_logits(pred, tgt, reduction='none')
            pt = torch.exp(-bce)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
            loss.append(focal_loss.mean())
    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_obj_logits.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()

    def classification_loss(self, pred_cls_logits, pred_obj_logits, assignments, gtl):
        """
        pred_cls_logits: [B, num_classes, W, D]
        pred_obj_logits: [B, 1, W, D]
        assignments: tensor of shape B, W, H, D
        gtl: tensor of shape B, 18, 14
        """
        
        # TODO
        # TODO
        # TODO
        # TODO
        
        loss = []
        num_classes = pred_cls_logits.shape[1]
    
        for b in range(self.B):
            valid_mask = (assignments[b] >= 0)
            bg_mask = (assignments[b] == -1)
    
            # ---- (1) CLASS LOSS for object voxels ----
            if valid_mask.any():
                tg_classes = gtl[b, :, 12]
                present_mask = gtl[b, :, 13] == 1
                tg_classes = tg_classes[present_mask]
                tg_classes[tg_classes == -1] = -100  # ignore invalid
                pred_voxels = pred_cls_logits[b].permute(1, 2, 0)[valid_mask]
                
                ce = F.cross_entropy(pred_voxels, tg_classes, reduction='none', ignore_index=-100)
                pt = torch.exp(-ce)
                focal = self.alpha * (1 - pt) ** self.gamma * ce
                loss.append(focal.mean())
    
            # ---- (2) BACKGROUND CONSISTENCY LOSS ----
            if bg_mask.any():
                high_obj_mask = torch.sigmoid(pred_obj_logits[b, 0]) > self.object_threshold
                bad_mask = bg_mask & high_obj_mask
                if bad_mask.any():
                    pred_bg_voxels = pred_cls_logits[b].permute(1, 2, 0)[bad_mask]
                    pred_probs = F.softmax(pred_bg_voxels, dim=-1)
                    target_probs = torch.full_like(pred_probs, 1.0 / num_classes)
                    kl_loss = F.kl_div(pred_probs.log(), target_probs, reduction='batchmean')
                    loss.append(self.zeta * kl_loss)  # small weight for regularization
    
        if not loss:
            return torch.tensor(0.0, device=pred_cls_logits.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
        
    def center_loss(self, pred_offsets, assignments, gtl, grid):
        """
        pred_offsets: [B, 2, W, D]
        grid: [W, D, 2]
        
        """
        # if assignments[0].device != oob_mask_valid.device:
        #     oob_mask_valid = oob_mask_valid.to(assignments[0].device)
        # TODO: check do we need again to use valid mask?
        
        loss = []
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
                # Mask for voxels inside image boundaries and for objects
                # TODO: if we need val;id mask, must be added here to logic
                valid_mask = (assignments[b] >= 0) & (assignments[b] != -2)  # [W, D]
                if valid_mask.sum() == 0:
                    continue
            
                # Only valid voxels
                i, k = torch.nonzero(valid_mask, as_tuple=True)  # each is a (N,) tensor
            
                # Finding gt_idx of valid voxels
                gt_indices = assignments[b][i, k] # is a (N,) tensor
            
                # Centers of valid voxels
                voxel_centers = grid[i, k, :]  # (N, 2)
            
                # Predicted offsets
                pred_offsets_valid = pred_offsets[b][:, i, k].permute(1, 0)  # (N, 2)
            
                # Predicted centers
                pred_obj_centers = voxel_centers.to(pred_offsets.device) + pred_offsets_valid  # (N, 2)
            
                # Corresponding gt centers for each valid voxel
                gt_centers = torch.stack([
                    gtl[b][idx]['bbox_bev'][2:4].to(pred_offsets.device) for idx in gt_indices], dim=0)  # (N, 2)
            
                # Compute L1 loss per voxel
                l1 = nn.functional.smooth_l1_loss(pred_obj_centers, gt_centers, reduction='none', beta = self.beta)  # (N, 2)
                l1 = l1.mean(dim=1)  # (N,)
            
                loss.append(l1)
        
        # del l1, gt_centers, voxel_centers, valid_mask
        
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_offsets.device, requires_grad=True)
        else:
            return torch.cat(loss).mean()
        
        
    def dimension_loss(self, pred_dims, center_voxels, gtl):
        """
        pred_dims: [B, 2, W, D]
        """
        
        loss = []
        
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
                for (i, j, k, gt_idx) in center_voxels[b]:
                    pred = pred_dims[b, :, i, k]
                    gt = gtl[b][gt_idx]['bbox_bev'][0:2].to(pred.device)
                    loss.append(nn.functional.l1_loss(pred, gt))
                    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_dims.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
        
    def yaw_loss(self, pred_yaw, center_voxels, gtl):
        """
        pred_yaw: [B, 1, W, D]
        """
        loss = []
        
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
                for (i, j, k, gt_idx) in center_voxels[b]:
                    pred = pred_yaw[b, 0, i, k]
                    gt = gtl[b][gt_idx]['bbox_bev'][4].to(pred.device)
                    
                    # minimal angle difference (in radians) in range [-π, π]
                    diff = (pred - gt + math.pi) % (2 * math.pi) - math.pi
                    
                    loss.append(nn.functional.smooth_l1_loss(diff, torch.tensor(0.0, device=pred_yaw.device)))

        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_yaw.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()

    def forward(self, prediction, gtl):
        """
        prediction is:
            pred[:num_class] = class probabilities,
            pred[num_class] = objecness score,
            pred[num_class + 1 : num_class + 3] = offsets from voxel center,
            pred[num_class + 3 : num_class + 5] = object dimensions,
            pred[-1] = object box yaw angle
            *** Prediction comes like [n, out_ch, w_res, d_res]
            
        gtl is is a Tensor:
            Shape: (B, 18, 14) where:
                B is batch size
                18 is maximum object per instance
                14 = 7 for 3d box, 5 for bev vox, 1 for category, and 1 for valid obj
            3d box order is: h, w, l, cx, cy, cz, yaw
            box bev order is w, l, cx, cz, yaw
            object class (zero to num_classes-1 and -1 for not important objects)

        """
        
        assert len(prediction) == len(gtl), "Values are not all equal, batch size mismatch with prediction"
        assert prediction.shape == (len(gtl), 10, self.grid.shape[0], self.grid.shape[2]), \
            f"Prediction has a wrong shape, expected shape is: [n, out_ch, w_res, h_res, d_res], received: {prediction.shape}"
            
        assert gtl.shape == (len(prediction), 18, 14), \
            f"Prediction has a wrong shape, expected shape is: [n, out_ch, w_res, h_res, d_res], received: {prediction.shape}"
        
        self.B = len(prediction)
        if self.B == 0:
            device = prediction.device
            return {k: torch.tensor(0.0, device=device) for k in ['obj_conf', 'cls_loss', 'center_loss', 'dim_loss', 'yaw_angle_loss', 'total']}
        
        self.loss = {}
        
        
        assignments = torch.full((self.B, self.grid.shape[0], self.grid.shape[1], self.grid.shape[2]),
                                      fill_value=-1, dtype=torch.long, device=self.grid.device)
        
        c_voxels = torch.full((self.B, 18, 4), fill_value=-1, dtype=torch.long, device=self.grid.device)
        
        assignments, c_voxels = assign_gt_to_voxels(assignments, c_voxels,
                                                              self.grid, gtl, self.voxel_size, self.lb)
        
        
        # Removing out of the bound detections from ground truth
        assignments, c_voxels, gtl = self._mark_oob_dets(assignments, c_voxels, gtl)
        """
        result will be:
            1. assignment has no more voxels assigned to oob objects
            2. c_voxels has all objects, but the gt index of oob ones is -1
            3. gtl has all obejcts but the valid flag of oob ones is zzero now
        """
                    
        assignments_bev = aggregate_assignment(assignments)
        del assignments
        
        # Objectness loss
        self.loss['obj_conf'] = self.object_conf_loss(prediction[:, self.num_c:self.num_c+1], assignments_bev)

        # class loss (might be useful if focal loss is used)
        self.loss['cls_loss'] = self.classification_loss(prediction[:, 0:self.num_c], assignments_bev, gtl)
        
        # bbox center loss
        self.loss['center_loss'] = self.center_loss(prediction[:, self.num_c+1 : self.num_c+3],
                                       assignments_bev, gtl, self.grid_bev)

        # bbox dim loss
        self.loss['dim_loss'] = self.dimension_loss(prediction[:, self.num_c+3: self.num_c+5],
                                       c_voxels, gtl)
        
        # yaw angle loss
        # Any normalization on each term? No, just weights
        assert prediction[:, self.num_c+5:].shape[1] == 1
        self.loss['yaw_angle_loss'] = self.yaw_loss(prediction[:, self.num_c+5:], c_voxels, gtl)

        # Total loss: Sum of all loss considering their importance based on self.loss_weights
        self.loss['total'] = self.loss_weights[0]*self.loss['obj_conf'] + \
                                self.loss_weights[1]*self.loss['cls_loss'] + \
                                self.loss_weights[2]*self.loss['center_loss'] + \
                                self.loss_weights[3]*self.loss['dim_loss'] + \
                                self.loss_weights[4]*self.loss['yaw_angle_loss']
                                
        if self.lb:
            print("-------------------- loss forward ended--------------------")
            
        return self.loss
    
