#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn
import torch
import math
    


def assign_gt_to_voxels(assignments, c_voxels, grid, gtl, voxel_size, ignore_class_id=-2):
    """
    Assigns ground truth objects to the 3D grid.

    Args:
        assignments: Tensor of shape [B, W, H, D] all filled by -1
        c_voxels: Tensor of shape [B, 18, 4] all filled by -1
        grid: Tensor of shape [W, H, D, 3] representing voxel centers (x, y, z)
        gtl: Shape: (B, 18, 14)
        
    Returns:
        assignments: Tensor of shape [B, W, H, D] with:   [-1: background | >= 0: index of gtl object | 2: ignored]
        c_voxels:Tensor of shape [B, 18, 4]:
            for each object, the center voxel closest to the object center is defined. index of gt_idx is the same as assignment
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
                    
                    if cat == ignore_class_id:
                        c_voxels[b, obj_idx] = torch.tensor([i, j, k, -2], device=device)
                    else:
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
      5. Index -3 has priority to all. even to background
    """
    B, W, H, D = assignments.shape
    bev = torch.full((B, W, D), -1, dtype=assignments.dtype, device=assignments.device)

    for b in range(B):
        ass = assignments[b]

        obj_mask = ass >= 0
        ign_mask = ass == -2
        blind_mask = ass == -3

        has_obj = obj_mask.any(dim=1)
        has_ign = ign_mask.any(dim=1)
        
        xs, zs = torch.nonzero(has_obj | has_ign, as_tuple=True)
        for x, z in zip(xs.tolist(), zs.tolist()):
            pillar = ass[x, :, z]
            objs = pillar[pillar >= 0]
            if len(objs) > 0:
                idx = torch.randint(0, len(objs), (1,), device=ass.device)
                bev[b, x, z] = objs[idx]
            else:
                igs = pillar[pillar == -2]
                if len(igs) > 0:
                    idx = torch.randint(0, len(igs), (1,), device=ass.device)
                    bev[b, x, z] = igs[idx]
        
        has_blind = blind_mask.any(dim=1)
        bev[b][has_blind] = -3
    return bev

class loss_bev(nn.Module):
    def __init__(self, cfg, grid, oob_mask_valid):
        super(loss_bev, self).__init__()
        self.cfg = cfg
        
        self.grid = grid
        assert self.grid.shape[-1] == 3, f"Expected grid[..., 3] for (x,y,z), got shape {grid.shape}"
        
        self.grid_bev = self._grid3d_to_grid2d()
        
        self.oob_mask_valid = oob_mask_valid
        # oob_mask_valid: [100, 30, 70], voxel is out of boundary of image --> False otherwise, True
        
        self.num_c = self.cfg.model.num_class
        self.loss_weights = self.cfg.loss.weight
        self.alpha = self.cfg.loss.alpha
        self.gamma = self.cfg.loss.gamma
        self.beta = self.cfg.loss.beta
        self.object_threshold = self.cfg.loss.object_threshold_loss
        self.zeta = self.cfg.loss.zeta
        self.voxel_size = self.cfg.grid_unc
        
    def _grid3d_to_grid2d(self):
        bev_grid = self.grid[:, 0, :, :][:, :, [0, 2]]
        # bev_grid will be 100, 70, 2
        return bev_grid
    
    def _mark_oo_FOV(self, assignments, c_voxels, gtl):
        """
        mark ground truth objects that fall outside the camera view.
        Args:
            assignments: [B, W, H, D]
            c_voxels:    [B, 18, 4] (i, j, k, obj_idx)
            gtl:         [B, 18, 14]
        Returns:
            All input arguments will be updated
        """
        _, max_objects, _ = gtl.shape
        
        for b in range(self.B):
            assignments[b][~self.oob_mask_valid] = -3
            
            if gtl[b, :, 13].sum() == 0:
                continue
            
            for m in range(max_objects):
                i, j, k, gt_idx = c_voxels[b, m, 0], c_voxels[b, m, 1], c_voxels[b, m, 2], c_voxels[b, m, 3]
                if gt_idx >= 0:
                    assert i >= 0 and j >= 0 and k >= 0
                    # ignored object not necessary to check
                    if not self.oob_mask_valid[i, j, k]:
                        # i, j, k is out of the boundary
                        c_voxels[b, m, -1] = -3
                        gtl[b, m, -1] = 0.0
                        gtl[b, m, 12] = -3.0
                    
        return assignments, c_voxels, gtl
    
    def object_conf_loss(self, pred_obj_logits, assignments):
        """
        Calculate objectness loss only [assignments = -3 is not included]
        pred_obj_logits: [B, 1, W, D]
        assignments: [B, W, D]
        """
        loss = []
                
        for b in range(self.B):
            mask = (assignments[b] >= -1)
            
            if not mask.any():
                # cannot happen, but for safety!
                continue

            pred = pred_obj_logits[b, 0][mask]                # shape: [Number of interested objects voxels and background]
            tgt = (assignments[b][mask] >= 0).float()         # 1 for object voxels and 0 for bg voxels
        
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
        assignments:     [B, W, D]
        gtl:             [B, 18, 14]
        ** gtl --> for validity flag in gtl[:, :, 13]:
                        if flag is 1 --> valid
                        if flag is zero -->
                                category in [:, :, 12] is -2 --> ignored object
                                category in [:, :, 12] is -3 --> Out of FOV objects
        """
        
        loss = []
        num_classes = pred_cls_logits.shape[1]
    
        for b in range(self.B):
            obj_mask = (assignments[b] >= 0)
            bg_mask = (assignments[b] == -1)
            
            # Loss in valid voxels
            if obj_mask.any():
                # Prepare ground truth class labels
                tg_classes = gtl[b, :, 12].clone()            # [18]
                tg_classes[gtl[b, :, 13] == 0] = -100         # invalid (gtl tensor zero padding) or OoB
                
                voxel_obj_indices = assignments[b][obj_mask].long()         # [N]
                target_tensor = tg_classes[voxel_obj_indices].long()        # [N]
                
                pred_voxels = pred_cls_logits[b].permute(1, 2, 0)[obj_mask]  # [N, num_classes]
                
                ce = nn.functional.cross_entropy(pred_voxels, target_tensor, reduction='none', ignore_index=-100)
                pt = torch.exp(-ce)
                focal = self.alpha * (1 - pt) ** self.gamma * ce
                loss.append(focal.mean())
    
            # loss if the voxel is not valid but it has high objectness score
            if bg_mask.any():
                high_obj_mask = torch.sigmoid(pred_obj_logits[b, 0]) > self.object_threshold
                bad_mask = bg_mask & high_obj_mask
                if bad_mask.any():
                    pred_bg_voxels = pred_cls_logits[b].permute(1, 2, 0)[bad_mask]
                    pred_probs = nn.functional.softmax(pred_bg_voxels, dim=-1)
                    target_probs = torch.full_like(pred_probs, 1.0 / num_classes)
                    kl_loss = nn.functional.kl_div(pred_probs.log(), target_probs, reduction='batchmean')
                    loss.append(self.zeta * kl_loss)  # small weight for regularization
    
        if not loss:
            return torch.tensor(0.0, device=pred_cls_logits.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
        
    def center_loss(self, pred_offsets, assignments, gtl, grid):
        """
        The loss for center is calculated for each voxel that is assigned to a valid object
        
        pred_offsets:   [B, 2, W, D]
        assignments:    [B, W, D]
        gtl:            [B, 18, 14] --> gtl[b, :, 9:11] are (cx_off, cz_off)
        grid:           [W, D, 2] -> (x, z)
        """
        loss = []
    
        for b in range(self.B):
            # Only compute if there are valid objects
            valid_mask = (assignments[b] >= 0)
            if valid_mask.sum() == 0:
                continue
    
            i, k = torch.nonzero(valid_mask, as_tuple=True)  # (N,)
            gt_indices = assignments[b][i, k].long()  # (N,)
    
            # all voxel centers that are assigned to a gt index
            voxel_centers = grid[i, k, :].to(pred_offsets.device)   # (N, 2)
            # predicted offsets for valid voxels
            pred_offsets_valid = pred_offsets[b][:, i, k].permute(1, 0)  # (N, 2)
            # predicted centers
            pred_obj_centers = voxel_centers.to(pred_offsets.device) + pred_offsets_valid  # (N, 2)
    
            # ground truth absolute centers
            gt_centers = gtl[b, gt_indices, 9:11].to(pred_offsets.device)
    
            # smooth L1 loss per voxel
            l1 = nn.functional.smooth_l1_loss(pred_obj_centers, gt_centers, reduction='none', beta=self.beta)
            l1 = l1.mean(dim=1)  # (N,)
    
            loss.append(l1.mean())
    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_offsets.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
            
        
    def dimension_loss(self, pred_dims, center_voxels, gtl):
        """
        Calculates loss for only and only for voxels that are closest to the object center (Dimension is constant for all voxels of an object)
        
        pred_dims:          [B, 2, W, D] --> (w, l)
        center_voxels:      [B, 18, 4] --> (i, j, k, gt_idx)
        gtl:                [B, 18, 14] --> gtl[b, :, 7:9] are (w, l)
        """
        loss = []
        
        for b in range(self.B):
            cv = center_voxels[b]  # [18, 4]
            gt_indices = cv[:, 3].long()  # [18]
            valid_mask = (gt_indices >= 0) & (gtl[b, gt_indices.clamp_min(0), 13] == 1)
            
            if not valid_mask.any():
                continue
    
            # Get valid entries
            i = cv[valid_mask, 0].long()
            k = cv[valid_mask, 2].long()
            gt_idx_valid = gt_indices[valid_mask]
    
            # Predictions for corresponding voxels
            pred = pred_dims[b, :, i, k].permute(1, 0)  # [N_valid, 2]
            gt = gtl[b, gt_idx_valid, 7:9].to(pred.device)  # [N_valid, 2]
    
            # L1 loss per object
            l1 = nn.functional.l1_loss(pred, gt, reduction='none').mean(dim=1)  # (N_valid,)
            loss.append(l1.mean())
    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_dims.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
        
    def yaw_loss(self, pred_yaw, center_voxels, gtl):
        """
        Calculates loss for only and only for voxels that are closest to the object center
        
        pred_yaw:           [B, 1, W, D]
        center_voxels:      [B, 18, 4] --> (i, j, k, gt_idx)
        gtl:                [B, 18, 14] --> gtl[..., 11] = yaw
        """
        loss_terms = []
    
        for b in range(self.B):
            centers = center_voxels[b]
            gt_indices = centers[:, 3].long()
            valid_mask = (gt_indices >= 0) & (gtl[b, gt_indices, -1] == 1)  # present and valid objects
    
            if valid_mask.sum() == 0:
                continue
    
            i = centers[valid_mask, 0].long()
            k = centers[valid_mask, 2].long()  # note: D dimension
            gt_idx = gt_indices[valid_mask]
    
            # Predicted yaw values at object center voxels
            pred_vals = pred_yaw[b, 0, i, k]  # (N,)
    
            # Ground-truth yaw from gtl
            gt_vals = gtl[b, gt_idx, 11].to(pred_yaw.device)  # (N,)
    
            # Minimal angular difference in range [−π, π]
            diff = (pred_vals - gt_vals + math.pi) % (2 * math.pi) - math.pi
    
            # Smooth L1 over differences
            loss = nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='mean', beta=self.beta)
            loss_terms.append(loss)
    
        if len(loss_terms) == 0:
            return torch.tensor(0.0, device=pred_yaw.device, requires_grad=True)
        else:
            return torch.stack(loss_terms).mean()

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
                18 is maximum object per instance [so, if there are less pobjects, the valid flag is zero]
                14 --> label items
            3d box order is: h, w, l, cx, cy, cz, yaw --> 0:7
            box bev order is w, l, cx, cz, yaw --> 7:12
            object class (zero to num_classes-1 and -1 for not important objects) --> 12
            valid flag: [1 valid, 0 non-valid] ---> 13
        """
        assert prediction.shape == (len(gtl), 10, self.grid.shape[0], self.grid.shape[2]), \
            f"Prediction has a wrong shape, expected shape is: [n, out_ch, w_res, h_res, d_res], received: {prediction.shape}"
        assert gtl.shape == (len(prediction), 18, 14), "Collate function must be checked!"
        
        self.B = len(prediction)
        self.loss = {}
        
        if self.B == 0:
            device = prediction.device
            return {k: torch.tensor(0.0, device=device) for k in ['obj_conf', 'cls_loss', 'center_loss', 'dim_loss', 'yaw_angle_loss', 'total']}

        assignments = torch.full((self.B, self.grid.shape[0], self.grid.shape[1], self.grid.shape[2]),
                                      fill_value=-1, dtype=torch.long, device=self.grid.device)

        c_voxels = torch.full((self.B, 18, 4), fill_value=-1, dtype=torch.long, device=self.grid.device)
        
        assignments, c_voxels = assign_gt_to_voxels(assignments, c_voxels, self.grid, gtl, self.voxel_size)
        # so we have 3d grid, all -1 except for voxels of a gt object which has the gt index
        
        # Marking out of the bound objects from ground truth by zeroing the valid flag
        # + marking the closest voxel to the center of out of the bound objects (the gt_idx will be -1)
        assignments, c_voxels, gtl = self._mark_oo_FOV(assignments, c_voxels, gtl)

        # making the 3d space to BEV (priority index: -3, [indx >= 0], -2, -1)
        assignments_bev = aggregate_assignment(assignments)
        del assignments
        
        # Objectness loss
        self.loss['obj_conf'] = self.object_conf_loss(prediction[:, self.num_c:self.num_c+1], assignments_bev)

        # class loss (might be useful if focal loss is used)
        self.loss['cls_loss'] = self.classification_loss(prediction[:, 0:self.num_c],
                                                         prediction[:, self.num_c:self.num_c+1], assignments_bev, gtl)
        
        # bbox center loss
        self.loss['center_loss'] = self.center_loss(prediction[:, self.num_c+1 : self.num_c+3],
                                       assignments_bev, gtl, self.grid_bev)

        # bbox dim loss
        self.loss['dim_loss'] = self.dimension_loss(prediction[:, self.num_c+3: self.num_c+5],
                                       c_voxels, gtl)
        
        # yaw angle loss
        self.loss['yaw_angle_loss'] = self.yaw_loss(prediction[:, self.num_c+5:], c_voxels, gtl)

        # Total loss: Sum of all loss considering their importance based on self.loss_weights
        self.loss['total'] = self.loss_weights[0]*self.loss['obj_conf'] + \
                                self.loss_weights[1]*self.loss['cls_loss'] + \
                                self.loss_weights[2]*self.loss['center_loss'] + \
                                self.loss_weights[3]*self.loss['dim_loss'] + \
                                self.loss_weights[4]*self.loss['yaw_angle_loss']
                                  
        return self.loss
    
