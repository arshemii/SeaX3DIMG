#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn
import torch
import math
    

class loss_bev(nn.Module):
    def __init__(self, cfg):
        super(loss_bev, self).__init__()
        self.cfg = cfg
        
        self.grid = cfg.grid[0]
        assert self.grid.shape[-1] == 3, f"Expected grid[..., 3] for (x,y,z), got shape {cfg.grid.shape}"
        
        self.grid_bev = self._grid3d_to_grid2d()
        
        self.oob_mask_valid = cfg.oob_mask_valid
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
        gtl:             [B, 18, 17]
        ** gtl --> for validity flag in gtl[:, :, 16]:
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
                tg_classes = gtl[b, :, 12].clone()            # [18] or max_object
                tg_classes[gtl[b, :, 16] == 0] = -100         # invalid (gtl tensor zero padding) or OoB
                
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
        gtl:            [B, 18, 17] --> gtl[b, :, 9:11] are (cx_off, cz_off)
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
            
        
    def dimension_loss(self, pred_dims, gtl):
        """
        Calculates loss for only and only for voxels that are closest to the object center (Dimension is constant for all voxels of an object)
        
        pred_dims:          [B, 2, W, D] --> (w, l)
        gtl:                [B, 18, 17] --> gtl[b, :, 7:9] are (w, l)
        """
        loss = []
        
        for b in range(self.B):
            cv = gtl[b, :, 13:16]  # [18, 3]
            gt_indices = cv[:, 3].long()  # [18]
            valid_mask = (gtl[b, gt_indices.clamp_min(0), 16] == 1)
            
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

    def disparity_loss(self, disparity_pred, disparity_gt, conf):
        raise NotImplementedError("must implement this loss!")

    def forward(self, prediction, gtl, assignments, assignments_bev, disparity_pred = None, disparity_gt = None, conf = None):
        """
        prediction is:
            pred[:num_class] = class probabilities,
            pred[num_class = objecness score,
            pred[num_class + 1 : num_class + 3] = offsets from voxel center,
            pred[num_class + 3 : num_class + 5] = object dimensions,
            pred[-1] = object box yaw angle
            *** Prediction comes like [n, out_ch, w_res, d_res]
            
        gtl is is a Tensor:
            Shape: (B, 18, 14) where:
                B is batch size
                18 is maximum object per instance [so, if there are less pobjects, the valid flag is zero]
                17 --> label items
            3d box order is: h, w, l, cx, cy, cz, yaw --> 0:7
            box bev order is w, l, cx, cz, yaw --> 7:12
            object class (zero to num_classes-1 and -1 for not important objects) --> 12
            closest voxel center --> 13:16
            valid flag: [1 valid, 0 non-valid] ---> 16
            
        disparity_pred shape (B, 1, 128, 240)
        conf shape (B, 1, 128, 240)
        disparity_gtl shape (B, 1, 128, 240)
        """
        assert prediction.shape == (len(gtl), 10, self.grid.shape[0], self.grid.shape[2]), \
            f"Prediction has a wrong shape, expected shape is: [n, out_ch, w_res, h_res, d_res], received: {prediction.shape}"
        assert gtl.shape == (len(prediction), 18, 14), "Collate function must be checked!"
        
        self.B = len(prediction)
        self.loss = {}
        
        if self.B == 0:
            device = prediction.device
            if self.cfg.loss.aux_loss:
                return {k: torch.tensor(0.0, device=device) for k in ['obj_conf', 'cls_loss', 'center_loss', 'dim_loss', 'yaw_angle_loss','disparity_loss', 'total']}
            else:
                return {k: torch.tensor(0.0, device=device) for k in ['obj_conf', 'cls_loss', 'center_loss', 'dim_loss', 'yaw_angle_loss', 'total']}

        
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
        
        if self.cfg.loss.aux_loss:
            assert disparity_pred != None and disparity_gt != None and conf != None
            # yaw angle loss
            self.loss['disparity_loss'] = self.disparity_loss(disparity_pred, disparity_gt, conf)

        # Total loss: Sum of all loss considering their importance based on self.loss_weights
        self.loss['total'] = self.loss_weights[0]*self.loss['obj_conf'] + \
                                self.loss_weights[1]*self.loss['cls_loss'] + \
                                self.loss_weights[2]*self.loss['center_loss'] + \
                                self.loss_weights[3]*self.loss['dim_loss'] + \
                                self.loss_weights[4]*self.loss['yaw_angle_loss']
                                
        if self.cfg.loss.aux_loss:
            self.loss['total'] += self.loss_weights[5]*self.loss['disparity_loss']
                                  
        return self.loss
    
