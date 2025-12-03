#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn
import torch
import math
    

class loss3d(nn.Module):
    def __init__(self, cfg):
        super(loss3d, self).__init__()
        self.cfg = cfg
        
        self.grid = cfg.grid_forward[0].to(self.cfg.device[0])
        assert self.grid.shape[-1] == 3, f"Expected grid[..., 3] for (x,y,z), got shape {cfg.grid_forward[0].shape}"
        
        self.oob_mask_valid = cfg.oob_mask_valid[0] # inside FOV --> True
        self.num_c = self.cfg.model.num_class
        # self.loss_weights = self.cfg.loss.weight
        self.alpha = self.cfg.loss.alpha
        self.gamma = self.cfg.loss.gamma
        self.beta = self.cfg.loss.beta
        self.object_threshold = self.cfg.loss.object_threshold_loss
        self.zeta = self.cfg.loss.zeta
        self.center_scale = torch.tensor([self.cfg.grid_unc[0],
                                          self.cfg.grid_unc[1],
                                          self.cfg.grid_unc[2]]).to(self.cfg.device[0])
      
    # DONE        
    def object_conf_loss(self, pred_obj_logits, assignments):
        """
        Focal BCE loss for objectness.
        pred_obj_logits: [B, 1, W, H, D]
        assignments:     [B, W, H, D]
        """
        loss = []
        count = 0
        for b in range(pred_obj_logits.shape[0]):
            mask = (assignments[b] >= -1)
            if not mask.any():
                continue
            
            pred = pred_obj_logits[b, 0][mask]
            pred = pred.clamp(-20, 20)
            tgt = (assignments[b][mask] >= 0).float()
            
            ##### ------- Safety check, mjst be removed -------
            has_nan = torch.isnan(pred).any()
            has_inf = torch.isinf(pred).any()
            if has_nan or has_inf:
                pred = torch.nan_to_num(pred, nan=1e-6, posinf=1e3)
                count += 1
            ##### --------------------------------------------
            
            # Standard CE with focal
            bce = nn.functional.binary_cross_entropy_with_logits(pred, tgt, reduction='none')
            pt  = torch.exp(-bce).clamp(min=1e-6, max=1-1e-6)
            
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
            
            ##### ------- Safety check, mjst be removed -------
            has_nan = torch.isnan(focal_loss).any()
            has_inf = torch.isinf(focal_loss).any()
            if has_nan or has_inf:
                focal_loss = torch.nan_to_num(focal_loss, nan=1e-6, posinf=1e3)
                count += 1
            ##### ---------------------------------------------
            
            loss.append(focal_loss.mean())
    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_obj_logits.device, requires_grad=True), count
        else:
            return torch.stack(loss).mean(), count
       
    # DONE    
    def classification_loss(self, pred_cls_logits, pred_obj_logits, assignments, gtl):
        """
        Numerically stable classification loss with focal weighting and background KL regularization.
    
        pred_cls_logits: [B, num_classes, W, H, D]
        pred_obj_logits: [B, 1, W, H, D]
        assignments:     [B, W, H, D]
        gtl:             [B, max_object, 12]
        """
        assert pred_cls_logits.shape[1] == self.num_c, "Class number mismatch in output and config"
        
        loss_terms = []
        count = 0
        for b in range(pred_cls_logits.shape[0]):
            # Masks
            obj_mask = (assignments[b] >= 0)
            bg_mask = (assignments[b] == -1)
    
            # --- OBJECT VOXELS ---
            if obj_mask.any():
                voxel_obj_indices = assignments[b][obj_mask].long()         # [N]
                target_tensor = gtl[b][voxel_obj_indices, 7].long()         # [N]
                
                assert (target_tensor >= 0).all(), "Target class < 0 detected"
    
                pred_voxels = pred_cls_logits[b].permute(1, 2, 3, 0)[obj_mask].clamp(-20, 20)  # [N, num_classes]
                
                ##### ------- Safety check, mjst be removed -------
                has_nan = torch.isnan(pred_voxels).any()
                has_inf = torch.isinf(pred_voxels).any()
                if has_nan or has_inf:
                    pred_voxels = torch.nan_to_num(pred_voxels, nan=1e-6, posinf=1e3)
                    count += 1
                ##### --------------------------------------------
                # Standard CE with focal
                ce = nn.functional.cross_entropy(pred_voxels, target_tensor, reduction='none', ignore_index=-100)
                pt = torch.exp(-ce)
                focal = self.alpha * (1 - pt) ** self.gamma * ce
                loss_terms.append(focal.mean())
    
            # --- BACKGROUND VOXELS KL REGULARIZATION ---
            if bg_mask.any():
                pred_bg_voxels = pred_cls_logits[b].permute(1, 2, 3, 0)[bg_mask].clamp(-20, 20)
                pred_probs = nn.functional.softmax(pred_bg_voxels, dim=-1).clamp(min=1e-6)  # safe softmax
                target_probs = torch.full_like(pred_probs, 1.0 / self.num_c).clamp(min=1e-6)
    
                # Only penalize voxels with high objectness
                high_obj_mask = torch.sigmoid(pred_obj_logits[b, 0][bg_mask]) > self.object_threshold
                if high_obj_mask.any():
                    pred_probs_high = pred_probs[high_obj_mask]
                    ##### ------- Safety check, mjst be removed -------
                    has_nan = torch.isnan(pred_probs_high).any()
                    has_inf = torch.isinf(pred_probs_high).any()
                    if has_nan or has_inf:
                        pred_probs_high = torch.nan_to_num(pred_probs_high, nan=1e-6, posinf=1e3)
                        count += 1
                    ##### --------------------------------------------
                    target_probs_high = target_probs[high_obj_mask]
                    kl_loss = nn.functional.kl_div(pred_probs_high.log(), target_probs_high, reduction='batchmean')
                    loss_terms.append(self.zeta * kl_loss)
    
        if not loss_terms:
            return torch.tensor(0.0, device=pred_cls_logits.device, requires_grad=True), count
        else:
            return torch.stack(loss_terms).mean(), count
       
    # DONE    
    def center_loss(self, pred_offsets, assignments, gtl):
        """
        The loss for center is calculated for each voxel that is assigned to a valid object
        
        pred_offsets:   [B, 3, W, H, D] --> inside the model head is normalized [-1, +1] using torch.tanh 
        assignments:    [B, W, H, D]
        gtl:            [B, 18, 12] --> gtl[b, :, 3:6] are (cx_off, ch_off, cz_off)
        """
        loss_terms = []
        count = 0
        for b in range(pred_offsets.shape[0]):
            # Mask for valid object voxels
            valid_mask = (assignments[b] >= 0)
            if not valid_mask.any():
                continue
    
            i, j, k = torch.nonzero(valid_mask, as_tuple=True)
            gt_indices = assignments[b][i, j, k].long()
            
            assert (gt_indices >= gtl.shape[1]).sum() == 0
    
            voxel_centers = self.grid[i, j, k, :].to(pred_offsets.device)    # (N, 3)
            pred_offsets_valid = pred_offsets[b][:, i, j, k].permute(1, 0)   # (N, 3)
    
            pred_obj_centers = voxel_centers + (pred_offsets_valid * self.center_scale)
            gt_centers = gtl[b, gt_indices, 3:6].to(pred_offsets.device)
    
            # Filter out any NaNs/Infs in pred or gt
            valid_values = (~torch.isnan(pred_obj_centers).any(dim=1)) & (~torch.isnan(gt_centers).any(dim=1))
            if not valid_values.any():
                continue
    
            pred_obj_centers = pred_obj_centers[valid_values]
            gt_centers = gt_centers[valid_values]
    
            l1 = nn.functional.smooth_l1_loss(pred_obj_centers, gt_centers, reduction='none', beta=self.beta)
            if torch.isnan(l1).any() or torch.isinf(l1).any():
                count += 1
                l1 = torch.nan_to_num(l1, nan=0.0, posinf=1e3, neginf=-1e3)
            l1 = l1.mean(dim=1)  # per voxel
            loss_terms.append(l1.mean())
    
        if not loss_terms:
            return torch.tensor(0.0, device=pred_offsets.device, requires_grad=True), count
        else:
            return torch.stack(loss_terms).mean(), count
    
    # DONE
    def dimension_loss(self, pred_dims, assignments, gtl):
        """
        Calculates loss only for voxels closest to the object center
        ** Dimension is constant for all voxels of an object
        
        pred_dims:          [B, 3, W, H, D] --> (w, h, l), passed already from softplus
        assignments:        [B, W, H, D]
        gtl:                [B, 18, 12] --> gtl[b, :, :3] are (w, h, l)
        """
        loss = []
        count = 0
        for b in range(pred_dims.shape[0]):
            valid_mask = (assignments[b] >= 0)
            
            if not valid_mask.any():
                continue
    
            unique_obj_indices = torch.unique(assignments[b][valid_mask].long())    # [unique indices --> U]
            
            assert (unique_obj_indices >= gtl.shape[1]).sum() == 0
    
            cv = gtl[b, unique_obj_indices, 8:11].long()  # [U, 3] --> voxel center closest to the object center
            i = cv[:, 0]
            j = cv[:, 1]
            k = cv[:, 2]
            
            pred = pred_dims[b, :, i, j, k].permute(1, 0)                    # [U, 3]
            gt = gtl[b, unique_obj_indices, :3].to(pred.device)              # [U, 3]
            
            ##### ------- Safety check, mjst be removed -------
            has_nan = torch.isnan(pred).any()
            has_inf = torch.isinf(pred).any()
            if has_nan or has_inf:
                pred = torch.nan_to_num(pred, nan=1e-6, posinf=1e3)
                count += 1
            ##### --------------------------------------------
            
            # L1 loss per object
            l1 = nn.functional.smooth_l1_loss(pred, gt + 1e-6, reduction='none').mean(dim=1)         # (N_valid,)
            loss.append(l1.mean())
    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_dims.device, requires_grad=True), count
        else:
            return torch.stack(loss).mean(), count
        
    # DONE
    def yaw_loss(self, pred_yaw, assignments, gtl):
        """
        Calculates loss only for voxels that are closest to the object center
        
        pred_yaw:           [B, 1, W, H, D]
        assignments:        [B, W, H, D]
        gtl:                [B, 18, 12] --> gtl[b, :, 6] is yaw
        """
        loss_terms = []
        count = 0
        for b in range(pred_yaw.shape[0]):
            valid_mask = (assignments[b] >= 0)
            
            if not valid_mask.any():
                continue
            
            unique_obj_indices = torch.unique(assignments[b][valid_mask].long())    # [unique indices --> U]   
            assert (unique_obj_indices >= gtl.shape[1]).sum() == 0
    
            cv = gtl[b, unique_obj_indices, 8:11].long()  # [U, 3] --> voxel center closest to the object center
            i = cv[:, 0]
            j = cv[:, 1]
            k = cv[:, 2]
            
            pred = pred_yaw[b, :, i, j, k].view(-1)                            # [U]
            pred = math.pi * pred
            gt = gtl[b, unique_obj_indices, 6].to(pred.device)                 # [U, ]
    
            # Sin-cos regression
            pred_sin, pred_cos = torch.sin(pred), torch.cos(pred)
            gt_sin, gt_cos = torch.sin(gt), torch.cos(gt)
    
            loss_sin = nn.functional.smooth_l1_loss(pred_sin, gt_sin, reduction='mean', beta=self.beta)
            loss_cos = nn.functional.smooth_l1_loss(pred_cos, gt_cos, reduction='mean', beta=self.beta)
            if torch.isnan(loss_sin).any() or torch.isinf(loss_sin).any():
                count += 1
                loss_sin = torch.nan_to_num(loss_sin, nan=0.0, posinf=1e3, neginf=-1e3)
                
            if torch.isnan(loss_cos).any() or torch.isinf(loss_cos).any():
                count += 1
                loss_cos = torch.nan_to_num(loss_cos, nan=0.0, posinf=1e3, neginf=-1e3)
                
            loss_terms.append(0.5 * (loss_sin + loss_cos))
    
        if len(loss_terms) == 0:
            return torch.tensor(0.0, device=pred_yaw.device, requires_grad=True), count
        else:
            return torch.stack(loss_terms).mean(), count

    # DONE
    def disparity_loss(self, disparity_pred, disparity_gtl, normalized = True):
        """
        disparity_pred: [B, 1, H/4, W/4]
        disparity_gtl:  [B, 1, H/4, W/4]
        """
        self.B = len(disparity_pred)
        count = 0

        if normalized:
            disparity_pred = disparity_pred / self.cfg.loss.max_disp
            disparity_gtl = disparity_gtl / self.cfg.loss.max_disp
        
        if disparity_pred.numel() == 0:
            return torch.tensor(0.0, device=disparity_pred.device, requires_grad=True), count
    
        assert disparity_gtl.shape == disparity_pred.shape, \
            f"gt is {disparity_gtl.shape} but pred is {disparity_pred.shape}"
        
        valid_mask = (disparity_gtl > 0)
    
        if not valid_mask.any():  # no valid pixels
            return torch.tensor(0.0, device=disparity_pred.device, requires_grad=True), count
        
        pred_valid = disparity_pred[valid_mask]   
        gtl_valid  = disparity_gtl[valid_mask]
        
        ##### ------- Safety check, mjst be removed -------
        has_nan = torch.isnan(pred_valid).any()
        has_inf = torch.isinf(pred_valid).any()
        if has_nan or has_inf:
            pred_valid = torch.nan_to_num(pred_valid, nan=1e-6, posinf=1e3)
            count += 1
        ##### --------------------------------------------

        if normalized:
            #return nn.functional.l1_loss(torch.log(pred_valid + 1e-6), torch.log(gtl_valid + 1e-6)), count
            return nn.functional.smooth_l1_loss(torch.log(pred_valid + 1e-6), torch.log(gtl_valid + 1e-6)), count
        else:
            return nn.functional.smooth_l1_loss(pred_valid, gtl_valid, reduction='mean', beta=self.beta), count

    def forward(self, prediction, disparity_pred,
                gtl, assignments, disparity_gtl):
        """
        prediction is a tuple:
            prediction[0] = objectness            --> [B,1,D,H,W]
            prediction[1] = box dim               --> [B,3,D,H,W]
            prediction[2] = center offsets        --> [B,3,D,H,W]
            prediction[3] = class probabilities   --> [B,K,D,H,W]
            prediction[4] = yaw angle             --> [B,1,D,H,W]
            
        gtl is is a Tensor --> [B, max_object, 12]
            max_object: maximum possible object in a frame (from dataset statistics)
            12 --> label items:
                BBOX:           [h, w, l, cx, cy, cz, yaw]                              --> 0:7
                category:       (zero to num_classes-1 and -2 for not ignored class)    --> 7
                closest voxel:                                                          --> 8:11
                valid flag:     [1 valid, 0 non-valid]                                  --> 11
                ** valid flag can be directly obtained from assignment, but for computational
                   efficiency, we have this **
            
        assignments:                        [B, W, H, D] --> -3: OOB, -2: ignored, -1: bg, rest are obj index
        disparity_pred:                     [B, 1, 128, 240]
        disparity_gtl:                      [B, 1, 128, 240]
        """
        assert gtl.shape == (len(prediction[0]), self.cfg.max_obj, 12), "Wrong gtl, Collate function must be checked!"
        
        self.B = len(prediction[0])
        self.loss = {}
        
        if self.B == 0:
            device = prediction.device
            if self.cfg.loss.aux_loss:
                return {k: torch.tensor(0.0, device=device) for k in ['obj_conf', 'cls_loss', 'center_loss', 'dim_loss', 'yaw_angle_loss','disparity_loss', 'total']}
            else:
                return {k: torch.tensor(0.0, device=device) for k in ['obj_conf', 'cls_loss', 'center_loss', 'dim_loss', 'yaw_angle_loss', 'total']}

        
        # Objectness loss
        self.loss['obj_conf'] = self.object_conf_loss(prediction[0], assignments)

        # class loss
        self.loss['cls_loss'] = self.classification_loss(prediction[3], prediction[0], assignments, gtl)
        
        # bbox center loss
        self.loss['center_loss'] = self.center_loss(prediction[2], assignments, gtl)

        # bbox dim loss
        self.loss['dim_loss'] = self.dimension_loss(prediction[1], assignments, gtl)
        
        # yaw angle loss
        self.loss['yaw_angle_loss'] = self.yaw_loss(prediction[4], assignments, gtl)
        
        if self.cfg.loss.aux_loss:
            assert disparity_pred != None and disparity_gtl != None
            self.loss['disparity_loss'] = self.disparity_loss(disparity_pred, disparity_gtl)

        # Total loss: Sum of all loss considering their importance based on self.loss_weights
        self.loss['total'] = self.loss_weights[0]*self.loss['obj_conf'] + \
                                self.loss_weights[1]*self.loss['cls_loss'] + \
                                self.loss_weights[2]*self.loss['center_loss'] + \
                                self.loss_weights[3]*self.loss['dim_loss'] + \
                                self.loss_weights[4]*self.loss['yaw_angle_loss']
                                
        if self.cfg.loss.aux_loss:
            self.loss['total'] += self.loss_weights[5]*self.loss['disparity_loss']
                                  
        return self.loss
    
