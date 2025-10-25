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
        
        self.grid = cfg.grid_forward[0]
        assert self.grid.shape[-1] == 3, f"Expected grid[..., 3] for (x,y,z), got shape {cfg.grid_forward[0].shape}"
        
        self.oob_mask_valid = cfg.oob_mask_valid[0] # inside FOV --> True
        self.num_c = self.cfg.model.num_class
        self.loss_weights = self.cfg.loss.weight
        self.alpha = self.cfg.loss.alpha
        self.gamma = self.cfg.loss.gamma
        self.beta = self.cfg.loss.beta
        self.object_threshold = self.cfg.loss.object_threshold_loss
        self.zeta = self.cfg.loss.zeta

    
    def object_conf_loss(self, pred_obj_logits, assignments):
        """
        If a voxel is inside the FOV (assignments != -3),
        and not an ignored object (assignment != -2), loss is calculated!
        
        pred_obj_logits:    [B, 1, W, H, D]
        assignments:        [B, W, H, D]
        """
        loss = []
                
        for b in range(self.B):
            mask = (assignments[b] >= -1)
            
            assert mask.any() == True

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
        pred_cls_logits: [B, num_classes, W, H, D]
        pred_obj_logits: [B, 1, W, H, D]
        assignments:     [B, W, H, D]
        gtl:             [B, max_object, 12]
        """
        assert pred_cls_logits.shape[1] == self.num_c, "Class number mismatch in output and config"
        
        loss = []
    
        for b in range(self.B):
            obj_mask = (assignments[b] >= 0)
            bg_mask = (assignments[b] == -1)
            
            # Loss in valid voxels
            if obj_mask.any():
                
                voxel_obj_indices = assignments[b][obj_mask].long()         # [N]
                target_tensor = gtl[b][voxel_obj_indices, 7].long()         # [N]
                
                mask_debug = (target_tensor < 0)
                assert mask_debug.any()  == False
                
                pred_voxels = pred_cls_logits[b].permute(1, 2, 3, 0)[obj_mask]  # [N, num_classes]
                
                ce = nn.functional.cross_entropy(pred_voxels, target_tensor, reduction='none', ignore_index=-100)
                pt = torch.exp(-ce)
                focal = self.alpha * (1 - pt) ** self.gamma * ce
                loss.append(focal.mean())
    
            # loss if the voxel has no object but it has high objectness score
            if bg_mask.any():
                high_obj_mask = torch.sigmoid(pred_obj_logits[b, 0]) > self.object_threshold
                bad_mask = bg_mask & high_obj_mask
                if bad_mask.any():
                    pred_bg_voxels = pred_cls_logits[b].permute(1, 2, 3, 0)[bad_mask]
                    pred_probs = nn.functional.softmax(pred_bg_voxels, dim=-1)
                    target_probs = torch.full_like(pred_probs, 1.0 / self.num_c)
                    kl_loss = nn.functional.kl_div(pred_probs.log(), target_probs, reduction='batchmean')
                    loss.append(self.zeta * kl_loss)  # small weight for regularization
    
        if not loss:
            return torch.tensor(0.0, device=pred_cls_logits.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
        
    def center_loss(self, pred_offsets, assignments, gtl):
        """
        The loss for center is calculated for each voxel that is assigned to a valid object
        
        pred_offsets:   [B, 3, W, H, D]
        assignments:    [B, W, H, D]
        gtl:            [B, 18, 12] --> gtl[b, :, 3:6] are (cx_off, ch_off, cz_off)
        """
        loss = []
    
        for b in range(self.B):
            # Only compute if there are valid objects
            valid_mask = (assignments[b] >= 0)
            if not valid_mask.any():
                continue
    
            i, j, k = torch.nonzero(valid_mask, as_tuple=True)  # (N,)
            gt_indices = assignments[b][i, j, k].long()  # (N,)
    
            # all voxel centers that are assigned to a gt index
            voxel_centers = self.grid[i, j, k, :].to(pred_offsets.device)   # (N, 3)
            # predicted offsets for valid voxels
            pred_offsets_valid = pred_offsets[b][:, i, j, k].permute(1, 0)  # (N, 3)
            # predicted centers
            pred_obj_centers = voxel_centers.to(pred_offsets.device) + pred_offsets_valid  # (N, 3)
    
            # ground truth absolute centers
            gt_centers = gtl[b, gt_indices, 3:6].to(pred_offsets.device)
    
            # smooth L1 loss per voxel
            l1 = nn.functional.smooth_l1_loss(pred_obj_centers, gt_centers, reduction='none', beta=self.beta)
            l1 = l1.mean(dim=1)  # (N,)
    
            loss.append(l1.mean())
    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_offsets.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
            
        
    def dimension_loss(self, pred_dims, assignments, gtl):
        """
        Calculates loss only for voxels closest to the object center
        ** Dimension is constant for all voxels of an object
        
        pred_dims:          [B, 3, W, H, D] --> (w, h, l)
        assignments:        [B, W, H, D]
        gtl:                [B, 18, 12] --> gtl[b, :, :3] are (w, h, l)
        """
        loss = []
        
        for b in range(self.B):
            valid_mask = (assignments[b] >= 0)
            
            if not valid_mask.any():
                continue
    
            unique_obj_indices = torch.unique(assignments[b][valid_mask].long())    # [unique indices --> U]        
    
            cv = gtl[b, unique_obj_indices, 8:11].long()  # [U, 3] --> voxel center closest to the object center
            i = cv[:, 0]
            j = cv[:, 1]
            k = cv[:, 2]
            
            pred = pred_dims[b, :, i, j, k].permute(1, 0)       # [U, 3]
            gt = gtl[b, unique_obj_indices, :3].to(pred.device)       # [U, 3]
    
            # L1 loss per object
            l1 = nn.functional.l1_loss(pred, gt, reduction='none').mean(dim=1)  # (N_valid,)
            loss.append(l1.mean())
    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_dims.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
        
    def yaw_loss(self, pred_yaw, assignments, gtl):
        """
        Calculates loss only for voxels that are closest to the object center
        
        pred_yaw:           [B, 1, W, H, D]
        assignments:        [B, W, H, D]
        gtl:                [B, 18, 12] --> gtl[b, :, 6] is yaw
        """
        loss_terms = []
    
        for b in range(self.B):
            valid_mask = (assignments[b] >= 0)
            
            if not valid_mask.any():
                continue
            
            unique_obj_indices = torch.unique(assignments[b][valid_mask].long())    # [unique indices --> U]        
    
            cv = gtl[b, unique_obj_indices, 8:11].long()  # [U, 3] --> voxel center closest to the object center
            i = cv[:, 0]
            j = cv[:, 1]
            k = cv[:, 2]
            
            pred = pred_yaw[b, :, i, j, k].permute(1, 0)       # [U, ]
            gt = gtl[b, unique_obj_indices, 6].to(pred.device)       # [U, ]
    
            # Minimal angular difference in range [−π, π]
            diff = (pred - gt + math.pi) % (2 * math.pi) - math.pi
    
            # Smooth L1 over differences
            loss = nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='mean', beta=self.beta)
            loss_terms.append(loss)
    
        if len(loss_terms) == 0:
            return torch.tensor(0.0, device=pred_yaw.device, requires_grad=True)
        else:
            return torch.stack(loss_terms).mean()


    def disparity_loss(self, disparity_pred, disparity_gtl):
        """        
        disparity_pred:             [B, 1, image_h / 4, image_w / 4]
        disparity_gtl               [B, image_h / 4, image_w / 4]
        """
        disparity_gtl = disparity_gtl.squeeze(1)
        
        if len(disparity_pred) != 0:
            assert disparity_gtl.shape == disparity_pred.shape, f"gt is {disparity_gtl.shape} but pred is {disparity_pred.shape}"
            
            valid_depth_mask = (disparity_gtl > 0)
            
            pred_valid = disparity_pred[valid_depth_mask]
            gtl_valid = disparity_gtl[valid_depth_mask]
            
            return nn.functional.smooth_l1_loss(pred_valid, gtl_valid, reduction='mean', beta=self.beta)
        else:
            return torch.tensor(0.0, device=disparity_pred.device, requires_grad=True)
            

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
    
