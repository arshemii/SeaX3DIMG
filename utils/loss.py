#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn
import torch
import numpy as np
    

def assign_gt_to_voxels(grid, gtl, voxel_size, debug, ignore_class_id=-1):
    """
    Assigns ground truth objects to the 3D grid.

    Args:
        grid: Tensor of shape [W, H, D, 3] representing voxel centers (x, y, z)
        gtl: List of dicts, each with:
            - 'category': class ID or -1 for ignored objects
            - 'bbox3d': (h, w, l, cx, cy, cz, yaw)
        Returns:
            assignments: Tensor of shape [W, H, D] with:
                - -1: background
                - >= 0: index of gtl object
                - -2: ignored (like tram)
    """
    W, H, D = grid.shape[0], grid.shape[1], grid.shape[2]
    device = grid.device
    
    assignments = torch.full((W, H, D), fill_value=-1, dtype=torch.long, device=device)
    center_voxels = []
    voxel_size = [unc for unc in voxel_size]
    
    if len(gtl) != 0:
        for idx, gt in enumerate(gtl):
            cat = gt["category"]
            h, w, l, cx, cy, cz, yaw = gt["bbox3d"]
            
            #zero correction for inside
            w = max(w, voxel_size[0])
            h = max(h, voxel_size[1])
            l = max(l, voxel_size[2])
                    
            # Compute voxel indices inside the box (simplified AABB logic)
            x_min, x_max = cx - w/2, cx + w/2
            y_min, y_max = cy - h/2, cy + h/2
            z_min, z_max = cz - l/2, cz + l/2
    
            xs, ys, zs = grid[:,:,:,0], grid[:,:,:,1], grid[:,:,:,2]
            inside = (xs >= x_min) & (xs <= x_max) & \
                     (ys >= y_min) & (ys <= y_max) & \
                     (zs >= z_min) & (zs <= z_max)
                     
            if int(cat) == ignore_class_id:
                assignments[inside] = -2  # Ignored class (e.g. Tram)
            else:
                assignments[inside] = idx  # Assign voxel to this gt
                
            # Find the voxel closest to GT center
            voxel_xyz = grid[inside]  # [N, 3]
            assert len(voxel_xyz) != 0 and len(gtl) != 0
            gt_center = torch.tensor([cx, cy, cz], device=grid.device)
            dists = torch.norm(voxel_xyz - gt_center, dim=1)
            if debug:
                print(f"==> shape of the voxel is {voxel_xyz.shape}")
                print(f"==> shape of the centers is {gt_center.shape}")
                print(f"==> length of the distances is {len(dists)}")
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
        self.loss_weights = self.cfg.loss.weight
        self.alpha = self.cfg.loss.alpha
        self.gamma = self.cfg.loss.gamma
        self.debug = self.cfg.debug
        self.voxel_size = self.cfg.grid_unc
        self.lb = True  # local debug
        
    def object_conf_loss(self, pred_obj_logits, voxel_assignments, gtl):
        """
        pred_obj_logits: [B, 1, W, H, D]
        voxel_assignments: a list of tensors with lenght = B and -1=bg, -2=ignored, >=0=object
        """
        loss = []
        
        if self.lb:
            print("==>   object_conf_loss checks: -----")
            print(f"==> value of self.B is {self.B}")
            print(f"length of gtl in confidence loss is: {len(gtl)}")
            
            
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
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
    
    def classification_loss(self, pred_cls_logits, center_voxels, gtl):
        """
        pred_cls_logits: [B, num_classes, W, H, D]
        center_voxels: list of per-batch lists of [(i, j, k, gt_idx)]
                        First list is for all batch, second list is for all dets in a frame
        """
        
        loss = []
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
                for (i, j, k, gt_idx) in center_voxels[b]:
                    # No need for one hot, should be [2]
                    target_cls = gtl[b][gt_idx]['category']
                    
                    if target_cls == -1:
                        continue
                    
                    pred = pred_cls_logits[b, :, i, j, k].unsqueeze(0)  # [1, C]
                    # Why not softmax? No softmax, raw logits
                    target_cls = target_cls.to(pred.device)
                    assert pred.device == target_cls.device
                    
                    ce = nn.functional.cross_entropy(pred, target_cls, reduction='none')
                    pt = torch.exp(-ce)
                    focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
                    loss.append(focal_loss.mean())

        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_cls_logits.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
    
    def center_loss(self, pred_offsets, center_voxels, gtl, grid):
        """
        pred_offsets: [B, 3, W, H, D]
        grid: [W, H, D, 3]
        """
        
        loss = []
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
                for (i, j, k, gt_idx) in center_voxels[b]:
                    voxel_center = grid[i, j, k, :]
                    pred_offset = pred_offsets[b, :, i, j, k]
                    pred_center = voxel_center + pred_offset
        
                    gt_center = gtl[b][gt_idx]['bbox3d'][3:6].to(pred_offset.device)
                    loss.append(nn.functional.l1_loss(pred_center, gt_center))
        
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_offsets.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
    
    def dimension_loss(self, pred_dims, center_voxels, gtl):
        """
        pred_dims: [B, 3, W, H, D]
        """
        
        loss = []
        
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
                for (i, j, k, gt_idx) in center_voxels[b]:
                    pred = pred_dims[b, :, i, j, k]
                    gt = gtl[b][gt_idx]['bbox3d'][0:3].to(pred.device)
                    loss.append(nn.functional.l1_loss(pred, gt))
                    
        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_dims.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
    
    def yaw_loss(self, pred_yaw, center_voxels, gtl):
        """
        pred_yaw: [B, 1, W, H, D]
        """
        loss = []
        
        for b in range(self.B):
            if len(gtl[b]) == 0:
                continue
            else:
                for (i, j, k, gt_idx) in center_voxels[b]:
                    pred = pred_yaw[b, 0, i, j, k]
                    gt = gtl[b][gt_idx]['bbox3d'][6].to(pred.device)
                    loss.append(nn.functional.smooth_l1_loss(pred, gt))

        if len(loss) == 0:
            return torch.tensor(0.0, device=pred_yaw.device, requires_grad=True)
        else:
            return torch.stack(loss).mean()
    
    def _drop_dets(self, assignments, init_c_voxels, init_gtl, oob_mask_valid):
        
        n = len(init_gtl)
        gtl = [[] for _ in range(n)]
        c_voxels = [[] for _ in range(n)]
        
        for bn in range(len(assignments)):
            if len(init_gtl[bn]) == 0:
                continue
            else:
                gtl_sample = []
                c_voxels_sample = []
                new_idx = 0
                for centers in init_c_voxels[bn]:
                    i, j, k, gt_idx = centers
                    if oob_mask_valid[i, j, k] == True:
                        c_voxels_sample.append((i, j, k, new_idx))
                        gtl_sample.append(init_gtl[bn][gt_idx])
                        new_idx += 1
                    else:
                        assignments[bn][assignments[bn] == gt_idx] = -1
                        
            gtl.append(gtl_sample)
            c_voxels.append(c_voxels_sample)
                        
        return assignments, c_voxels, gtl

    def forward(self, prediction, init_gtl, grid, oob_mask_valid):
        """
        prediction is:
            pred[:num_class] = class probabilities,
            pred[num_class] = objecness score,
            pred[num_class + 1 : num_class + 4] = offsets from voxel center,
            pred[num_class + 4 : num_class + 7] = object dimensions,
            pred[-1] = object box yaw angle
            
        gtl is is a list (length is num_batch) where for gt in gtl[index]:
            gt['category'] = object class (zero to num_classes-1 and -1 for not important objects)
            gt['bbox3d'] = order is: h, w, l, cx, cy, cz, yaw
            
        grid is:
            the grid with x, y, z of each voxel to match prediction with gt objects
            
        oob_mask_valid is:
            shape 100, 30, 70 and where the voxel is out of boundary of image --> False otherwise, True
            
        *** Prediction comes like [n, out_ch, w_res, h_res, d_res]
        """
        if self.lb:
            print("-------------------- loss forward started--------------------")
            print(f" ==> gtl is list?  {isinstance (init_gtl, list)}")
            if len(init_gtl) > 0:
                print(f" ==> gtl[0] is list?  {isinstance (init_gtl[0], list)}")
        
        # first part: a function to match each gt detection to corresponding voxels and find which voxel is closest to the box center
        assert grid.shape[-1] == 3, f"Expected grid[..., 3] for (x,y,z), got shape {grid.shape}"
        self.B = len(prediction)
        self.loss = {}
        
        if self.lb:
            print(f" ==>  Value of self.B is: {self.B}, lenght of prediction is: {len(prediction)}, lenght of init_gtl is: {len(init_gtl)}")
            print(f" ==>  shape of prediction is :  {prediction.shape}")
            if len(init_gtl) > 0:
                print(f" ==>  detection numbers in first sample of batch :  {len(init_gtl[0])}")
        
        if self.B == 0:
            device = prediction.device
            return {k: torch.tensor(0.0, device=device) for k in ['obj_conf', 'cls_loss', 'center_loss', 'dim_loss', 'yaw_angle_loss', 'total']}
        
        assignments = []
        init_c_voxels = []
        for i in range(self.B):
            ass, center_voxels = assign_gt_to_voxels(grid, init_gtl[i], self.voxel_size, self.debug)  # shape of ass: (res_w, res_h, res_d)
            assignments.append(ass)
            init_c_voxels.append(center_voxels)
            
        if self.lb:
            print(f" ==>  lenght of init voxel cntr: {len(init_c_voxels)}, and init assignment is: {len(assignments)}")
            if len(init_c_voxels) > 0:
                print(f" ==>  detection number of 1st sample based on voxel centers: {len(init_c_voxels[0])}")
        
        # Intermdiate step: refine detections (drop out)
        assignments, c_voxels, gtl = self._drop_dets(assignments,
                                                     init_c_voxels,
                                                     init_gtl,
                                                     oob_mask_valid)
        
        if self.lb:
            print(f" ==>  Dropped lenght --> voxel centers: {len(c_voxels)}, assignment: {len(assignments)}, gtl: {len(gtl)}")
            if len(c_voxels) > 0:
                print(f" ==>  detection number of 1st droped sample:  {len(c_voxels[0])}")
                assert len(init_c_voxels[0]) >= len(c_voxels[0]), "After drop, must be equal or less detections!"
        
        # Second part: objectness loss
        self.loss['obj_conf'] = self.object_conf_loss(prediction[:, self.num_c:self.num_c+1], assignments, gtl)
        
        # Third part: class loss (might be useful if focal loss is used)
        self.loss['cls_loss'] = self.classification_loss(prediction[:, 0:self.num_c], c_voxels, gtl)
        
        # Forth part: bbox center loss
        self.loss['center_loss'] = self.center_loss(prediction[:, self.num_c+1 : self.num_c+4],
                                       c_voxels, gtl, grid)
        
        # Fifth part: bbox dim loss
        self.loss['dim_loss'] = self.dimension_loss(prediction[:, self.num_c+4: self.num_c+7],
                                       c_voxels, gtl)
        
        # Sixth part: yaw angle loss
        # Any normalization on each term? No, just weights
        assert prediction[:, self.num_c+7:].shape[1] == 1
        self.loss['yaw_angle_loss'] = self.yaw_loss(prediction[:, self.num_c+7:], c_voxels, gtl)
                
        # Total loss: Sum of all loss considering their importance based on self.loss_weights
        self.loss['total'] = self.loss_weights[0]*self.loss['obj_conf'] + \
                                self.loss_weights[1]*self.loss['cls_loss'] + \
                                self.loss_weights[2]*self.loss['center_loss'] + \
                                self.loss_weights[3]*self.loss['dim_loss'] + \
                                self.loss_weights[4]*self.loss['yaw_angle_loss']
                                
        if self.lb:
            print("-------------------- loss forward ended--------------------")
            
        return self.loss
    
    def accumulate_loss(self):
        raise NotImplementedError("To accumulate each iteration and so so so on")
        
        


def test_loss_function():
    from model_cong import config_generator
    cfg = config_generator()
    
    dW, dH, dD = (20.0, 12.0, 42.0)
    voxel_size = (0.2, 0.4, 0.6)
    
    # Compute grid size
    W = int(round(dW / voxel_size[0]))
    H = int(round(dH / voxel_size[1]))
    D = int(round(dD / voxel_size[2]))

    print(f"Grid size: W={W}, H={H}, D={D}")

    # Generate voxel grid in real-world coordinates [3, W, H, D]
    x = torch.arange(W).float() * voxel_size[0]
    y = torch.arange(H).float() * voxel_size[1]
    z = torch.arange(D).float() * voxel_size[2]
    grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=0)

    # Prepare prediction tensor
    B = cfg.num_batch
    C = cfg.model.num_class
    out_ch = C + 1 + 3 + 3 + 1  # class + objectness + offset + size + yaw
    pred = torch.zeros((B, out_ch, W, H, D))

    # Fill with synthetic values
    pred[:, :C] = torch.randn(B, C, W, H, D).clamp(-2, 2)  # class logits
    pred[:, C] = torch.randn(B, W, H, D).clamp(-2, 2)      # objectness logit
    pred[:, C+1:C+4] = (torch.rand(B, 3, W, H, D) - 0.5) * torch.tensor(voxel_size).reshape(1, 3, 1, 1, 1)
    pred[:, C+4:C+7] = torch.rand(B, 3, W, H, D) * 3 + 1  # dimensions in [1, 4] m
    pred[:, -1] = torch.rand(B, W, H, D) * np.pi * 2 - np.pi  # yaw in [-π, π]

    # Generate synthetic ground truth
    gtl = []
    for b in range(B):
        num_obj = np.random.randint(1, 6)  # 1 to 5 objects
        objs = []
        for _ in range(num_obj):
            category = np.random.randint(0, C)
            cx = np.random.rand() * dW
            cy = np.random.rand() * dH
            cz = np.random.rand() * dD
            h, w, l = np.random.rand(3) * 2 + 1  # [1, 3] meters
            yaw = np.random.rand() * 2 * np.pi - np.pi
            bbox3d = np.array([h, w, l, cx, cy, cz, yaw], dtype=np.float32)
            objs.append({
                'category': category,
                'bbox2d': None,
                'bbox3d': bbox3d,
                'truncation': None,
                'occlusion': None,
                'angle_observation': None
            })
        gtl.append(objs)

    # Instantiate and compute loss
    loss_fn = loss_3d(cfg)
    loss_output = loss_fn(pred, gtl, grid)

    return grid, pred, gtl, loss_output
