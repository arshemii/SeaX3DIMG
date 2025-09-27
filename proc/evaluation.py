#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 17:38:07 2025

@author: arash
"""

import torch
import utils.eval_utils as eu

class Evaluate:
    def __init__(self, cfg, prediction, gt, grid):
        super(Evaluate, self).__init__()
        """
        prediction is a list: each element is a torch.tensor on CPU with (num_batch, 10, res_w, res_z)
        
        gt is is a list (length is num_batch) where for each gt in gt[index]:
            gt['category'] = object class (zero to num_classes-1 and -1 for not important objects)
            gt['bbox3d'] = order is: h, w, l, cx, cy, cz, yaw
            and mut be present:
                gt['bbox_bev'] order is w, l, cx, cz, yaw
            
        grid is:
            the grid with x, y, z of each voxel to match prediction with gt objects
        """
        self.cfg = cfg
        
        self.grid = eu.grid3d_to_grid2d(grid)  # TODO: should I move to CPU? No. I will use GPU!
        self.preds = prediction
        self.gt = gt
        
    def _pred_preparation(self):
        self.preds[:, 0, :, :] = torch.sigmoid(self.preds[:, 0, :, :])
        self.preds[:, 1:5, :, :] = torch.softmax(self.preds[:, 1:5, :, :], dim=1)
        self.preds = eu.compress_tensor(self.preds)
        self.preds = eu.add_voxel_centers(self.preds, self.grid)
        self.preds_list = eu.filter_detections(self.preds, self.cfg.eval.objectness_threshold)
    
    def _frame_eval(self, pred_f, gt_f):
        # pred_f is tensor of shape [10, res_w, res_z]
        # gt_f is a list, where for each element, there is a dict
        
        
        
        
        return 2