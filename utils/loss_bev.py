#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 17:16:14 2025

@author: arash
"""

import torch.nn as nn

class loss_bev(nn.Module):
    def __init__(self, cfg):
        super(loss_bev, self).__init__()
        self.cfg = cfg
        self.num_c = self.cfg.model.num_class
        # TODO: change loss weights for BEV
        self.loss_weights = self.cfg.loss.weight
        
        self.alpha = self.cfg.loss.alpha
        self.gamma = self.cfg.loss.gamma
        self.beta = self.cfg.loss.beta
        self.voxel_size = self.cfg.grid_unc
        self.lb = self.cfg.debug_loss  # local debug
        
        
    
    
    
    def forward(self, prediction, init_gtl, grid, oob_mask_valid):
        """
        prediction is:
            pred[:num_class] = class probabilities,
            pred[num_class] = objecness score,
            pred[num_class + 1 : num_class + 4] = offsets from voxel center,
            pred[num_class + 4 : num_class + 7] = object dimensions,
            pred[-1] = object box yaw angle
            *** Prediction comes like [n, out_ch, w_res, h_res, d_res]
            
        init_gtl is is a list (length is num_batch) where for gt in gtl[index]:
            gt['category'] = object class (zero to num_classes-1 and -1 for not important objects)
            gt['bbox3d'] = order is: h, w, l, cx, cy, cz, yaw
            
        grid is:
            the grid with x, y, z of each voxel to match prediction with gt objects
            
        oob_mask_valid is:
            shape 100, 30, 70 and where the voxel is out of boundary of image --> False otherwise, True
        """