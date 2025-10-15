#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note: all the inputs to these modules are in shape:
    (1, 256, 100, 30, 70)
    where we try to keep the spatial dimension the same since they are position clues
"""

import torch
import torch.nn as nn


class head_occupancy(nn.Module):
    def __init__(self, cfg):
        super(head_occupancy, self).__init__()
        self.cfg = cfg
        self.h_resolution = int(round(self.cfg.grid_size[1]/self.cfg.grid_unc[1]))
        self.out_ch = 1
            
        self.branch_in_ch = 256
        self.branch_out_ch = 32
            
        self.voxel_conv3d_branch1 = nn.Sequential(nn.Conv3d(self.branch_in_ch, 128, kernel_size=(3, 3, 3), stride=(1, 2, 1), padding=(1, 1, 1)),
                                        nn.BatchNorm3d(128),
                                        nn.ReLU(),
                                        nn.Conv3d(128, self.branch_out_ch, kernel_size=(3, 3, 3), stride=(1, 2, 1), padding=(1, 1, 1)),
                                        nn.BatchNorm3d(self.branch_out_ch),
                                        nn.ReLU())
        self.voxel_conv3d_branch2 = nn.Sequential(nn.Conv3d(self.branch_in_ch, 128, kernel_size=(5, 5, 5), stride=(1, 2, 1), padding=(2, 2, 2)),
                                        nn.BatchNorm3d(128),
                                        nn.ReLU(),
                                        nn.Conv3d(128, self.branch_out_ch, kernel_size=(5, 5, 5), stride=(1, 2, 1), padding=(2, 2, 2)),
                                        nn.BatchNorm3d(self.branch_out_ch),
                                        nn.ReLU())
        self.conv2d_merge = nn.Sequential(nn.Conv2d(self.branch_out_ch, self.branch_out_ch, kernel_size=3, stride = 1, padding=1),
                                          nn.BatchNorm2d(self.branch_out_ch),
                                          nn.ReLU(),
                                          nn.Conv2d(self.branch_out_ch, self.out_ch, kernel_size=3, stride = 1, padding=1))
        
        
    def forward(self, x):
        x1 = self.voxel_conv3d_branch1(x)
        x1 = torch.max(x1, dim=3, keepdim=True).values
        
        x2 = self.voxel_conv3d_branch2(x)
        x2 = torch.max(x2, dim=3, keepdim=True).values
        
        x_merge = torch.cat([x1, x2], dim=3)
        
        x_merge = torch.max(x_merge, dim = 3, keepdim=False).values
        
        out = self.conv2d_merge(x_merge)
        # TODO: what is this output shape? what will be the grid???
        # solved, output is in 2D grid W, L (100, 70)
        return out
            
    
class head_box_2d_bev(nn.Module):
    def __init__(self, cfg):
        super(head_box_2d_bev, self).__init__()
        self.cfg = cfg
        self.h_resolution = int(round(self.cfg.grid_size[1]/self.cfg.grid_unc[1]))
        self.out_ch = self.cfg.model.num_class + 5 + 1 # with confidence score
            
        self.branch_in_ch = 256
        self.branch_out_ch = 32
            
        self.voxel_conv3d_branch1 = nn.Sequential(nn.Conv3d(self.branch_in_ch, 128, kernel_size=(3, 3, 3), stride=(1, 2, 1), padding=(1, 1, 1)),
                                        nn.BatchNorm3d(128),
                                        nn.ReLU(),
                                        nn.Conv3d(128, self.branch_out_ch, kernel_size=(3, 3, 3), stride=(1, 2, 1), padding=(1, 1, 1)),
                                        nn.BatchNorm3d(self.branch_out_ch),
                                        nn.ReLU())
        self.voxel_conv3d_branch2 = nn.Sequential(nn.Conv3d(self.branch_in_ch, 128, kernel_size=(5, 5, 5), stride=(1, 2, 1), padding=(2, 2, 2)),
                                        nn.BatchNorm3d(128),
                                        nn.ReLU(),
                                        nn.Conv3d(128, self.branch_out_ch, kernel_size=(5, 5, 5), stride=(1, 2, 1), padding=(2, 2, 2)),
                                        nn.BatchNorm3d(self.branch_out_ch),
                                        nn.ReLU())
        self.conv2d_merge = nn.Sequential(nn.Conv2d(self.branch_out_ch, self.branch_out_ch, kernel_size=3, stride = 1, padding=1),
                                          nn.BatchNorm2d(self.branch_out_ch),
                                          nn.ReLU(),
                                          nn.Conv2d(self.branch_out_ch, self.out_ch, kernel_size=3, stride = 1, padding=1))
        
        
    def forward(self, x):
        x1 = self.voxel_conv3d_branch1(x)
        x1 = torch.max(x1, dim=3, keepdim=True).values
        
        x2 = self.voxel_conv3d_branch2(x)
        x2 = torch.max(x2, dim=3, keepdim=True).values
        
        x_merge = torch.cat([x1, x2], dim=3)
        
        x_merge = torch.max(x_merge, dim = 3, keepdim=False).values
        
        out = self.conv2d_merge(x_merge)
        # TODO: what is this output shape? what will be the grid???
        # solved, output is in 2D grid W, L (100, 70)
        return out
        
        
