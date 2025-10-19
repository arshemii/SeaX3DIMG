#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note: all the inputs to these modules are in shape:
    (1, 256, 100, 30, 70)
    where we try to keep the spatial dimension the same since they are position clues
"""

import torch
import torch.nn as nn

        
        
class head_occupancy_bev(nn.Module):
    def __init__(self, cfg):
        super(head_occupancy_bev, self).__init__()
        self.cfg = cfg
        self.h_resolution = int(round(self.cfg.grid_size[1]/self.cfg.grid_unc[1]))
        self.out_ch = 1
        self.branch_in_ch = 256
        self.branch_out_ch = 32

        self.voxel_conv3d_branch1 = nn.Sequential(
            nn.Conv3d(self.branch_in_ch, 128, kernel_size=(3,3,3), stride=(1,2,1), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, self.branch_out_ch, kernel_size=(3,3,3), stride=(1,2,1), padding=(1,1,1)),
            nn.BatchNorm3d(self.branch_out_ch),
            nn.ReLU()
        )
        self.voxel_conv3d_branch2 = nn.Sequential(
            nn.Conv3d(self.branch_in_ch, 128, kernel_size=(5,5,5), stride=(1,2,1), padding=(2,2,2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, self.branch_out_ch, kernel_size=(5,5,5), stride=(1,2,1), padding=(2,2,2)),
            nn.BatchNorm3d(self.branch_out_ch),
            nn.ReLU()
        )
        # learned height aggregation: a conv that maps along height axis to compute per-height weights
        # We will use a 1x1x1 conv to produce a scalar weight per height slice
        self.height_weight_conv = nn.Conv3d(self.branch_out_ch, 1, kernel_size=(3,1,1), padding=(1,0,0))
        nn.init.kaiming_normal_(self.height_weight_conv.weight, nonlinearity='relu')
        self.conv2d_merge = nn.Sequential(
            nn.Conv2d(self.branch_out_ch, self.branch_out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.branch_out_ch),
            nn.ReLU(),
            nn.Conv2d(self.branch_out_ch, self.out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x: (B, C, X, Y, Z) where Y is height dimension index 3 in original (after reshape)
        x1 = self.voxel_conv3d_branch1(x)  # (B, branch_out_ch, X, H', Z)
        x2 = self.voxel_conv3d_branch2(x)  # same dims

        # compute height weights for x1 and x2
        w1 = torch.sigmoid(self.height_weight_conv(x1))  # (B,1,X,H',Z)
        w2 = torch.sigmoid(self.height_weight_conv(x2))

        # weighted sum along height (dim=3)
        x1_weighted = (x1 * w1).sum(dim=3, keepdim=True)  # (B,branch_out_ch, X,1,Z)
        x2_weighted = (x2 * w2).sum(dim=3, keepdim=True)

        # merge along height dimension (now single slice)
        x_merge = torch.cat([x1_weighted, x2_weighted], dim=3)  # (B,branch_out_ch, X,2,Z)  (we will collapse)
        x_merge = x_merge.sum(dim=3)  # (B,branch_out_ch, X, Z)

        # convert to 2D conv input: (B, C, W, L) expecting channels first
        out = self.conv2d_merge(x_merge)
        return out


class head_box_bev(head_occupancy_bev):
    def __init__(self, cfg):
        super(head_box_bev, self).__init__(cfg)
        # override out_ch to match box dim
        self.out_ch = self.cfg.model.num_class + 6  # num_class + 5 box params + 1 confidence
        self.cfg = cfg
        self.h_resolution = int(round(self.cfg.grid_size[1]/self.cfg.grid_unc[1]))
        self.branch_in_ch = 256
        self.branch_out_ch = 32

        self.voxel_conv3d_branch1 = nn.Sequential(
            nn.Conv3d(self.branch_in_ch, 128, kernel_size=(3,3,3), stride=(1,2,1), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, self.branch_out_ch, kernel_size=(3,3,3), stride=(1,2,1), padding=(1,1,1)),
            nn.BatchNorm3d(self.branch_out_ch),
            nn.ReLU()
        )
        self.voxel_conv3d_branch2 = nn.Sequential(
            nn.Conv3d(self.branch_in_ch, 128, kernel_size=(5,5,5), stride=(1,2,1), padding=(2,2,2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, self.branch_out_ch, kernel_size=(5,5,5), stride=(1,2,1), padding=(2,2,2)),
            nn.BatchNorm3d(self.branch_out_ch),
            nn.ReLU()
        )
        # rebuild conv2d_merge to output right number of channels
        self.conv2d_merge = nn.Sequential(
            nn.Conv2d(self.branch_out_ch, self.branch_out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.branch_out_ch),
            nn.ReLU(),
            nn.Conv2d(self.branch_out_ch, self.out_ch, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        # x: (B, C, X, Y, Z) where Y is height dimension index 3 in original (after reshape)
        x1 = self.voxel_conv3d_branch1(x)  # (B, branch_out_ch, X, H', Z)
        x2 = self.voxel_conv3d_branch2(x)  # same dims

        # compute height weights for x1 and x2
        w1 = torch.sigmoid(self.height_weight_conv(x1))  # (B,1,X,H',Z)
        w2 = torch.sigmoid(self.height_weight_conv(x2))

        # weighted sum along height (dim=3)
        x1_weighted = (x1 * w1).sum(dim=3, keepdim=True)  # (B,branch_out_ch, X,1,Z)
        x2_weighted = (x2 * w2).sum(dim=3, keepdim=True)

        # merge along height dimension (now single slice)
        x_merge = torch.cat([x1_weighted, x2_weighted], dim=3)  # (B,branch_out_ch, X,2,Z)  (we will collapse)
        x_merge = x_merge.sum(dim=3)  # (B,branch_out_ch, X, Z)

        # convert to 2D conv input: (B, C, W, L) expecting channels first
        out = self.conv2d_merge(x_merge)
        return out