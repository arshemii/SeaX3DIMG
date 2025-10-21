#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note: all the inputs to these modules are in shape:
    (1, 256, 100, 30, 70)
    where we try to keep the spatial dimension the same since they are position clues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def convbn_3d(in_planes, 
              out_planes, 
              kernel_size, 
              stride, 
              pad, 
              dilation=1,
              gn=False, 
              groups=32
              ):
    return nn.Sequential(nn.Conv3d(in_planes, 
                                   out_planes, 
                                   kernel_size=kernel_size, 
                                   padding=pad, 
                                   dilation=dilation,
                                   stride=stride,
                                   bias=False
                                   ),
                         nn.BatchNorm3d(out_planes) if not gn else nn.GroupNorm(groups, out_planes)
                         )

class hourglass(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, 
                                             inplanes*2, 
                                             kernel_size=3, 
                                             stride=2, 
                                             pad=1, 
                                             gn=gn
                                             ),
                                   nn.ReLU(inplace=True)
                                   )
        
        self.conv2 = convbn_3d(inplanes*2, 
                               inplanes*2, 
                               kernel_size=3, 
                               stride=1, 
                               pad=1, 
                               gn=gn
                               )

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, 
                                             inplanes*2, 
                                             kernel_size=3, 
                                             stride=2, 
                                             pad=1, 
                                             gn=gn
                                             ),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, 
                                             inplanes*2, 
                                             kernel_size=3, 
                                             stride=1, 
                                             pad=1, 
                                             gn=gn
                                             ),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, 
                                                      inplanes*2, 
                                                      kernel_size=3, 
                                                      padding=1, 
                                                      output_padding=1, 
                                                      stride=2,
                                                      bias=False
                                                      ),
                                   nn.BatchNorm3d(inplanes*2) if not gn else nn.GroupNorm(32, inplanes * 2)
                                   )  # +conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, 
                                                      inplanes, 
                                                      kernel_size=3, 
                                                      padding=1, 
                                                      output_padding=1, 
                                                      stride=2,
                                                      bias=False
                                                      ),
                                   nn.BatchNorm3d(inplanes) if not gn else nn.GroupNorm(32, inplanes)
                                   )  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post
        
class ConvGRUCell2D(nn.Module):
    """Small ConvGRU cell for BEV-level temporal aggregation."""
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2,
                                 kernel_size=kernel_size, padding=padding, bias=True)
        self.conv_h = nn.Conv2d(input_channels + hidden_channels, hidden_channels,
                                kernel_size=kernel_size, padding=padding, bias=True)
        self.activation = nn.ReLU()

    def forward(self, x, h):
        # x: (B, Cx, H, W), h: (B, Ch, H, W)
        if h is None:
            h = torch.zeros(x.size(0), self.conv_h.out_channels, x.size(2), x.size(3),
                            device=x.device, dtype=x.dtype)
        combined = torch.cat([x, h], dim=1)
        zr = self.conv_zr(combined)
        z, r = torch.split(zr, zr.size(1) // 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        combined2 = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined2))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class LightweightMemoryModule(nn.Module):
    """
    Memory module:
      - encode voxel -> BEV
      - update BEV with ConvGRU (optionally warp previous BEV)
      - decode BEV -> voxel-shaped memory to concat with voxel features
    """

    def __init__(self, voxel_shape, voxel_feat_channels, mem_bev_channels=32, mem_voxel_channels=32):
        """
        Args:
          voxel_shape: tuple (X, Y, Z) grid dims
          voxel_feat_channels: channels of voxel input (e.g., 256)
          mem_bev_channels: small channel dimension for BEV memory
          mem_voxel_channels: channels for voxel memory that will be concatenated
        """
        super().__init__()
        self.X, self.Y, self.Z = voxel_shape  # ordering used in your code
        self.mem_bev_ch = mem_bev_channels
        self.mem_voxel_ch = mem_voxel_channels

        # BEV encoder: compress voxel -> BEV (reduce channels then spatial conv)
        # Input: voxel.mean(dim=4) -> (B, C, X, Y)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(voxel_feat_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, self.mem_bev_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.mem_bev_ch),
            nn.ReLU()
        )

        # Temporal aggregator: ConvGRU on BEV
        self.gru = ConvGRUCell2D(input_channels=self.mem_bev_ch, hidden_channels=self.mem_bev_ch, kernel_size=3)

        # BEV -> voxel decoder: small conv to lift BEV to mem_voxel_ch, then expand in Z
        self.bev_to_voxel = nn.Sequential(
            nn.Conv2d(self.mem_bev_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.mem_voxel_ch, kernel_size=1, bias=True)
        )

        # Optional small 3D refinement conv (lightweight)
        self.voxel_refine = nn.Sequential(
            nn.Conv3d(self.mem_voxel_ch, self.mem_voxel_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.mem_voxel_ch),
            nn.ReLU()
        )

    def encode_bev(self, voxel):
        # voxel: (B, C, X, Y, Z)
        # project along Z axis (height) -> BEV
        bev = voxel.mean(dim=4)     # (B, C, X, Y)
        bev = self.bev_encoder(bev) # (B, mem_bev_ch, X, Y)
        return bev

    def decode_to_voxel_memory(self, bev):
        # bev: (B, mem_bev_ch, X, Y)
        out = self.bev_to_voxel(bev)  # (B, mem_voxel_ch, X, Y)
        # expand to voxel by repeating along Z dimension (cheap). Then optional 3D refine.
        out = out.unsqueeze(-1).expand(-1, -1, -1, -1, self.Z).contiguous()  # (B, mem_voxel_ch, X, Y, Z)
        out = self.voxel_refine(out)  # small 3D conv to add cross-Z coupling
        return out

    def forward_create(self, voxel):
        """
        Create initial memory from single voxel input.
        Returns: voxel_memory (B, mem_voxel_ch, X, Y, Z) and bev_memory (B, mem_bev_ch, X, Y)
        """
        bev = self.encode_bev(voxel)
        # initial hidden state is bev itself (or zeros); use GRU with None to init
        h_new = self.gru(bev, None)  # (B, mem_bev_ch, X, Y)
        voxel_mem = self.decode_to_voxel_memory(h_new)  # voxel-shaped memory
        return voxel_mem, h_new

    def update(self, prev_bev, voxel, ego_warp_fn=None):
        """
        Update memory given previous BEV state and current voxel.
        Args:
          prev_bev: (B, mem_bev_ch, X, Y) or None
          voxel: current voxel (B, C, X, Y, Z)
          ego_warp_fn: optional function(prev_bev)->warped_prev_bev to account for ego motion
        Returns:
          voxel_memory (B, mem_voxel_ch, X, Y, Z), new_bev (B, mem_bev_ch, X, Y)
        """
        if prev_bev is not None and ego_warp_fn is not None:
            prev_bev = ego_warp_fn(prev_bev)  # user-provided warp (identity if not provided)

        bev_curr = self.encode_bev(voxel)  # encode current
        # fuse current bev and previous bev (concatenate then small conv via GRU)
        # we use GRU where input is bev_curr and hidden state prev_bev
        new_bev = self.gru(bev_curr, prev_bev)
        voxel_mem = self.decode_to_voxel_memory(new_bev)
        return voxel_mem, new_bev

        
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