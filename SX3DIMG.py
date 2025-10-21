#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 19:53:22 2025

@author: arash
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.pose_hrnet import get_pose_net
from modules.matching import Level
from modules import heads as HD

class SX3DIMG(nn.Module):
    def __init__(self, cfg, is_train_backbone):
        super(SX3DIMG, self).__init__()
        self.cfg = cfg
        self.is_train_backbone = is_train_backbone
        self.device = self.cfg.device[0]
        
        self.h, self.w = self.cfg.model.in_size
        self.grid = cfg.grid[0].permute(3,0,1,2).to(dtype=torch.float32)
        self.num_voxels = self.cfg.grid_resolution[0] * self.cfg.grid_resolution[1] * self.cfg.grid_resolution[2]
        
        self.oob_mask_flat = self.cfg.oob_mask_valid[0]
        self.grid_flat_filtered = self.cfg.grid_flat_filtered[0].unsqueeze(0).to(self.device)
        
        self.backbone = self.feature_net()
        self._init_head()
        self._init_matching_layers()
        self.disp_feat_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.match_reduce = nn.Conv2d(80, 128, kernel_size=1, bias=False)
        
        
        self.conv2d_memory = nn.Conv2d(in_channels=48, out_channels=3,
                                      kernel_size=5, padding=2, bias=False)
        self.bn_memory = nn.BatchNorm2d(num_features=3)
        self.upsample_disp = nn.ConvTranspose2d(in_channels=17, out_channels=32,
                                                kernel_size=4, stride=2, padding=1)
        self.downsample_feat = nn.Conv2d(in_channels=51, out_channels=96,
                                         kernel_size=3, stride=2, padding=1)
        self.bn_match_1 = nn.BatchNorm2d(num_features=128)
        self.conv2d_match = nn.Conv2d(in_channels=128, out_channels=256,
                                      kernel_size=3, padding=1, bias=False)
        self.bn_match_2 = nn.BatchNorm2d(num_features=256)
        self.relu_create_mem = nn.ReLU()
        self.relu_matching = nn.ReLU()


    def feature_net(self):
        if self.cfg.model.back.name == 'hrnet-w48':
            feat_net = get_pose_net(self.cfg, self.is_train_backbone)
            feat_net.to(self.device)
        else:
            raise NotImplementedError("Must implement resnet with output of shape (1, 48, 128, 128)")
        if self.logs:
            logger.info("Feature extraction network has initialized.")
        return feat_net

    def _init_matching_layers(self):
        self.hrnet_disp = Level(self.cfg.model.hrnet_cout,
                              self.cfg.model.max_disp, 1)

    def _init_head(self):
        self.head = HD.head_3d_detection(self.cfg)

    def create_memory_forward(self, voxel):
        # TODO: write a function to cteare memory
        return memory
    
    def matching_module(self, feature_l, feature_r, base):
        """
        feature_l, feature_r:
             Intermediate outputs of HRNet (B, 48, h/4, w/4)
             
        base:
            Final output of the HRNet (B, 48, h/4, w/4)
        """
        # Level.forward now returns (h_out, cv_ref, disp, w)
        _, _, disp, conf = self.hrnet_disp(feature_l, feature_r)

        disp_up = F.interpolate(disp, size=(feature_l.shape[2], feature_l.shape[3]),
                                mode='bilinear', align_corners=True)                # (B,1,h/4,w/4)
        conf_up = F.interpolate(conf, size=(feature_l.shape[2], feature_l.shape[3]),
                                mode='bilinear', align_corners=True)

        disp_feat = self.disp_feat_conv(disp_up)  # (B,32,h/4,w/4)
   
        # Concatenate base, disp_feat
        combined = torch.cat([base, disp_feat], dim=1)  # (B, 32 + 48, h/4, w/4)

        combined = self.match_reduce(combined)  # (B, 128, h/4, w/4)
    
        base_down = self.bn_match_1(combined)
        base_down = self.conv2d_match(base_down)
        base_down = self.bn_match_2(base_down)
        base_down = self.relu_matching(base_down)   # (B, 256, h/4, w/4)
    
        return base_down, disp_up, conf_up
    
    def _voxel_filler(self, voxel, valid_indices, conf_for_points=None):
        """
        voxel: sampled features from grid_sample -> shape: (B, C, N_valid, 1) or (B, C, N_valid)
        valid_indices: 1D LongTensor indices into full voxel flatten position (N_valid,)
        conf_for_points: (B,1,N_valid) confidence for each sampled point (optional)
        Returns:
           full_voxel: (B, C, total_num_voxels) with averaged contributions
        """
        B = voxel.size(0)
        C = voxel.size(1)
        N_valid = voxel.size(2)  # number of valid sampling points
    
        # prepare full tensor and a weight counter
        full_voxel = torch.zeros((B, C, self.num_voxels), dtype=voxel.dtype, device=voxel.device)
        weight_accum = torch.zeros((B, 1, self.num_voxels), dtype=voxel.dtype, device=voxel.device)
    
        # voxel currently shape (B, C, N_valid, 1) sometimes; squeeze last dim if present
        if voxel.dim() == 4 and voxel.size(-1) == 1:
            voxel = voxel.squeeze(-1)  # now (B,C,N_valid)
    
        # prepare confidence weights (default to 1)
        if conf_for_points is None:
            # shape (B,1,N_valid) with ones
            conf_for_points = torch.ones((B, 1, N_valid), dtype=voxel.dtype, device=voxel.device)
        else:
            # ensure shape (B,1,N_valid)
            if conf_for_points.dim() == 2:
                conf_for_points = conf_for_points.unsqueeze(1)
    
        # scatter_add the weighted features and weights
        # we use scatter_add along the last dimension via index expansion
        # full_voxel[b, :, idx] += voxel[b, :, :]*conf[b,0,:]
        # vectorize using scatter_add:
        # expand valid_indices to shape (B, C, N_valid) for feature scattering
        idx = valid_indices.view(1, 1, -1).expand(B, C, -1)  # (B, C, N_valid)
        # weights repeated to match channels
        w = conf_for_points.expand(B, C, -1)  # (B, C, N_valid)
    
        # weighted features
        weighted_feats = voxel * w  # (B, C, N_valid)
    
        # scatter_add into full_voxel
        full_voxel = full_voxel.scatter_add(dim=2, index=idx, src=weighted_feats)
    
        # scatter_add weights into weight_accum
        idx_w = valid_indices.view(1, 1, -1).expand(B, 1, -1)
        weight_accum = weight_accum.scatter_add(dim=2, index=idx_w, src=conf_for_points)
    
        # avoid division by zero: mask where weight_accum == 0
        nonzero = weight_accum > 0
        # normalize features
        full_voxel = full_voxel / (weight_accum + (1.0 - nonzero.float()))  # zeros remain zero where weight_accum is zero
    
        return full_voxel  # shape (B,C,num_voxels)
    
    def voxelizer(self, tensor, conf_map=None):
        """
        Inputs:
            tensor:     (B, 256, h/4, w/4) --> matched_tensor
            conf_map:   (B,1,h/4, w/4)
        Returns: full_voxel reshaped to (B, C, X, Y, Z)
        """
        B, C, Hf, Wf = tensor.shape
    
        # grid_flat_filtered shape: (1, N_valid, 1, 2)
        grid_batched = self.cfg.grid_flat_filtered[0].expand(B, -1, -1, -1)  # (B, N_valid, 1, 2)

        sampled = F.grid_sample(tensor, grid_batched, mode='bilinear', align_corners=True)  # (B, C, N_valid, 1)

        if conf_map is not None:
            conf_sampled = F.grid_sample(conf_map, grid_batched, mode='bilinear', align_corners=True)  # (B,1,N_valid,1)
            conf_sampled = conf_sampled.squeeze(-1)  # (B,1,N_valid)
        else:
            conf_sampled = None
    
        valid_indices = self.cfg.oob_mask_flat[0].nonzero(as_tuple=False).squeeze(1).to(tensor.device)  # (N_valid,)
    
        full_voxel_flat = self._voxel_filler(sampled, valid_indices, conf_for_points=conf_sampled)  # (B,256,num_voxels)
    
        full_voxel = full_voxel_flat.reshape(B, C, self.grid_resolution[0], self.grid_resolution[1], self.grid_resolution[2])
    
        return full_voxel
        
    def forward(self, img_l, img_r, memory, create_memory = False):
        """
        img_l and img_r:    Shape (B, 3, 288, 960) --> h=288, w=960
        mem_left:           ?????????

        """            
        # Feature extraction from each image
        left_f, left_f_inter = self.backbone(img_l)     # shape for outputs: (1, 48, h/4, h/4)
        _, right_f_inter = self.backbone(img_r)         # shape for outputs: (1, 48, h/4, h/4)

        # matching stage
        # (B, 256, h/4, w/4), (B, 1, h/4, w/4), (B, 1, h/4, w/4)
        matched_tensor, disp_upsampled, conf_upsampled = self.matching_module(left_f_inter,
                                                                              right_f_inter,
                                                                              left_f)
        if not self.cfg.model.conf_voxel:
            conf_upsampled = None    
        voxel = self.voxelizer(matched_tensor, conf_upsampled) # (B, 256, X, Y, Z)
        forward_mem = create_memory_forward(voxel)

        if create_memory:
            return forward_mem
        else:
            assert memory is not None
            voxel = torch.cat([voxel, memory], dim=1)
            out = self.head(voxel)
            if self.cfg.loss.aux_loss:
                return out, disp_upsampled, forward_mem
            else:
                return out, forward_mem
        
def load_weights_from_checkpoint(model, checkpoint_path, logger=None):
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        if logger:
            logger.info(f"==> Loaded model checkpoint from: {checkpoint_path}")
        else:
            print(f"==> Loaded model checkpoint from: {checkpoint_path}")
    else:
        if logger:
            logger.info("==> No checkpoint provided. Training from scratch or initializing backbone only.")
        else:
            print("==> No checkpoint provided. Training from scratch or initializing backbone only.")


def get_SX3D_model(cfg, is_train=True, logger = logger):
    
    is_train_backbone = is_train and cfg.model.back.init_weight
    
    model = SX3DIMG(cfg, is_train_backbone=is_train_backbone)
    
    # Always try to load full model checkpoint if provided
    if cfg.model.sx3d.use_checkpoint and cfg.model.sx3d.checkpoint:
        load_weights_from_checkpoint(model, cfg.model.sx3d.checkpoint, logger)
    
    return model
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
