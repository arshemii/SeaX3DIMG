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
        self.num_voxels = self.cfg.grid_resolution[0] * self.cfg.grid_resolution[1] * self.cfg.grid_resolution[2] # 153600
        
        self.oob_mask_flat = self.cfg.oob_mask_flat[0]
        self.grid_flat_filtered = self.cfg.grid_flat_filtered[0].unsqueeze(0).to(self.device) # torch.Size([1, 104031, 1, 2])
        
        self.backbone = self.feature_net()
        
        if self.cfg.loss.heads != ['disp']:
            print("The head is initialized with disparity + other outputs!")
            self._init_head()
            
        self._init_matching_layers()
        self.disp_feat_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.weighted_refine = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        

        self.bn_match_1 = nn.BatchNorm2d(num_features=32)
        self.conv2d_match = nn.Conv2d(in_channels=32, out_channels=64,
                                      kernel_size=3, padding=1, bias=False)
        self.bn_match_2 = nn.BatchNorm2d(num_features=64)
        self.relu_matching = nn.ReLU()        
        
        self.conv_3dvoxel = nn.Sequential(nn.Conv3d(80, self.cfg.model.head.inplanes, 3, 1, padding=1, bias=False),
                                      nn.GroupNorm(num_groups=8, num_channels=self.cfg.model.head.inplanes),
                                      nn.ReLU(inplace=True))


    def feature_net(self):
        if self.cfg.model.back.name == 'hrnet-w48':
            feat_net = get_pose_net(self.cfg, self.is_train_backbone)
            feat_net.to(self.device)
        else:
            raise NotImplementedError("Must implement resnet with output of shape (1, 48, 128, 128)")
        return feat_net

    def _init_matching_layers(self):
        self.hrnet_disp = Level(self.cfg.model.hrnet_cout,
                              self.cfg.model.max_disp, 1)

    def _init_head(self):
        if len(self.cfg.loss.heads) > 1:
            self.head = HD.head_3d_detection(self.cfg)
        else:
            self.head = None
    
    def matching_module(self, feature_l, feature_r):
        """
        feature_l, feature_r:
             Intermediate outputs of HRNet (B, 48, h/4, w/4)
             
        """
        # Level.forward now returns (h_out, cv_ref, disp, w)
        _, _, disp, _, conf = self.hrnet_disp(feature_l, feature_r)
        
        # TODO: should it be normalized?
        

        disp_up = F.interpolate(disp, size=(feature_l.shape[2], feature_l.shape[3]),
                                mode='bilinear', align_corners=True)                # (B,1,h/4,w/4)
        conf_up = F.interpolate(conf, size=(feature_l.shape[2], feature_l.shape[3]),
                                mode='bilinear', align_corners=True)

        disp_feat = self.disp_feat_conv(disp_up)  # (B,32,h/4,w/4)
        
        if not self.cfg.model.conf_voxel:
            weighted_disp = disp_feat * conf_up # (B,32,h/4,w/4)
        else:
            weighted_disp = disp_feat

        weighted_disp = self.weighted_refine(weighted_disp)  # (B, 32, h/4, w/4)
    
        weighted_disp = self.bn_match_1(weighted_disp)
        weighted_disp = self.conv2d_match(weighted_disp)  # ch = 64
        weighted_disp = self.bn_match_2(weighted_disp)
        weighted_disp = self.relu_matching(weighted_disp)   # (B, 64, h/4, w/4)
    
        return weighted_disp, disp_up
    
    def _voxel_filler(self, voxel, valid_indices):
        """
        voxel: (B, C, N_valid, 1) or (B, C, N_valid)
        valid_indices: (N_valid,) — positions inside full voxel grid
        Returns:
           full_voxel: (B, C, num_voxels)
        """
    
        # ---- Normalize voxel shape ----
        if voxel.dim() == 4 and voxel.size(-1) == 1:
            voxel = voxel.squeeze(-1)   # (B, C, N_valid)
    
        B, C, N = voxel.shape
        device = voxel.device

    
        # ---- Initialize full voxel volume with zeros (OOB = zero semantics) ----
        full_voxel = torch.zeros(
            (B, C, self.num_voxels),
            dtype=voxel.dtype,
            device=device
        )
    
        # ---- Direct assignment: each valid voxel index gets exactly 1 feature ----
        # No scatter_add needed because no two features share the same index.
        full_voxel[:, :, valid_indices] = voxel
    
        return full_voxel
    
    
    def voxelizer(self, tensor, base_feat):
        """
        Inputs:
            tensor:      (B, 32, h/4, w/4) --> matched_tensor
            base_feat:   (B,48,h/4, w/4)
        Returns: full_voxel reshaped to (B, C, X, Y, Z)
        """
        B, C, Hf, Wf = tensor.shape
        _, C_base, _,  _ = base_feat.shape
    
        # grid_flat_filtered shape: (1, N_valid, 1, 2)
        grid_batched = self.grid_flat_filtered.expand(B, -1, -1, -1)  # (B, N_valid, 1, 2)

        sampled_matched = F.grid_sample(tensor, grid_batched,
                                     mode='bilinear', align_corners=True)  # (B, 32, N_valid, 1)
        
        sampled_base = F.grid_sample(base_feat, grid_batched,
                                     mode='bilinear', align_corners=True)  # (B, 48, N_valid, 1)
    
        sampled = torch.cat([sampled_matched, sampled_base], dim=1)  # chanells = 48 + 32 = 80
        
        del grid_batched
        valid_indices = self.oob_mask_flat.nonzero(as_tuple=False).squeeze(1).to(tensor.device)  # (N_valid,) 104031
    
        full_voxel_flat = self._voxel_filler(sampled, valid_indices)  # (B,80,num_voxels)
    
        full_voxel = full_voxel_flat.reshape(B, C + C_base, self.cfg.grid_resolution[0],
                                             self.cfg.grid_resolution[1],
                                             self.cfg.grid_resolution[2])
    
        
        del full_voxel_flat
        return full_voxel
        
    def forward(self, img_l, img_r):
        """
        img_l and img_r:    Shape (B, 3, 288, 960) --> h=288, w=960
        mem_left:           shape (B, 3, X, Y, Z)

        """    
                
        # Feature extraction from each image
        left_f, left_f_inter = self.backbone(img_l)     # shape for outputs: (1, 48, h/4, h/4)
        _, right_f_inter = self.backbone(img_r)         # shape for outputs: (1, 48, h/4, h/4)

        # matching stage
        # (B, 64, h/4, w/4), (B, 1, h/4, w/4)
        matched_tensor, disp_upsampled = self.matching_module(left_f_inter, right_f_inter)
        
        #disp_upsampled = F.softplus(disp_upsampled)
        
        if self.cfg.loss.heads == ['disp']:
            return disp_upsampled

        voxel = self.voxelizer(matched_tensor, left_f) # (B, 80, X, Y, Z)
        
        del matched_tensor, left_f_inter, right_f_inter
        
        voxel = self.conv_3dvoxel(voxel)  # increase channels
        out = self.head(voxel) # 5 tensors
        
        return out, disp_upsampled

  
def load_weights_from_checkpoint(model, checkpoint_path, device):
    if checkpoint_path is not None:        
        ckpt = torch.load(checkpoint_path, map_location=device)
        if 'model_state' in ckpt:
            print("The provided checkpoint does have model_state key !")
            model.load_state_dict(ckpt['model_state'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

        print(f"==> Loaded whole SX3D model checkpoint from: {checkpoint_path}")
    else:
        print("==> No checkpoint provided. Training from scratch or initializing backbone only.")


def get_SX3D_model(cfg, is_train=True):
    
    is_train_backbone = is_train and cfg.model.back.init_weight
    
    model = SX3DIMG(cfg, is_train_backbone=is_train_backbone)
    
    # Always try to load full model checkpoint if provided
    if cfg.model.sx3d.use_checkpoint and cfg.model.sx3d.checkpoint_exp:
        load_weights_from_checkpoint(model, cfg.model.sx3d.checkpoint_exp, cfg.device[0])
        
        if cfg.loss.freeze:
            print("No grad for frozen layers ...")
            if 'disp' in cfg.loss.freezed_output:
                for p in model.hrnet_disp.parameters():
                    p.requires_grad = False
    
    return model
    

    
    
    
    
    
    
    
    
    
    
    
