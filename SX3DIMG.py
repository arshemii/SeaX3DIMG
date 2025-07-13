#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 19:53:22 2025

@author: arash
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import transforms
from modules.pose_hrnet import get_pose_net
from modules.matching import Level
from utils.grid_generator import GridGenerator,cam_to_img, grid_for_sample, oob_voxels
from modules import heads as HD
from utils.debug_util import setup_logger

logger = setup_logger('SX3DIMG_logs', './logging_dir')

class SX3DIMG(nn.Module):
    def __init__(self, cfg, is_train_backbone):
        super(SX3DIMG, self).__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.logs = self.cfg.logging
        # self.logger = setup_logger('SX3DIMG_logs', self.cfg.log_dir)
        self.is_train_backbone = is_train_backbone
        
        self.device = self.cfg.device[0]
        
        self.h, self.w = self.cfg.model.in_size
        
        self.grid_obj = GridGenerator(self.cfg.grid_size, self.cfg.grid_unc) # points in cam coordinates
        self.grid = self.grid_obj.get_grid()['grid'].to(dtype=torch.float32)
        self.grid = self.grid.to(self.device)
        self.grid_resolution = tuple(int(round(size / res)) for size, res in zip(self.cfg.grid_size, self.cfg.grid_unc))
        
        
        if self.cfg.camera.P_l is not None:
            self.P_l = cfg.camera.P_l[0].to(self.device)
            self.grid_img = cam_to_img(self.grid, self.P_l)
            self.oob_mask = oob_voxels(self.grid_img, self.cfg.model.in_size).to(self.device)
            self.oob_mask_valid = ~self.oob_mask
            self.oob_mask_flat = self.oob_mask_valid.view(-1)
            self.grid_flat = grid_for_sample(self.grid_img, (self.h, self.w))
            self.grid_flat_filtered = self.grid_flat[0][self.oob_mask_flat]
            self.grid_flat_filtered = self.grid_flat_filtered.unsqueeze(0).to(self.device)
        
        self.backbone = self.feature_net()
        
        if self.cfg.model.head == 'box2d':
            self._init_2d_head()
        elif self.cfg.model.head == 'box3d':
            self._init_3d_head()
        else:
            raise NotImplementedError("other representation ehad methods!")
        
        self._init_matching_layers()

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

        
    def create_memory(self, img_l):
        left_f_mem = self.backbone(img_l)
        
        feat_l = self.conv2d_memory(left_f_mem)
        
        feat_l = self.bn_memory(feat_l)
        
        feat_l = self.relu_create_mem(feat_l)
        
        return feat_l
    
    def matching_module(self, feature_l, feature_r, base):
        # Adding disparity clues includes:
            # disparity hypithesis
            # confidence
        h, _, _, w = self.hrnet_disp(feature_l, feature_r) # 1, 16 + 1, h/4, w/4
        # upsample and downsample
        disp_up = self.upsample_disp(torch.cat([h, w], dim = 1)) # 32 ch
        base_down = self.downsample_feat(base)  # 96 ch
        
        base_down = torch.cat([base_down, disp_up], dim = 1) # 128
        base_down = self.bn_match_1(base_down)
        
        assert base_down.shape[2] == int(self.h/8)
        
        base_down = self.conv2d_match(base_down)
        base_down = self.bn_match_2(base_down)
        base_down = self.relu_matching(base_down)
        
        return base_down
    
    def _voxel_filler(self, voxel):
        # recunstruct 3d feature map
        # voxel of shape torch.Size([1, 256, num_valid_points, 1])
        # Prepare empty tensor
        full_voxel = torch.zeros((1, 256, 100 * 30 * 70), dtype=voxel.dtype, device=voxel.device)
        
        # Remove batch and last dim → shape: [256, num_valid_points]
        voxel_squeezed = voxel.squeeze(0).squeeze(-1)
        
        # Insert into valid positions
        valid_indices = self.oob_mask_flat.nonzero(as_tuple=False).squeeze(1)  # shape: [valid_voxels]
        full_voxel[0, :, valid_indices] = voxel_squeezed  # full_voxel: [1, 256, total_voxels]
        
        return full_voxel
    
    def voxelizer(self, tensor):
        # sample features in 2d image points
        # grid is in 1, n_h*n_w*n_d, 1, 2 in VU. So:
        
        N_F = tensor.shape[1]
                
        voxel = F.grid_sample(tensor, self.grid_flat_batch,
                            mode='bilinear', align_corners=True)
        
        full_voxel = self._voxel_filler(voxel)
        
        full_voxel = full_voxel.reshape(tensor.shape[0], N_F,
                                        self.grid_resolution[0], self.grid_resolution[1], self.grid_resolution[2])
        
        return full_voxel
    
    def _init_2d_head(self):
        self.head = HD.head_box_2d_bev(self.cfg)
        
    def _init_3d_head(self):
        self.head = HD.head_box_3d(self.cfg)
        
    def forward(self, img_l, img_r, mem_left, calib):
        """
        img_l and img_r: a torch tensor of shape (n, 3, 512, 960)
        mem_left and mem_right: tensors of shape 1, 3, 128, 240

        """
        if self.debug:
            print("==> Beginning of forward")
        

        if self.cfg.camera.P_l is None:
            raise NotImplementedError("Must a valid P provided!")
        else:
            self.grid_flat_batch = self.grid_flat_filtered.repeat(len(img_l), 1, 1, 1).to(self.device)
        
        if self.debug:
            print("==> Grid is generated, flattened, and batched")
            print("==> Feature extraction from current frame statrted")
            
        # Feature extraction from each image
        left_f = self.backbone(img_l)
        right_f = self.backbone(img_r) # shape for outputs: (1, 48, 128, 240)

        if self.debug:        
            print("==> Feature extraction done!")

        # this is the first model output
        mem_l = self.conv2d_memory(left_f)
        mem_l = self.bn_memory(mem_l)
        output_memory = self.relu_create_mem(mem_l)
        if self.debug:        
            print("==> output memory is created")
        assert output_memory.shape[2] == int(self.h/4)
        
        # keeping left features for final concatenation
        base_feature = torch.cat([left_f, mem_left], dim = 1)
        if self.debug:
            print("==> Stereo matching started")
        # matching stage
        matched_tensor = self.matching_module(left_f, right_f, base_feature)
        
        if self.debug:  
            print("==> Voxelization started")
        voxel = self.voxelizer(matched_tensor)
        
        if self.debug:
            print("==> Detection head started")
        if self.cfg.model.head == 'box2d':
            out = self.head(voxel)
        elif self.cfg.model.head == 'box3d':
            out = self.head(voxel)
        else:
            raise NotImplementedError("other representation ehad methods!")
        # TODO: to remove extra outputs
        return out, output_memory, self.grid, self.oob_mask_valid
    
    def init_weights(self):
        raise NotImplementedError("not yet implemented")
        
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
    
    
def test_model(h, w):
    from model_cong import config_generator
    cfg = config_generator()
    
    model = get_SX3D_model(cfg)
    print("==> model is generated")
    img_l = {
        "img_tensor": torch.randn(1, 3, h, w),
        "P": torch.randn(3, 4)
        }
    
    img_r = {
        "img_tensor": torch.randn(1, 3, h, w),
        "P": torch.randn(3, 4)
        }
    
    model.eval()
    out = model(img_l, img_r)
    
    return out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    