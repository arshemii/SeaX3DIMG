#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 19:53:22 2025

@author: arash
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from modules.pose_hrnet import get_pose_net
from modules.matching import Level
from utils.grid_generator import GridGenerator,cam_to_img, grid_for_sample
from modules import heads as HD
from utils.debug_util import setup_logger

logger = setup_logger('SX3DIMG_logs', './logging_dir')

class SX3DIMG(nn.Module):
    def __init__(self, cfg, is_train_backbone):
        super(SX3DIMG, self).__init__()
        self.cfg = cfg
        self.logs = self.cfg.logging
        # self.logger = setup_logger('SX3DIMG_logs', self.cfg.log_dir)
        self.is_train_backbone = is_train_backbone
        
        self.h, self.w = self.cfg.model.in_size
        
        self.grid_obj = GridGenerator(self.cfg.grid_size, self.cfg.grid_unc) # points in cam coordinates
        self.grid = self.grid_obj.get_grid()['grid']
        self.grid_resolution = tuple(int(round(size / res)) for size, res in zip(self.cfg.grid_size, self.cfg.grid_unc))
        
        
        if self.cfg.camera.P_l is not None:
            assert self.cfg.data.type == 'sequential'
            self.P_l = cfg.camera.P_l
            self.grid_img = cam_to_img(self.grid, self.P_l)
            self.grid_flat = grid_for_sample(self.grid_img, (self.h, self.w))
        
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
        self.relu_create_art_mem = nn.ReLU()
        self.relu_matching = nn.ReLU()
        
        

    def feature_net(self):
        if self.cfg.model.back.name == 'hrnet-w48':
            feat_net = get_pose_net(self.cfg, self.is_train_backbone)
        else:
            raise NotImplementedError("Must implement resnet with output of shape (1, 48, 128, 128)")
        if self.logs:
            logger.info("Feature extraction network has initialized.")
        return feat_net

    def _init_matching_layers(self):
        self.hrnet_disp = Level(self.cfg.model.hrnet_cout,
                              self.cfg.model.max_disp, 1)

        
    def _create_memory(self, tensor_f):
        feat_l = self.conv2d_memory(tensor_f[0])
        feat_r = self.conv2d_memory(tensor_f[1])
        
        feat_l = self.bn_memory(feat_l)
        feat_r = self.bn_memory(feat_r)
        
        feat_l = self.relu_create_mem(feat_l)
        feat_r = self.relu_create_mem(feat_r)
        
        return feat_l, feat_r

    def _create_artificial_memory(self, imgs):
        """
        input is a tuple opf left and right image, shape n, 3, 512, 960
        
        method:
            augmet the pari to mimic a previous frame
            only for training
            
        
        """
        
        temporal_aug = transforms.Compose([
                                    transforms.RandomAffine(degrees=0, translate=(0.015, 0.015)),
                                    transforms.ColorJitter(brightness=0.05, contrast=0.05),
                                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                                    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # Gaussian noise
                                    transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)) 
                                ])
        # input tensor shape: (1, 48, 128, 240) normalized
        img_L_prev = temporal_aug(imgs[0].clone())
        img_R_prev = temporal_aug(imgs[1].clone())
        
        left_f_mem = self.backbone(img_L_prev)
        right_f_mem = self.backbone(img_R_prev) # shape for outputs: (1, 48, 128, 240)
        
        left_f_mem = self.conv2d_memory(left_f_mem)
        right_f_mem = self.conv2d_memory(right_f_mem)
        
        left_f_mem = self.bn_memory(left_f_mem)
        right_f_mem = self.bn_memory(right_f_mem)
        
        left_f_mem = self.relu_create_art_mem(left_f_mem)
        right_f_mem = self.relu_create_art_mem(right_f_mem)
        
        return left_f_mem, right_f_mem
    
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
        
        assert base_down.shape == torch.Size([1, 128, int(self.h/8), int(self.w/8)])
        
        base_down = self.conv2d_match(base_down)
        base_down = self.bn_match_2(base_down)
        base_down = self.relu_matching(base_down)
        
        return base_down
    
    def voxelizer(self, tensor):
        # sample features in 2d image points
        # grid is in 1, n_elevation, n_w*n_d, 2 in VU. So:
        
        N_F = tensor.shape[1]
        
        voxel = F.grid_sample(tensor ,self.grid_flat,
                            mode='bilinear', align_corners=True)
        voxel = voxel.reshape(1, N_F, self.grid_resolution[1], self.grid_resolution[0], self.grid_resolution[2])
        
        return voxel
    
    def _init_2d_head(self):
        self.head = HD.head_box_2d_bev(self.cfg)
        
    def _init_3d_head(self):
        self.head = HD.head_box_3d(self.cfg)
        
    def forward(self, img_left, img_right, temp_memory = None):
        """
        img_l and img_r: a torch tensor of shape (n, 3, 512, 960)
        tem_memory: a tuple of tensors of shape 1, 3, 128, 240

        """
        assert temp_memory == None and self.cfg.data.type == 'random'
        
        print("==> st 1")
        
        # data seperation
        img_l, img_r = img_left['img_tensor'], img_right['img_tensor']
        if self.cfg.camera.P_l is None:
            assert self.cfg.data.type == 'random'
            self.P_l = img_left['P']
            self.grid_img = cam_to_img(self.grid, self.P_l)
            self.grid_flat = grid_for_sample(self.grid_img, (self.h, self.w)) # 1, N, 1, 2 in VU
        
        print("==> st 2 - before features")
        # Feature extraction from each image
        left_f = self.backbone(img_l)
        right_f = self.backbone(img_r) # shape for outputs: (1, 48, 128, 240)

        print("==> st 3 - after features")
        
        if not temp_memory:
            temp_memory = self._create_artificial_memory((img_l, img_r))
        
        # this is the first model output
        output_memory = self._create_memory((left_f, right_f))
        assert output_memory[0].shape == torch.Size([1, 3, int(self.h/4), int(self.w/4)])
        
        # keeping left features for final concatenation
        base_feature = torch.cat([left_f, temp_memory[0]], dim = 1)
        print("==> st 4 - before matching")
        # matching stage
        matched_tensor = self.matching_module(left_f, right_f, base_feature)
        print("==> st 5 - after matching")
        
        voxel = self.voxelizer(matched_tensor)
        print("==> st 6 - after voxel")
        # detection head
        if self.cfg.model.head == 'box2d':
            out = self.head(voxel)
        elif self.cfg.model.head == 'box3d':
            out = self.head(voxel)
        else:
            raise NotImplementedError("other representation ehad methods!")
            
        return out, output_memory
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    