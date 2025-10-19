#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 19:53:22 2025

@author: arash
"""

from yacs.config import CfgNode as CN
import numpy as np
import torch
from utils.grid_generator import GridGenerator

def config_generator():
    cfg = CN()
    cfg.model = CN()
    cfg.data = CN()
    cfg.camera = CN()
    cfg.model.back = CN()
    cfg.model.sx3d = CN()
    cfg.model.head = CN()
    cfg.dev = CN()
    cfg.loss = CN()
    cfg.eval = CN()
    
    cfg.dev.scheduler = 'OClr'  # options: 'OClr', 'CAlr'
    cfg.dev.num_epoch = 30
    cfg.dev.t_max = cfg.dev.num_epoch - 10
    cfg.dev.eta_min = 1e-5
    cfg.dev.mode = None
    cfg.dev.lr = 1e-4
    cfg.dev.weight_decay = 1e-4
    cfg.dev.eval_in_train = False
    cfg.dev.continue_training = False
    
    cfg.num_batch = 1
    cfg.num_worker = 0
    
    cfg.log_dir = './logging_dir'
    cfg.logging = False
    
    cfg.debug = False
    cfg.debug_loss = False
    
    #cfg.device = ['cpu']
    cfg.device = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
    
    # Model params
    cfg.model.num_class = 4
    #cfg.model.in_size = (512, 960)
    cfg.model.orig_size = (384, 1214)
    cfg.model.in_size = (288, 960)
    cfg.data.scale_1 = cfg.model.in_size[1]/cfg.model.orig_size[1]
    cfg.data.scale_0 = cfg.model.in_size[0]/cfg.model.orig_size[0]
    
    cfg.data.scale = max(cfg.data.scale_1, cfg.data.scale_0)
    
    cfg.model.head = 'bev_box'  # other is bev_occupancy
    cfg.model.back.name = 'hrnet-w48'
    cfg.model.unet_cout = 2
    cfg.model.hrnet_cout = 48
    cfg.model.max_disp = 16
    
    ################################
    
    cfg.model.sx3d.use_checkpoint = False
    cfg.model.sx3d.checkpoint_3d = './checkpoints_3d/'
    cfg.model.sx3d.checkpoint_bev = './checkpoints_bev/'
    ####################################
    
    cfg.data.path = './dataset/sequential/'
    cfg.data.filter = [{
        "trunc": 0.8,
        "occl": [0, 1, 2]}]
    
    cfg.camera.P_l = [torch.tensor([[5.5771e+02, 0.0000e+00, 4.7116e+02, 3.4672e-02],
                                             [0.0000e+00, 5.5771e+02, 1.3361e+02, 1.6725e-04],
                                             [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.7459e-06]
                                             ], dtype=torch.float32)]
    # example:
    # tensor([[5.5771e+02, 0.0000e+00, 4.7116e+02, 3.4672e+01],
    #        [0.0000e+00, 5.5771e+02, 1.3361e+02, 1.6725e-01],
    #        [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.7459e-03]], dtype=torch.float64)
    
    cfg.model.sx3d.drop_out = 0.10
    
    
    
    cfg.radar_fusion = False
    cfg.model.sx3d.init_weight = True
    
    # dataset range:
        # x--> -40 to +40
        # y--> -0.64 to +3.86
        # z --> +94
    
    cfg.max_obj = 18
    cfg.grid_size = (25.0, 10.0, 50.0)  # H from -2 to 10
    cfg.grid_unc = (0.40, 0.60, 0.40)
    cfg.grid_resolution = tuple(int(round(size / res)) for size, res in zip(cfg.grid_size, cfg.grid_unc))
    
    if cfg.grid_size[2] < 90.0:
        cfg.short_grid_range = True
    
    cfg.grid_resolution = tuple(int(round(size / res)) for size, res in zip(cfg.grid_size, cfg.grid_unc))
    cfg.H_off = 4
    cfg.H_min = -cfg.grid_size[1]/2 + cfg.H_off
    cfg.H_max = cfg.grid_size[1]/2 + cfg.H_off
    
    grid_obj = GridGenerator(cfg.grid_size, cfg.grid_unc, cfg.H_off) # points in cam coordinates
    cfg.grid = [grid_obj.get_grid()['grid'].to(dtype=torch.float32).permute(1,2,3,0)]
    
    cfg.model.sx3d.is_confidence = True
    
    cfg.data.categories = ["Car", "DontCare", "Pedestrian", "Van", "Tram", "Misc", "Person_sitting", "Cyclist", "Truck"]
    cfg.data.cl0 = ["Car", "Van"]
    cfg.data.cl1 = ["Truck"]
    cfg.data.cl2 = ["Pedestrian"]
    cfg.data.cl3 = ["Cyclist"]
    cfg.data.cl4 = ["DontCare", "Tram"]  # no need to predict, must be removed also from data labeling
    cfg.data.cl5 = ["Misc", "Person_sitting"] # no need to predict, must be removed also from data labeling
    cfg.data.ignore_class_id = -2
    
    cfg.data.mean = [np.array([0.485, 0.456, 0.406])]
    cfg.data.std = [np.array([0.229, 0.224, 0.225])]
    cfg.data.img_layout = 'rgb'
    
    cfg.data.max_obj_per_frame = 15
    
    
    cfg.model.back.out = "features"
    cfg.model.back.init_weight = False
    cfg.model.back.pretrained_path = 'weights/hrnet_w48-8ef0771d.pth'
    cfg.model.back.extra = [{"INP_SIZE": [288, 960],
                            "HEATMAP_SIZE": [72, 240],
                            "INP_SIZE_SCALE": [1],
                            "NAME": 'hrnet-w48',
                            "NUM_JOINTS": 17,
                            "OUT": "features",
                            
                            "STAGE2": {"NUM_MODULES": 1,
                                        "NUM_BLOCKS": [4, 4],
                                        "NUM_CHANNELS": [48, 96],
                                        "BLOCK": "BASIC",
                                        "NUM_BRANCHES": 2,
                                        "FUSE_METHOD": "SUM"},
                            
                            "STAGE3": {"NUM_MODULES": 4,
                                        "NUM_BLOCKS": [4, 4, 4],
                                        "NUM_CHANNELS": [48, 96, 192],
                                        "BLOCK": "BASIC",
                                        "NUM_BRANCHES": 3,
                                        "FUSE_METHOD": "SUM"},
                            
                            "STAGE4": {"NUM_MODULES": 3,
                                        "NUM_BLOCKS": [4, 4, 4, 4],
                                        "NUM_CHANNELS": [48, 96, 192, 384],
                                        "BLOCK": "BASIC",
                                        "NUM_BRANCHES": 4,
                                        "FUSE_METHOD": "SUM"},
                            
                            "PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1',
                                                    'transition1', 'stage2', 'transition2',
                                                    'stage3', 'transition3', 'stage4'],
                            
                            "FINAL_CONV_KERNEL": 1}]


    cfg.loss.weight = [1.0, 1.0, 0.75, 0.65, 0.2]
    cfg.loss.aux_loss = False
    if cfg.loss.aux_loss:
        cfg.loss.weight.append(0.3)
        cfg.model.return_disp = True
    cfg.loss.alpha = 0.25
    cfg.loss.gamma = 2.0
    cfg.loss.beta = 1.0
    cfg.loss.object_threshold_loss = 0.5
    cfg.loss.zeta = 0.5    # to penalize background voxels if objectness is high
    

    cfg.eval.save_dir = './eval_dir/'
    cfg.eval.debug = False
    cfg.eval.iou_list = [0.10, 0.25, 0.50, 0.75, 0.90]
    cfg.eval.objectness_threshold = 0.5
    cfg.eval.range_limit = False
    cfg.eval.range = 0
    if cfg.eval.range_limit:
        assert cfg.eval.range <= cfg.grid_size[2]
    
    cfg.eval.eval_device = [torch.device('cpu')]
    cfg.eval.cl0 = "Car, Van"
    cfg.eval.cl1 = "Truck"
    cfg.eval.cl2 = "Pedestrian"
    cfg.eval.cl3 = "Cyclist"
    
    return cfg



