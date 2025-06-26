#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 19:53:22 2025

@author: arash
"""

from yacs.config import CfgNode as CN
import numpy as np

def config_generator():
    cfg = CN()
    cfg.model = CN()
    cfg.data = CN()
    cfg.camera = CN()
    cfg.model.back = CN()
    cfg.model.sx3d = CN()
    cfg.model.head = CN()
    
    cfg.log_dir = './logging_dir'
    cfg.logging = False
    
    # Model params
    cfg.model.num_class = 4
    #cfg.model.in_size = (512, 960)
    cfg.model.orig_size = (384, 1214)
    cfg.model.in_size = (288, 960)
    cfg.data.scale_1 = cfg.model.in_size[1]/cfg.model.orig_size[1]
    cfg.data.scale_0 = cfg.model.in_size[0]/cfg.model.orig_size[0]
    
    cfg.data.scale = max(cfg.data.scale_1, cfg.data.scale_0)
    
    cfg.model.head = 'box3d'
    cfg.model.back.name = 'hrnet-w48'
    cfg.model.unet_cout = 2
    cfg.model.hrnet_cout = 48
    cfg.model.max_disp = 16
    
    ################################
    
    cfg.model.sx3d.use_checkpoint = False
    cfg.model.sx3d.checkpoint = 'Path to checkpoint'
    ####################################
    
    cfg.data.type = 'random' # other option :'sequential'
    cfg.data.path = './kitti/'
    cfg.data.filter = {
        "trunc": 0.8,
        "occl": [0, 1, 2]}
    
    cfg.camera.P_l = None
    
    cfg.model.sx3d.drop_out = 0.10
    
    
    
    cfg.radar_fusion = False
    cfg.model.sx3d.init_weight = True
    
    cfg.grid_size = (20.0, 12.0, 42.0)
    cfg.grid_unc = (0.2, 0.4, 0.6)
    
    cfg.model.sx3d.is_confidence = True
    
    cfg.data.categories = ["Car", "DontCare", "Pedestrian", "Van", "Tram", "Misc", "Person_sitting", "Cyclist", "Truck"]
    cfg.data.cl0 = ["Car", "Van"]
    cfg.data.cl1 = ["Truck"]
    cfg.data.cl2 = ["Pedestrian"]
    cfg.data.cl3 = ["Cyclist"]
    cfg.data.cl4 = ["DontCare", "Tram"]  # no need to predict, must be removed also from data labeling
    cfg.data.cl5 = ["Misc", "Person_sitting"] # no need to predict, must be removed also from data labeling
    
    cfg.data.mean = np.array([0.485, 0.456, 0.406])
    cfg.data.std = np.array([0.229, 0.224, 0.225])
    cfg.data.img_layout = 'rgb'
    
    
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


    return cfg