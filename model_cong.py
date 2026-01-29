#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documentation:

    P value after conversion:
    tensor([[5.5771e+02, 0.0000e+00, 4.7116e+02, 3.4672e+01],
           [0.0000e+00, 5.5771e+02, 1.3361e+02, 1.6725e-01],
           [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.7459e-03]], dtype=torch.float64)

    Dataset range:
        x--> -40 to +40
        y--> -0.64 to +3.86
        z --> +94

"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)


from yacs.config import CfgNode as CN
import numpy as np
import torch
import utils.data_utils as du
from utils.grid_generator import GridGenerator, cam_to_img, grid_for_sample, oob_voxels

def grid_setup(grid_size, grid_unc, input_size, H_off, p_l):
    grid_obj = GridGenerator(grid_size, grid_unc, H_off) # points in cam coordinates
    grid = grid_obj.get_grid()['grid'].to(dtype=torch.float32)  # 3d gird
    grid_forward = grid.permute(1,2,3,0)
    grid_img = cam_to_img(grid, p_l)  # grid point on image frame
    oob_mask = oob_voxels(grid_img, input_size)
    oob_mask_valid = ~oob_mask    # points outside of the camera FOV are False
    oob_mask_flat = oob_mask_valid.view(-1)
    grid_flat = grid_for_sample(grid_img, input_size)
    grid_flat_filtered = grid_flat[0][oob_mask_flat]
    
    return [grid_forward], [oob_mask_valid], [oob_mask_flat], [grid_flat_filtered]

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
    cfg.test = CN()
    
    
    cfg.dev.scheduler = 'CAlr'  # options: 'OClr', 'CAlr', 'CAlrW'
    #cfg.dev.num_epoch = 45
    cfg.dev.num_epoch = 9
    cfg.dev.t_max = cfg.dev.num_epoch
    cfg.dev.grad_steps = 8
    cfg.dev.eta_min = 7e-6 * cfg.dev.grad_steps # first setup 9e-6
    cfg.dev.mode = 'train'
    cfg.dev.lr = 9e-6 * cfg.dev.grad_steps # first setup 1e-5
    cfg.dev.weight_decay = 1e-4
    cfg.dev.eval_in_train = False
    cfg.dev.continue_training = False
    
    
    
    cfg.num_batch = 1
    cfg.num_worker = 2
    
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
    
    cfg.model.head.type = '3d_box'
    cfg.model.head.inplanes = 128
    cfg.model.back.name = 'hrnet-w48'  # other option DDRNet-23-slim
    
    if cfg.model.back.name == 'hrnet-w48':
        cfg.model.back.pretrained_path = 'weights/hrnet_w48-8ef0771d.pth'
    elif cfg.model.back.name == 'DDRNet-23-slim':
        # cfg.model.back.bn_mom = 0.1
        cfg.model.back.pretrained_path = 'weights/DDRNet23s_imagenet.pth'
    else:
        raise NotImplementedError("Only DDRNet and HRNet are available for backbone")
    
    cfg.model.unet_cout = 2
    cfg.model.hrnet_cout = 48
    cfg.model.max_disp = 16
    cfg.model.conf_voxel = True
    cfg.model.sx3d.drop_out = 0.10
    cfg.model.sx3d.use_checkpoint = True
    cfg.model.sx3d.checkpoint_exp = './checkpoints_exp/staged_training/fullhead.pth'
    cfg.model.sx3d.checkpoint_3d = './checkpoints_3d/'
    cfg.model.sx3d.checkpoint_bev = './checkpoints_bev/'
    
    cfg.data.path = './dataset/sequential/'
    cfg.data.filter = [{"trunc": 1.0,
                        "occl": [0, 1, 2, 3]}]
    
    cfg.data.object_AABB_scale = 1.2 # TODO: if model predicts a lot of objects, increase it 
    
    cfg.camera.p_l_original = [torch.tensor([
                                            [7.215377e+02, 0.0, 6.095593e+02, 4.485728e+01],
                                            [0.0, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                                            [0.0, 0.0, 1.0, 2.745884e-03]
                                            ], dtype=torch.float32)]
    
    cfg.camera.P_l = [torch.tensor([[5.5771e+02, 0.0000e+00, 4.7116e+02, 3.4672e-02],
                                             [0.0000e+00, 5.5771e+02, 1.3361e+02, 1.6725e-04],
                                             [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.7459e-06]
                                             ], dtype=torch.float32)]
    
    cfg.camera.P_r = [torch.tensor([[ 5.5771e+02,  0.0000e+00,  4.7116e+02, -2.6243e+02],
                                            [ 0.0000e+00,  5.5771e+02,  1.3361e+02,  1.7004e+00],
                                            [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  2.7299e-03],
                                           ], dtype=torch.float32)]
    
    cfg.camera.R0 = [torch.tensor([[ 0.9999239 ,  0.00983776, -0.00744505],
                                       [-0.0098698 ,  0.9999421 , -0.00427846],
                                       [ 0.00740253,  0.00435161,  0.9999631 ],
                                       ], dtype=torch.float32)]
    
    cfg.camera.V2C = [torch.tensor([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                       [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
                                       [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
                                       ], dtype=torch.float32)]
    cfg.camera.disp = [du.get_focal_baseline(cfg.camera.P_l[0], cfg.camera.P_r[0])]
    
    cfg.camera.focal = [cfg.camera.disp[0][0]]
    cfg.camera.base = [cfg.camera.disp[0][1]]
    
    
    cfg.radar_fusion = False
    cfg.model.sx3d.init_weight = True
    
    cfg.max_obj = 18
    
    cfg.grid_size = (19.7, 7.2, 63.4)  # H from -2 to 10
    cfg.grid_unc = (0.36, 0.36, 0.39)
    cfg.grid_resolution = tuple(int(round(size / res)) for size, res in zip(cfg.grid_size, cfg.grid_unc))


    
    if cfg.grid_size[2] < 90.0:
        cfg.short_grid_range = True
    
    cfg.grid_resolution = tuple(int(round(size / res)) for size, res in zip(cfg.grid_size, cfg.grid_unc))
    cfg.H_off = 4
    cfg.H_min = -cfg.grid_size[1]/2 + cfg.H_off
    cfg.H_max = cfg.grid_size[1]/2 + cfg.H_off
    
    
    cfg.grid_forward, cfg.oob_mask_valid, cfg.oob_mask_flat, cfg.grid_flat_filtered = grid_setup(cfg.grid_size,
                                                                                                    cfg.grid_unc, cfg.model.in_size,
                                                                                                    cfg.H_off, cfg.camera.P_l[0])    
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
    
    
    cfg.model.back.out = "features"
    cfg.model.back.init_weight = False
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


    # Staging:
    # experiment with stage 1:
    # cfg.loss.heads = ['disp', 'obj_head', 'cls_head', 'cnt_head', 'dim_head', 'yaw_head']
    cfg.loss.heads = ['disp', 'obj_head', 'cls_head', 'cnt_head', 'dim_head', 'yaw_head']
    cfg.loss.w_total_previous = [0.08, 0.08, 0.05, 0.10]  # main experiment with 0.15, [0.05, 0.10], [0.02, 0.05, 0.20],  [0.02, 0.02, 0.05, 0.15]
    cfg.loss.w_yaw = 0.90
    cfg.loss.freeze = False
    cfg.loss.freezed_output = ['disp']
    

    cfg.loss.alpha = 0.55  # TODO: if model predicts a lot of objects, increase it 
    cfg.loss.gamma = 2.0
    cfg.loss.beta = 1.4
    cfg.loss.object_threshold_loss = 0.5
    cfg.loss.heatmap_thr = 0.5
    cfg.loss.zeta = 0.2    # to penalize background voxels if objectness is high
    cfg.loss.optimized = True
    cfg.loss.debug = False
    cfg.loss.track = True
    cfg.loss.max_disp = 192.0 * cfg.model.in_size[1] / (cfg.model.orig_size[1])
    

    cfg.eval.save_dir = './eval_dir/'
    cfg.eval.save_dir_gt = './eval_dir_gt/'
    cfg.eval.split_eval = './dataset/sequential/val.txt'
    cfg.eval.score_th = 0.20
    cfg.eval.batch_size = 4
    cfg.eval.num_workers = 0
    cfg.eval.topk = 30
    cfg.eval.local_maxima_kernel = 7
    cfg.eval.cl0 = "Car, Van"
    cfg.eval.cl1 = "Truck"
    cfg.eval.cl2 = "Pedestrian"
    cfg.eval.cl3 = "Cyclist"
    cfg.eval.class_names = ["Car", "Truck", "Person", "Cyclist"]
    
    return cfg



