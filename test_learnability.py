#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:01:20 2025

@author: arash
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import random
from utils.loss import loss3d
from torch.utils.data import DataLoader
from utils.data_utils import collate_fn
from utils.kitti_sx3d import kitti_sx3d
from SX3DIMG import get_SX3D_model


def check_env():
    from model_cong import config_generator
    cfg = config_generator()
    
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
    
        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i} name:", torch.cuda.get_device_name(i))
            print("  Total memory (GB):", round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2))
            print("  Multiprocessors (SMs):", torch.cuda.get_device_properties(i).multi_processor_count)
            print("  CUDA capability:", torch.cuda.get_device_properties(i).major, ".", torch.cuda.get_device_properties(i).minor)
    return cfg

def train_epoch(model, dataloader, loss_fn, cfg):

    rand_idx = random.randint(0, len(dataloader) - 1)
    for i, batch in enumerate(dataloader):
        if i == rand_idx:
            break
                    
    batch["left_img"] = batch["left_img"].to(cfg.device[0])
    batch["right_img"] = batch["right_img"].to(cfg.device[0])
        
    with torch.no_grad():
        # model forward
        outputs, disp = model(batch["left_img"], batch["right_img"],
                                   None, create_memory = False, mode = cfg.dev.mode,
                                   lw = cfg.loss.weights)
        obj = outputs[0]
        dim = outputs[1]
        centerx = outputs[2]
        cls_logits = outputs[3]
        yaw = outputs[4]
        
        del batch["left_img"], batch["right_img"], outputs
        assert "label" in batch.keys()
        
        batch["label"] = batch["label"].to(cfg.device[0])
        batch['assignment'] = batch['assignment'].to(cfg.device[0])
        batch["disparity"] = batch["disparity"].to(cfg.device[0])
            
        # Loss calculation
        loss = {}

        loss['disparity_loss'], cnt_disp = loss_fn.disparity_loss(disp, batch["disparity"])
        del disp, batch["disparity"]
         
        loss['obj_conf'], cnt_obj = loss_fn.object_conf_loss(obj, batch['assignment'])
        loss['cls_loss'], cnt_cls = loss_fn.classification_loss(cls_logits, obj, batch['assignment'], batch["label"])
        del cls_logits, obj
            
        loss['center_loss'], cnt_center = loss_fn.center_loss(centerx, batch['assignment'], batch["label"])
        del centerx

            
        loss['dim_loss'], cnt_dim = loss_fn.dimension_loss(dim, batch['assignment'], batch["label"])
        del dim

        loss['yaw_angle_loss'], cnt_yaw = loss_fn.yaw_loss(yaw, batch['assignment'], batch["label"])
        del yaw
            
        del batch["label"], batch['assignment']
            
        loss['total'] = cfg.loss.weights[0] * loss['obj_conf'] + \
                        cfg.loss.weights[1] * loss['cls_loss'] + \
                        cfg.loss.weights[2] * loss['center_loss'] + \
                        cfg.loss.weights[3] * loss['dim_loss'] + \
                        cfg.loss.weights[4] * loss['yaw_angle_loss'] + \
                        cfg.loss.weights[5] * loss['disparity_loss']
            
        torch.cuda.empty_cache()
        
        total_loss = loss['total'].item()
        
        term_loss = {
            'objectness': loss['obj_conf'].item(),
            'classification': loss['cls_loss'].item(),
            'center': loss['center_loss'].item(),
            'dimension': loss['dim_loss'].item(),
            'rotation': loss['yaw_angle_loss'].item(),
            'disparity': loss['disparity_loss'].item(),
            }
        
        weighted_term_loss = {
            'objectness': cfg.loss.weights[0] * loss['obj_conf'].item(),
            'classification': cfg.loss.weights[1] * loss['cls_loss'].item(),
            'center': cfg.loss.weights[2] * loss['center_loss'].item(),
            'dimension': cfg.loss.weights[3] * loss['dim_loss'].item(),
            'rotation': cfg.loss.weights[4] * loss['yaw_angle_loss'].item(),
            'disparity': cfg.loss.weights[5] * loss['disparity_loss'].item()
            }
        
    return total_loss, term_loss, weighted_term_loss
    
    
#######################################

cfg = check_env()

model = get_SX3D_model(cfg)
model = model.to(cfg.device[0])
model = model.eval()

loss_fn = loss3d(cfg)

dataset = kitti_sx3d(cfg, mode = 'train_small')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                 collate_fn=collate_fn, num_workers=cfg.num_worker)



for i in range(10):
    total_loss, term_loss, weighted_term_loss = train_epoch(model, dataloader, loss_fn, cfg)
    print(total_loss)