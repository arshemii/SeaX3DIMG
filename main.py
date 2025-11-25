#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script handle everything for training process

Notes:
    1. change the dataset module to have previous step images [done]
    2. use mixed precision training if GPU is needed 
       mixed precision only does computation of forward and backward in f16 [done]
    3. number of workers either 2 or 4 [done]
    4. Shall use a profiling tool [not necessary]

"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import glob
import re
import argparse 
import torch

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

def training(cfg):
    from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
    from proc.train import Trainer
    from utils.data_utils import collate_fn
    from utils.kitti_sx3d import kitti_sx3d
    from SX3DIMG import get_SX3D_model
    from utils.loss import loss3d
    
    device = cfg.device[0]
    
    print("Preparing dataset ...")
    dataset = kitti_sx3d(cfg, mode = 'train')
    
    print("Preparing the model ...")
    model = get_SX3D_model(cfg)
    model.to(device)
    model.train()
    if cfg.loss.freeze:
        if 'disp' in cfg.loss.freezed_output:
            print("Freezing disparity generator ...")
            model.hrnet_disp.eval()
    print("Model is initialized completely ...")
         
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = cfg.dev.lr, weight_decay = cfg.dev.weight_decay)
    if cfg.dev.scheduler == 'CAlr':
        print("Training initialized and scheduled with CosineAnnealingLR")
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max = cfg.dev.t_max, eta_min = cfg.dev.eta_min)
        print(f"T max is: {cfg.dev.t_max}, LR is: {cfg.dev.lr}, LR min is: {cfg.dev.eta_min}, Weight decay is: {cfg.dev.weight_decay}")                         
    elif cfg.dev.scheduler == 'CAlrW':
        print("Training initialized and scheduled with CosineAnnealingWarmRestarts")
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=cfg.dev.num_epoch // 3, T_mult=2, eta_min=cfg.dev.eta_min)
    elif cfg.dev.scheduler == 'OClr':
        print("Training initialized and scheduled with OneCycleLR")
        scheduler = OneCycleLR(optimizer, max_lr=cfg.dev.lr, total_steps=cfg.dev.num_epoch * len(dataset),
                               pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4)
    else:
        raise NotImplementedError("No other scheduler is implemented!")
        

    loss_fn = loss3d(cfg)
            

    if len(os.listdir(cfg.model.sx3d.checkpoint_3d)) == 0:
        resume_checkpoint = None
    else:
        resume_checkpoint = max(glob.glob("./checkpoint_3d/checkpoint_epoch_*.pth"), key=lambda x: int(re.findall(r'\d+', x)[-1]))
        print(f"Start training with {resume_checkpoint}")

    trainer = Trainer(cfg, model, dataset, collate_fn,
                 optimizer, scheduler, loss_fn, resume_checkpoint)
    
    trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for Detection inference. Use flags:")
    
    parser.add_argument(
        '--mode', 
        type=str,
        choices=['train', 'test', 'eval', 'inference'],
        required=True,
        help="Development mode!"
    )
    
    cfg = check_env()
    
    cfg.dev.mode = parser.parse_args().mode
    
    if cfg.dev.mode == 'train':
        training(cfg)
    else:
        raise NotADirectoryError("Other modes not implemented yet!")
        
    
    
