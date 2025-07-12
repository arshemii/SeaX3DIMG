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
import glob
import re
import argparse 
import torch

def prepare_env():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
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
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from proc.train import Trainer
    from utils.data_utils import collate_fn
    from utils.kitti_sx3d import kitti_sx3d
    from SX3DIMG import get_SX3D_model
    from utils.grid_generator import GridGenerator
    from utils.loss import loss_3d
    
    device = cfg.device[0]
    
    dataset = kitti_sx3d(cfg)
    
    model = get_SX3D_model(cfg)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = cfg.dev.lr, weight_decay = cfg.dev.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max = cfg.dev.t_max, eta_min = cfg.dev.eta_min)
    
    grid_obj = GridGenerator(cfg.grid_size, cfg.grid_unc) # points in cam coordinates
    grid = grid_obj.get_grid()['grid'].to(dtype=torch.float32).permute(1,2,3,0)
    grid = grid.to(device)
    
    loss_fn = loss_3d(cfg)
    
    if cfg.dev.eval_in_train == False:
        metric_module = None
    else:
        raise NotADirectoryError("Evaluation metric is not finished yet!")
    
    if len(os.listdir(cfg.model.sx3d.checkpoint)) == 0:
        resume_checkpoint = None
    else:
        resume_checkpoint = max(glob.glob("checkpoint_epoch_*.pth"), key=lambda x: int(re.findall(r'\d+', x)[-1]))
    
    trainer = Trainer(cfg, model, dataset, grid, collate_fn, metric_module,
                 optimizer, scheduler, loss_fn, resume_checkpoint)
    
    trainer.train()
    
def evaluation(cfg):
    raise NotADirectoryError("Evaluation function is not finished yet!")

def test(cfg):
    raise NotADirectoryError("Test function is not finished yet!")

def inference(cfg):
    raise NotADirectoryError("Inference function is not finished yet!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for Detection inference. Use flags:")
    
    parser.add_argument(
        '--mode', 
        type=str,
        choices=['train', 'test', 'eval', 'inference'],
        required=True,
        help="Development mode!"
    )
    
    cfg = prepare_env()
    
    cfg.dev.mode = parser.parse_args().mode
    
    if cfg.dev.mode == 'train':
        training(cfg)
    elif cfg.dev.mode == 'test':
        test(cfg)
    elif cfg.dev.mode == 'eval':
        evaluation(cfg)
    else:
        inference(cfg)
        
    
    
