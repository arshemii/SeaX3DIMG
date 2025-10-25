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
    from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
    from proc.train import Trainer
    from utils.data_utils import collate_fn
    from utils.kitti_sx3d import kitti_sx3d
    from SX3DIMG import get_SX3D_model
    from utils.loss import loss3d
    
    device = cfg.device[0]
    
    # Preparing the model ...
    
    model = get_SX3D_model(cfg)
    model.to(device)
    model.train()
    
    # Preparing dataset ...
    
    dataset = kitti_sx3d(cfg)
    
    # Dataset is ready. wooooow!
         
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = cfg.dev.lr, weight_decay = cfg.dev.weight_decay)
    if cfg.dev.scheduler == 'CAlr':
        print("Training initialized and scheduled with CosineAnnealingLR")
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max = cfg.dev.t_max, eta_min = cfg.dev.eta_min)
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
    
def evaluation(cfg):
    from utils.data_utils import collate_fn
    from utils.kitti_sx3d import kitti_sx3d
    from SX3DIMG import get_SX3D_model
    from proc.evaluation import evaluate_model
    
    
    
    # model preparation
    checkpoint_dir = cfg.model.sx3d.checkpoint_bev
    latest_ckpt = max(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")),
        key=lambda x: int(re.search(r"checkpoint_epoch_(\d+).pth", x).group(1)))
    
    print(f"Latest checkpoint to start is: {checkpoint_dir}")
    
    model = get_SX3D_model(cfg)
    model = model.to(cfg.device[0])
    cfg.oob_mask_valid = model.return_boundary_mask()
    checkpoint = torch.load(latest_ckpt, map_location = cfg.device[0])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    dataset = kitti_sx3d(cfg)
    
    results = evaluate_model(model, dataset, collate_fn, cfg, debug = cfg.eval.debug)
    
    eval_range = cfg.eval.range if cfg.eval.range_limit else 90.0
    
    print(f"------------------- Evaluation Results for {eval_range} -------------------")    
    for iou_th, res in results.items():
        print(f"\nIoU threshold = {iou_th:.2f}")
        print("Class |  TP   FP   FN | Precision | Recall |   AP")
        print("--------------------------------------------------------")
        for c, stats in res['per_class'].items():
            print(f"{c:5d} | {stats['TP']:3d} {stats['FP']:3d} {stats['FN']:3d} "
                  f"| {stats['precision']:.3f}    | {stats['recall']:.3f} | {stats['AP']:.5f}")
        print("--------------------------------------------------------")
        print(f"mAP@{iou_th:.2f} = {res['mAP']:.3f}")
    print("=========================================================")

    

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
    
    cfg = check_env()
    
    cfg.dev.mode = parser.parse_args().mode
    
    if cfg.dev.mode == 'train':
        training(cfg)
    elif cfg.dev.mode == 'test':
        test(cfg)
    elif cfg.dev.mode == 'eval':
        evaluation(cfg)
    else:
        inference(cfg)
        
    
    
