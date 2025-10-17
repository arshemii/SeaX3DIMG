#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:12:44 2025

@author: arash
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.grid_generator import GridGenerator
from utils.data_utils import collate_fn
from utils.kitti_sx3d import kitti_sx3d

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


def assign_gt_to_voxels(grid, gtl, voxel_size, list_objects_all, bad_objects):
    """
    Assigns ground truth objects to the 3D grid.

    Args:
        assignments: Tensor of shape [B, W, H, D] all filled by -1
        c_voxels: Tensor of shape [B, 18, 4] all filled by -1
        grid: Tensor of shape [W, H, D, 3] representing voxel centers (x, y, z)
        gtl: Shape: (B, 18, 14)
        
    Returns:
        assignments: Tensor of shape [B, W, H, D] with:   [-1: background | >= 0: index of gtl object | 2: ignored]
        c_voxels:Tensor of shape [B, 18, 4]:
            for each object, the center voxel closest to the object center is defined. index of gt_idx is the same as assignment
    """
   
    B, max_objects, _ = gtl.shape
    W, H, D = grid.shape[:3]
    
    for b in range(B):
        if gtl[b, :, 13].sum() == 0:
             continue
        else:
            for obj_idx in range(max_objects):
                if int(gtl[b, obj_idx, 13]) == 1:
                    obj_ls = []
                    h, w, l = gtl[b, obj_idx, 0:3]
                    cx, cy, cz = gtl[b, obj_idx, 3:6]
                    obj_ls.append((w, h, l))
                    obj_ls.append((cx, cy, cz))
                    
                    # if object is smaller than an edge of the voxel:
                    w = max(w, voxel_size[0] + 0.02*voxel_size[0])
                    h = max(h, voxel_size[1] + 0.02*voxel_size[1])
                    l = max(l, voxel_size[2] + 0.02*voxel_size[2])
                    obj_ls.append((w, h, l))
                    
                    # AABB
                    x_min, x_max = cx - w / 2, cx + w / 2
                    y_min, y_max = cy - h / 2, cy + h / 2
                    z_min, z_max = cz - l / 2, cz + l / 2
                    
                    obj_ls.append((x_min, x_max, y_min, y_max, z_min, z_max))
                                        
                    # mask voxels inside this object
                    xs, ys, zs = grid[..., 0], grid[..., 1], grid[..., 2]
                    inside = (xs >= x_min) & (xs <= x_max) & \
                             (ys >= y_min) & (ys <= y_max) & \
                             (zs >= z_min) & (zs <= z_max)
                    if inside.sum() == 0:
                        grid_x_min, grid_x_max = xs.min(), xs.max()
                        grid_y_min, grid_y_max = ys.min(), ys.max()
                        grid_z_min, grid_z_max = zs.min(), zs.max()
                        
                        obj_ls.append((grid_x_min, grid_x_max, grid_y_min, grid_y_max, grid_z_min, grid_z_max))
                        
                        bad_objects.append(obj_ls)
                    obj_ls.append(inside.sum())            
                    list_objects_all.append(obj_ls)            



def test_assignment():
    
    cfg = check_env()
    
    
    device = cfg.device[0]
    
    grid_obj = GridGenerator(cfg.grid_size, cfg.grid_unc, cfg.H_off) # points in cam coordinates
    grid = grid_obj.get_grid()['grid'].to(dtype=torch.float32).permute(1,2,3,0)
    grid = grid.to(device)
    
    dataset = kitti_sx3d(cfg)
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                                 collate_fn=collate_fn, num_workers=cfg.num_worker)
    
    list_objects_all = []
    bad_objects = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Test of assignment")
    for batch_idx, batch in pbar:
        batch["label"] = batch["label"].to(device)
        
        assign_gt_to_voxels(grid, batch["label"], cfg.grid_unc, list_objects_all, bad_objects)
        pbar.set_postfix({"progress": f"{batch_idx+1}/{len(dataloader)}"})

        
    return list_objects_all, bad_objects
        
        

