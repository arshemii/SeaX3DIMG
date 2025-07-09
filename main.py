#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script handle everything for training process

Notes:
    1. change the dataset module to have previous step images

"""
#import sys
#import os
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#import torch
#from torch.utils.data import DataLoader
#from utils.data_utils import collate_fn
#from utils.kitti_sx3d import kitti_sx3d

# from SX3DIMG import get_SX3D_model
# from model_cong import config_generator
# cfg = config_generator()


# device = cfg.device[0]
# print(device)

# model = get_SX3D_model(cfg)
# model.to(device)
# model.eval()

# dataset = kitti_sx3d(cfg)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
#                              collate_fn=collate_fn, num_workers=2)

# for idx, batch in enumerate(dataloader):
#     print("==> batch is working ...")
#     if idx == 5:
#         #batch_id = idx
#         sample = batch
#         break
        



# sample["left_img_previous"] = sample["left_img_previous"].to(device)
# sample["left_img"] = sample["left_img"].to(device)
# sample["right_img"] = sample["right_img"].to(device)
# sample["calib"] = sample["calib"].to(device)

# print(torch.cuda.memory_summary())

# temporal_l = model.create_memory(sample["left_img_previous"])


# output = model(sample["left_img"],
#                sample["right_img"],
#                temporal_l.to(device),
#                sample["calib"])


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from utils.data_utils import collate_fn
from utils.kitti_sx3d import kitti_sx3d

from SX3DIMG import get_SX3D_model
from model_cong import config_generator
cfg = config_generator()

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.grid_generator import GridGenerator
from utils.loss import loss_3d

epoch = 1

device = cfg.device[0]
model = get_SX3D_model(cfg)
model.to(device)
model.train()
running_loss = 0.0
avg_loss = 0.0

dataset = kitti_sx3d(cfg)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)  # T_max = epochs
grid_obj = GridGenerator(cfg.grid_size, cfg.grid_unc) # points in cam coordinates
grid = grid_obj.get_grid()['grid'].to(dtype=torch.float32).permute(1,2,3,0)
grid = grid.to(device)
loss_fn = loss_3d(cfg)
    
pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
for batch_idx, batch in pbar:
    print(f"batch {batch_idx} started")
    if batch_idx == 5:
        break
    
    for key in ["left_img", "left_img_previous", "right_img", "calib"]:
        if key == "calib":
            batch["calib"] = batch["calib"].to(device)
        elif key == "label":
            for label in batch["label"]:
                for k in label.keys():
                    if k in ["category", "bbox3d", "bbox2d"]:
                        label[k] = label[k].to(device)
        else:
            batch[key] = batch[key].to(device)
    
    optimizer.zero_grad()
    
    #create temporal memory for both left and right image from t - dt
    temporal_l = model.create_memory(batch["left_img_previous"])
    outputs = model(batch["left_img"], batch["right_img"], temporal_l, batch["calib"])[0]
    
    loss = loss_fn(outputs, batch["label"], grid)
    loss['total'].backward()
    optimizer.step()
    
    running_loss += loss['total'].item()
    avg_loss = running_loss / (batch_idx + 1)

scheduler.step()