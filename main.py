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

import torch
from torch.utils.data import DataLoader
from utils.data_utils import collate_fn
from utils.kitti_sx3d import kitti_sx3d

from SX3DIMG import get_SX3D_model
from model_cong import config_generator
cfg = config_generator()



model = get_SX3D_model(cfg)
model.eval()

dataset = kitti_sx3d(cfg)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                             collate_fn=collate_fn, num_workers=4)
i = 2
for batch in dataloader:
    if i == 2:
        #batch_id = idx
        sample = batch
        
temporal_l = model.create_memory(sample["left_img_previous"])
