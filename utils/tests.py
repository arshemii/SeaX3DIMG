#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 20:41:03 2025

@author: arash
"""

def forward_test():
    from model_cong import config_generator
    cfg = config_generator()
    
    cfg.device = ['cpu']
    
    from utils.kitti_sx3d import kitti_sx3d
    from SX3DIMG import get_SX3D_model
    from torch.utils.data import DataLoader
    from utils.data_utils import collate_fn

    device = cfg.device[0]
    
    dataset = kitti_sx3d(cfg)
    
    dataloader = DataLoader(dataset, batch_size=cfg.num_batch, shuffle=True,
                                 collate_fn=collate_fn, num_workers=cfg.num_worker)
    
    for idx, batch in enumerate(dataloader):
        if idx == 15:
            break
    batch["calib"] = batch["calib"].to(device)
    batch["left_img"] = batch["left_img"].to(device)
    batch["left_img_previous"] = batch["left_img_previous"].to(device)
    batch["right_img"] = batch["right_img"].to(device)

    for sample in batch["label"]:
        for label in sample:
            label['category'] = label['category'].to(device)
            label['bbox2d'] = label['bbox2d'].to(device)
            label['bbox3d'] = label['bbox3d'].to(device)
    
    model = get_SX3D_model(cfg)
    model.to(device)
    model.eval()
    
    temporal_l = model.create_memory(batch["left_img_previous"])
    #outputs = self.model(batch["left_img"], batch["right_img"], temporal_l, batch["calib"])[0]
    outputs = model(batch["left_img"], batch["right_img"], temporal_l, batch["calib"])
    
    out, output_memory, grid, voxel, left_f, base_feature, matched_tensor, oob_mask_valid = outputs
    
    return out, output_memory, grid, voxel, left_f, base_feature, matched_tensor, oob_mask_valid


out, output_memory, grid, voxel, left_f, base_feature, matched_tensor, oob_mask_valid = forward_test()


false_indices = (oob_mask_valid == False).nonzero(as_tuple=False)
true_indices = (oob_mask_valid == True).nonzero(as_tuple=False)

true_sample = true_indices[155]
voxel_value_true = voxel[0, :, true_sample[0], true_sample[1], true_sample[2]]
out_true = out[0, :, true_sample[0], true_sample[1], true_sample[2]]


false_sample = false_indices[122]
voxel_value_false = voxel[0, :, false_sample[0], false_sample[1], false_sample[2]]
out_false = out[0, :, false_sample[0], false_sample[1], false_sample[2]]












