#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:21:33 2025

@author: arash
"""
from torch.utils.data import Dataset
import data_utils as du
import cv2

class kitti_sx3d(Dataset):
    def __init__(self, cfg, mode = 'train'):
        super(kitti_sx3d, self).__init__()
        self.mode = mode
        self.cfg = cfg
        
        if self.mode == 'train':
            self.data_dir = self.cfg.data.path + 'training/'
            self.DF = du.parse_id_file(self.data_dir + 'ImageSets/train.txt', self.data_dir)
            self._label_parse()
        elif self.mode == 'val':
            self.data_dir = self.cfg.data.path + 'training/'
            self.DF = du.parse_id_file(self.data_dir + 'ImageSets/val.txt', self.data_dir)
            self._label_parse()
        elif self.mode == 'test':
            self.data_dir = self.cfg.data.path + 'testing/'
            self.DF = du.parse_id_file(self.data_dir + 'ImageSets/test.txt', self.data_dir)
        else:
            raise NotImplementedError("Only test and train available!")
        
        self._calibration_parse()

        
        
    def _calibration_parse(self):
        for instance in self.DF:
            instance["calib_params"] = du.parse_calibration(instance["calib_path"])
            
    def _label_parse(self):
        for instance in self.DF:
            instance["labels"] = du.parse_label(instance["label_path"], self.cfg)
        
    def __getitem__(self, index):
        instance = self.DF[index]
        
        self.img_l =  cv2.imread(instance['img_l_path'], 1 | 128 )   
        self.img_r =  cv2.imread(instance['img_r_path'], 1 | 128 )
        
        self.img_l = cv2.cvtColor(self.img_l, cv2.COLOR_BGR2RGB)
        self.img_r = cv2.cvtColor(self.img_r, cv2.COLOR_BGR2RGB)
        
        self.img_l, scale, crop, cirection = du.img_resize(self.img_l, self.cfg.model.in_size)
        self.img_r, _, _, _ = du.img_resize(self.img_r, self.cfg.model.in_size)
        
        self.P_l_converted = du.parse_calibration(instance['calib_params']['P2'])
        
        self.img_l = du.img_normalize(self.img_l, self.cfg.data.mean, self.cfg.data.std)
        self.img_r = du.img_normalize(self.img_r, self.cfg.data.mean, self.cfg.data.std)
        
        data = {"left_img": self.img_l,
                "right_img": self.img_r,
                "calib": self.P_l_converted}
        
        if "labels" in instance.keys():
            data["label"] = instance["labels"]
        
        return data
        
    def __len__(self):
        return len(self.DF)
