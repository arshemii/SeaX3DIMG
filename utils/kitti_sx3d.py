#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:21:33 2025

@author: arash
"""
from torch.utils.data import Dataset
import utils.data_utils as du
import cv2

class kitti_sx3d(Dataset):
    def __init__(self, cfg, mode = 'train'):
        super(kitti_sx3d, self).__init__()
        self.mode = mode
        self.cfg = cfg
        
        if self.mode == 'train':
            self.data_dir = self.cfg.data.path + 'training/'
            self.DF = du.parse_id_file(self.cfg.data.path + 'train.txt', self.data_dir)
            #if self.cfg.debug:
            #    print(f"==> length of the dataframe is: {len(self.DF)}")
            #    print(f"==> Keys are: {self.DF[10].keys()}")
            self._label_parse()
        elif self.mode == 'val':
            self.data_dir = self.cfg.data.path + 'training/'
            self.DF = du.parse_id_file(self.cfg.data.path + 'val.txt', self.data_dir)
            self._label_parse()
        elif self.mode == 'test':
            self.data_dir = self.cfg.data.path + 'testing/'
            self.DF = du.parse_id_file(self.cfg.data.path + 'test.txt', self.data_dir)
        else:
            raise NotImplementedError("Only test and train available!")
        
        self._calibration_parse()

        
        
    def _calibration_parse(self):
        for instance in self.DF:
            # print(instance["calib_path"])
            instance["calib_params"] = du.parse_calibration(instance["calib_path"])
            
    def _label_parse(self):
        for instance in self.DF:
            instance["labels"] = du.parse_label(instance['label_path'], self.cfg)
            
    def statistics(self):
        assert self.mode != 'test'
        
        x = []
        y = []
        z =[]
        
        min_y = 0
        max_y = 0
        
        for inst in self.DF:
            for det in inst['labels']:
                box = det['bbox3d'][3:6]
                
                if box[1] < min_y:
                    min_y = box[1]
                    min_id = inst["ID"]
                if box[1] > max_y:
                    max_y = box[1]
                    max_id = inst["ID"]
                    
                x.append(box[0])
                y.append(box[1])
                z.append(box[2])
        
        print(f"X statistics - min: {min(x)} - max: {max(x)}")
        print(f"Y statistics - min: {min(y)} - max: {max(y)}")
        print(f"Z statistics - min: {min(z)} - max: {max(z)}")
        print(f"Max Y index: {max_id}, Min Y index: {min_id}")
        
    def __getitem__(self, index):
        instance = self.DF[index]
        
        self.img_l =  cv2.imread(instance['img_l_path'], 1 | 128 )  
        self.img_l_previous =  cv2.imread(instance['img_l_path_previous'], 1 | 128 ) 
        self.img_r =  cv2.imread(instance['img_r_path'], 1 | 128 )
        #self.img_r_previous =  cv2.imread(instance['img_r_path_previous'], 1 | 128 )
        
        self.img_l = cv2.cvtColor(self.img_l, cv2.COLOR_BGR2RGB)
        self.img_l_previous = cv2.cvtColor(self.img_l_previous, cv2.COLOR_BGR2RGB)
        self.img_r = cv2.cvtColor(self.img_r, cv2.COLOR_BGR2RGB)
        #self.img_r_previous = cv2.cvtColor(self.img_r_previous, cv2.COLOR_BGR2RGB)
        
        self.img_l, scale, crop, direction = du.img_resize(self.img_l, self.cfg.model.in_size)
        self.img_l_previous, _, _, _ = du.img_resize(self.img_l_previous, self.cfg.model.in_size)
        self.img_r, _, _, _ = du.img_resize(self.img_r, self.cfg.model.in_size)
        #self.img_r_previous, _, _, _ = du.img_resize(self.img_r_previous, self.cfg.model.in_size)
        
        self.P_l_converted = du.convert_calibration(instance['calib_params']['P2'], scale, crop, direction)
        
        self.img_l = du.img_normalize(self.img_l, self.cfg.data.mean[0], self.cfg.data.std[0])
        self.img_l_previous = du.img_normalize(self.img_l_previous, self.cfg.data.mean[0], self.cfg.data.std[0])
        self.img_r = du.img_normalize(self.img_r, self.cfg.data.mean[0], self.cfg.data.std[0])
        #self.img_r_previous = du.img_normalize(self.img_r_previous, self.cfg.data.mean[0], self.cfg.data.std[0])
        
        data = {"left_img": self.img_l,
                "left_img_previous": self.img_l_previous,
                "right_img": self.img_r,
                "calib": self.P_l_converted.view(1, -1),
                "id": instance["ID"]}
        
        if "labels" in instance.keys():
            data["label"] = instance["labels"]
        
        return data
        
    def __len__(self):
        return len(self.DF)


def test_data():
    from model_cong import config_generator
    cfg = config_generator()
    
    return kitti_sx3d(cfg)