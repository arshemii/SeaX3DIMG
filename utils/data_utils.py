#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:28:34 2025

@author: arash
"""
import numpy as np
import cv2
import torch

def box_generator_2d(detection):
    # order: xmin, ymin, xmax, ymax
    # written in this way fro more readability
    return np.array([detection[4],
                    detection[5],
                    detection[6],
                    detection[7]])

def box_generator_3d(detection):
    # order: h, w, l, cx, cy, cz, yaw
    # written in this way fro more readability
    return np.array([detection[8], detection[9], detection[10],
                    detection[11], detection[12], detection[13],
                    detection[14]])

def parse_id_file(set_path, data_dir):
    result = []
    with open(set_path, 'r') as f:
        for _, line in enumerate(f):
            line = line.strip()
            if line:
                result.append({"ID": line})
                
    for ids in result:
        instance_str = ids["ID"]
        ids["img_l_path"] = data_dir + 'image_2/' + instance_str + '.png'
        ids["img_r_path"] = data_dir + 'image_3/' + instance_str + '.png'
        ids["calib_path"] = data_dir + 'calib/' + instance_str + '.txt'
        if 'train' in data_dir:
            ids["label_path"] = data_dir + 'label_2/' + instance_str + '.txt'
        
    return result

def parse_calibration(calib_path):
    with open(calib_path, 'r') as f:
        parameters = f.readlines()
        params = {}
        for line in parameters:
            if len(line) < 2:
                continue
            line = line.split(' ')
            line[0] = line[0][:-1]
            line[1:] = [float(x) for x in line[1:]]
            params[line[0]] = np.array(line[1:])
            
    calib_params = {
        'P2': np.reshape(params['P2'], [3,4]),
        'P3': np.reshape(params['P3'], [3,4]),
        'R0': np.reshape(params['R0_rect'], [3,3]),
        'V2C': np.reshape(params['Tr_velo_to_cam'], [3,4])}
    
    return calib_params

def parse_label(label_path, cfg):
    
    accepted_occlusion = cfg.data.filter[0]["occl"]
    trunc_threshold = cfg.data.filter[0]["trunc"]
    cl0 = cfg.data.cl0
    cl1 = cfg.data.cl1
    cl2 = cfg.data.cl2
    cl3 = cfg.data.cl3
    cl_ignore = cfg.data.cl4 + cfg.data.cl5
        
    all_det_in_instance = []
    with open(label_path, 'r') as f:
        detection_list = f.readlines()
        for det in detection_list:
            # example detection_line:
                # Misc 0.00 0 -1.82 804.79 167.34 995.43 327.94 1.63 1.48 2.37 3.23 1.59 8.55 -1.47
            if len(det) == 0:
                continue
            det = det.split(' ')
            det[1:] = [float(x) for x in det[1:]]
            
            if det[2] not in accepted_occlusion:
                continue
            if det[1] > trunc_threshold:
                continue
            
            if det[0] in cl_ignore:
                category = -1
            else:
                if det[0] in cl0:
                    category = 0
                elif det[0] in cl1:
                    category = 1
                elif det[0] in cl2:
                    category = 2
                else:
                    category = 3
            
            if len(det) == 16:
                score = det[15]
            else:
                score = None
            
            one_det_in_instance = {
                'category': category,
                'bbox2d': box_generator_2d(det),
                'bbox3d': box_generator_3d(det),
                'truncation': det[1],
                'occlusion': int(det[2]),
                'angle_observation': det[3],
                'score': score}
            all_det_in_instance.append(one_det_in_instance)
            
    return all_det_in_instance

def img_resize(img, target_size):
    scale_1 = target_size[1]/img.shape[1]
    scale_0 = target_size[0]/img.shape[0]
    
    scale = max(scale_1, scale_0)
    
    resized_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    
    if scale_1 > scale_0:
        start_y = (resized_img.shape[0] - target_size[0]) // 2
        end_y = start_y + target_size[0]
        final_img = resized_img[start_y:end_y]
        
        assert final_img.shape[0] == target_size[0] and final_img.shape[1] == target_size[1]
        
        crop = (resized_img.shape[0] - target_size[0]) // 2
        
        return final_img, scale, crop, 'h'
    else:
        start_x = (resized_img.shape[1] - target_size[1]) // 2
        end_x = start_x + target_size[1]
        final_img = resized_img[start_x:end_x]
        
        assert final_img.shape[0] == target_size[0] and final_img.shape[1] == target_size[1]
        
        crop = (resized_img.shape[1] - target_size[1]) // 2
        
        return final_img, scale, crop, 'w'

def convert_calibration(P, scale, crop, direction):
    P = P.copy()
    
    P[0, 0] *= scale  # fx
    P[0, 2] *= scale  # cx
    P[1, 1] *= scale  # fy
    P[1, 2] *= scale  # cy
    P[0, 3] *= scale
    P[1, 3] *= scale
    
    if direction == 'h':
        # Account for vertical crop from top
        P[1, 2] -= crop
    else:
        P[0, 2] -= crop
    
    P = torch.from_numpy(P)
    return P


def img_normalize(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    
    # Convert to tensor (C, H, W)
    img = torch.from_numpy(img).permute(2, 0, 1)
    
    return img