#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:28:34 2025

@author: arash
"""
import numpy as np
import cv2
import torch
import os
import torch.nn.functional as F

def box_generator_2d(detection):
    # order: xmin, ymin, xmax, ymax, yaw
    # written in this way fro more readability
    return np.array([detection[4],
                    detection[5],
                    detection[6],
                    detection[7],
                    detection[14]])


def box_generator_3d(detection):
    # order: h, w, l, cx, cy, cz, yaw
    # written in this way fro more readability
    return np.array([detection[8], detection[9], detection[10],
                    detection[11], detection[12], detection[13],
                    detection[14]])

def box_generator_3d_to_bev(bbox_3d):
    # order: w, l, cx, cz, yaw
    return np.array([bbox_3d[1], bbox_3d[2],
                     bbox_3d[3], bbox_3d[5],
                     bbox_3d[6]])

def parse_id_file(set_path, data_dir):
    result = []
    with open(set_path, 'r') as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            instance_str = line
            label_path = os.path.join(data_dir, 'label_2', instance_str + '.txt')
            prev2_path = os.path.join(data_dir, 'prev_2', instance_str + '_02.png')
            
            # Filter: check if necessary files exist
            if not os.path.exists(prev2_path):
                continue
            if 'train' in data_dir and not os.path.exists(label_path):
                continue

            entry = {
                "ID": instance_str,
                "img_l_path": os.path.join(data_dir, 'prev_2', instance_str + '_01.png'),
                "img_l_path_previous": os.path.join(data_dir, 'prev_2', instance_str + '_02.png'),
                "img_r_path": os.path.join(data_dir, 'prev_3', instance_str + '_01.png'),
                "img_r_path_previous": os.path.join(data_dir, 'prev_3', instance_str + '_02.png'),
                "calib_path": os.path.join(data_dir, 'calib', instance_str + '.txt'),
                "lidar_path": os.path.join(data_dir, 'velodyne', instance_str + '.bin')
            }

            if 'train' in data_dir:
                entry["label_path"] = label_path

            result.append(entry)

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
                category = -2
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
                score = torch.tensor(det[15])
            else:
                score = None
            
            one_det_in_instance = {
                'category': torch.tensor([category]),
                'bbox2d': torch.from_numpy(box_generator_2d(det)),
                'bbox3d': torch.from_numpy(box_generator_3d(det)),
                'truncation': det[1],
                'occlusion': int(det[2]),
                'angle_observation': det[3],
                'score': score}
            bbox_bev = box_generator_3d_to_bev(box_generator_3d(det))
            one_det_in_instance['bbox_bev'] = torch.from_numpy(bbox_bev)
            
            if cfg.short_grid_range:
                x = one_det_in_instance['bbox3d'][3]
                y = one_det_in_instance['bbox3d'][4]
                z = one_det_in_instance['bbox3d'][5]
            
                # Drop objects outside the defined 3D range
                if abs(x) > cfg.grid_size[0] / 2.0:  # beyond left/right limits
                    continue
                if y < cfg.H_min or y > cfg.H_max:   # outside vertical range
                    continue
                if z < 0 or z > cfg.grid_size[2]:    # outside forward depth range
                    continue
            
            all_det_in_instance.append(one_det_in_instance)
            
    return all_det_in_instance


def aggregate_assignment(assignments: torch.Tensor) -> torch.Tensor:
    """
    Collapse [W, H, D] voxel assignments into [W, D] BEV map.

    Rules:
      1. If any object index (>=0) exists → pick one randomly.
      2. If mix of object and ignored (-2) → pick from object ones.
      3. If only ignored (-2) → pick one randomly from ignored.
      4. Else (all -1) → -1 background.
      5. Index -3 has priority to all. even to background
    """
    W, H, D = assignments.shape
    bev = torch.full((W, D), -1, dtype=assignments.dtype, device=assignments.device)

    ass = assignments

    obj_mask = ass >= 0
    ign_mask = ass == -2
    blind_mask = ass == -3

    has_obj = obj_mask.any(dim=1)
    has_ign = ign_mask.any(dim=1)
        
    xs, zs = torch.nonzero(has_obj | has_ign, as_tuple=True)
    for x, z in zip(xs.tolist(), zs.tolist()):
        pillar = ass[x, :, z]
        objs = pillar[pillar >= 0]
        if len(objs) > 0:
            idx = torch.randint(0, len(objs), (1,), device=ass.device)
            bev[x, z] = objs[idx]
        else:
            igs = pillar[pillar == -2]
            if len(igs) > 0:
                idx = torch.randint(0, len(igs), (1,), device=ass.device)
                bev[x, z] = igs[idx]
    
    has_blind = blind_mask.any(dim=1)
    bev[has_blind] = -3
    return bev

def voxel_assigner(label, cfg, debug = False):
    """
    The function produces:
        1. An assignment with the shape 1, W, H, D where:
            a. If voxel is out of boundary --> -3
            b. if voxel belongs to ignore classes --> -2
            c. if voxel is background --> -1
            e. if voxel is an object of interest --> zero to higher (object index)
        ** If a voxel is out of FOV, in any case is must be -3
        
        2. A voxel center tensor to tell that for each obj position in center_voxel (cfg.max_obj poses), what
            is the i, j, and k of the voxel closest to the object center
            
        3, A tensor of True and False showinf if in the tensor position index there is an object or not

    """
    # a 3d space specifying marking of each
    assignments = torch.full((cfg.grid_resolution[0], cfg.grid_resolution[1], cfg.grid_resolution[2]),
                                  fill_value=-1, dtype=torch.long)

    center_voxels = torch.full((cfg.max_obj, 3), fill_value=-1, dtype=torch.long)
    
    valid_obj_mask = torch.full((cfg.max_obj,), fill_value=0, dtype=torch.long)
    
    assert len(label) <= cfg.max_obj
        
    if len(label) == 0:
        # there is no object at ll, mark only background and OOB
        assignments[~cfg.oob_mask_valid[0]] = -3
    else:
        for gt_idx, det in enumerate(label):
            
            valid_obj_mask[gt_idx]= 1
                        
            h, w, l = det['bbox3d'][:3]
            cx, cy, cz = det['bbox3d'][3:6]
            cat = int(det['category'])
            
            if debug:
                print(f"before {w}, {h}, {l}")
            
            # if object is smaller than an edge of the voxel:
            w = torch.maximum(w, torch.tensor(cfg.grid_unc[0] * 1.02, device=w.device, dtype=w.dtype))
            h = torch.maximum(h, torch.tensor(cfg.grid_unc[1] * 1.02, device=h.device, dtype=h.dtype))
            l = torch.maximum(l, torch.tensor(cfg.grid_unc[2] * 1.02, device=l.device, dtype=l.dtype))
            
            if debug:
                print(f"after {w}, {h}, {l}")
            
            # AABB
            x_min, x_max = cx - w / 2, cx + w / 2
            y_min, y_max = cy - h / 2, cy + h / 2
            z_min, z_max = cz - l / 2, cz + l / 2
            
            if debug:
                print(f"object range x is: {x_min}, {x_max}")
                print(f"object range y is: {y_min}, {y_max}")
                print(f"object range z is: {z_min}, {z_max}")
            

            # mask voxels inside this object
            xs, ys, zs = cfg.grid_forward[0][..., 0], cfg.grid_forward[0][..., 1], cfg.grid_forward[0][..., 2]
            
            if debug:
                print(f"grid x shape --> {xs.shape}")
                print(f"grid point distances: {cfg.grid_unc[0]}, {cfg.grid_unc[1]}, {cfg.grid_unc[2]}")
                print(f"grid x range is: {xs.min()}, {xs.max()}")
                print(f"grid y range is: {ys.min()}, {ys.max()}")
                print(f"grid z range is: {zs.min()}, {zs.max()}")
            
            inside = (xs >= x_min) & (xs <= x_max) & \
                     (ys >= y_min) & (ys <= y_max) & \
                     (zs >= z_min) & (zs <= z_max)
            
            
            
            assert inside.sum() != 0
    
            # assign voxel values
            if cat == cfg.data.ignore_class_id:
                assignments[inside] = cfg.data.ignore_class_id
            else:
                assignments[inside] = gt_idx
                
            # find voxel closest to GT center
            voxel_coords = cfg.grid_forward[0][inside].to(cx.device)
            dists = torch.norm(voxel_coords - det['bbox3d'][3:6].to(cx.device), dim=1)
            min_idx = torch.argmin(dists)
            idx_flat = torch.nonzero(inside, as_tuple=False)[min_idx]
            i, j, k = idx_flat.tolist()
            
            center_voxels[gt_idx, :] = torch.tensor([i, j, k])
        assignments[~cfg.oob_mask_valid[0]] = -3
        
    # assignment_bev = aggregate_assignment(assignments)
        
    return assignments, center_voxels, valid_obj_mask
    

def img_resize(img, target_size):
    scale_1 = target_size[1]/img.shape[1]
    scale_0 = target_size[0]/img.shape[0]
    
    scale = max(scale_1, scale_0)
    
    resized_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    
    if scale_1 > scale_0:  # need vertical crop
        start_y = (resized_img.shape[0] - target_size[0]) // 2
        end_y = start_y + target_size[0]
        final_img = resized_img[start_y:end_y, :]
        crop = start_y
        direction = 'h'
    else:  # need horizontal crop
        start_x = (resized_img.shape[1] - target_size[1]) // 2
        end_x = start_x + target_size[1]
        final_img = resized_img[:, start_x:end_x]
        crop = start_x
        direction = 'w'
        
    assert final_img.shape[0] == target_size[0] and final_img.shape[1] == target_size[1]
    return final_img, scale, crop, direction

def test_resize(img_path='/home/arash/SeaX3DIMG/dataset/training/image_2/000057.png'):
    import matplotlib.pyplot as plt
    import cv2

    img_l = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

    target_sizes = [(200, 1242), (50, 1200), (375, 1000), (300, 300)]
    imgs, infos = [img_l], ["Original"]

    for tg_size in target_sizes:
        resized, scale, crop, direction = img_resize(img_l, tg_size)
        imgs.append(resized)
        infos.append(f"{tg_size} | scale={scale:.2f} | crop={crop} ({direction})")

    plt.figure(figsize=(18, 4))
    for i, im in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(im)
        plt.title(infos[i], fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def convert_calibration(P, scale, crop, direction):
    P = P.copy()

    # scale intrinsics
    P[0,0] *= scale
    P[1,1] *= scale
    P[0,2] *= scale
    P[1,2] *= scale

    # scale baseline translation if nonzero
    P[0,3] *= scale
    P[1,3] *= scale

    # adjust for crop offset
    if direction == 'h':  # vertical crop
        P[1,2] -= crop
    else:                 # horizontal crop
        P[0,2] -= crop

    return torch.from_numpy(P)

def test_calib_transform(target_size = (288, 960),
                         calib_path = '/home/arash/SeaX3DIMG/dataset/training/calib/000057.txt',
                         img_path='/home/arash/SeaX3DIMG/dataset/training/image_2/000057.png'):
    
    P = parse_calibration(calib_path)['P2']
    
    import cv2
    img_l = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    
    print(f"Original P for shape: {np.shape(img_l)} is: ")
    print(P)
    
    final_img, scale, crop, direction = img_resize(img_l, target_size)
    
    P_conv = convert_calibration(P, scale, crop, direction)
    
    print(f"Converted P for shape: {target_size} is: ")
    print(P_conv)
    
    return P_conv


def img_normalize(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    
    # Convert to tensor (C, H, W)
    img = torch.from_numpy(img).permute(2, 0, 1)
    
    return img.to(torch.float32)

def project_ref_to_rect(pts_3d_ref, R0):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))

def cart2hom(pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

def project_velo_to_ref(pts_3d_velo, V2C):
        pts_3d_velo = cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(V2C))

def project_velo_to_rect(pts_3d_velo, cfg):
        pts_3d_ref = project_velo_to_ref(pts_3d_velo, cfg.camera.V2C[0])
        return project_ref_to_rect(pts_3d_ref, cfg.camera.R0[0])

def project_rect_to_image(pts_3d_rect, P2):
    """Project rectified camera coordinates to image plane."""
    pts_3d_hom = cart2hom(pts_3d_rect)       # [N,4]
    pts_2d = np.dot(pts_3d_hom, P2.T)        # [N,3]
    pts_2d[:, 0] /= pts_2d[:, 2]             # u = x / z
    pts_2d[:, 1] /= pts_2d[:, 2]             # v = y / z
    
    a = pts_2d[:, :2]
    b = pts_3d_rect[:, 2]
    
    return a, b

def create_depth_map(pcc, P2, im_shape=(375, 1242)):
    pts_img, depth = project_rect_to_image(pcc, P2)
    u, v = pts_img[:, 0], pts_img[:, 1]

    # Remove NaN / Inf depth
    valid = (~np.isnan(u)) & (~np.isnan(v)) & (~np.isinf(u)) & (~np.isinf(v)) & (depth > 0)
    u, v, depth = u[valid], v[valid], depth[valid]

    # Round to nearest pixel indices
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    # Filter valid points inside image bounds
    valid = (u >= 0) & (v >= 0) & (u < im_shape[1]) & (v < im_shape[0])
    u, v, depth = u[valid], v[valid], depth[valid]

    depth_map = np.zeros(im_shape, dtype=np.float32)

    for i in range(len(depth)):
        if depth_map[v[i], u[i]] == 0 or depth[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = depth[i]

    return torch.from_numpy(depth_map)

def pcl_as_depth(pcl_path, cfg):
    pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # [N, 3] forward dis, left distance, up distance
    
    pcc = project_velo_to_rect(pcl, cfg)
    
    pcc_img = create_depth_map(pcc, cfg.camera.P_l[0], cfg.model.in_size)
    
    gt_disp = (cfg.camera.focal[0] * cfg.camera.base[0]) / pcc_img
    
    gt_disp[pcc_img == 0] = 0  # avoid division by zero
    
    gt_disp = gt_disp.unsqueeze(0).unsqueeze(0) 
    
    pcl_down = F.interpolate(gt_disp, scale_factor=0.25, mode='bilinear', align_corners=False)
    
    return pcl_down.squeeze(0).squeeze(0) 
    
def get_focal_baseline(P2):

    f = P2[0, 0]

    B = -P2[0, 3] / f

    return f, B

def collate_fn(batch):
    # a batch is a list
    images_l = torch.stack([item['left_img'] for item in batch])          # [B, 3, H, W]
    images_l_p = torch.stack([item['left_img_previous'] for item in batch])
    images_r = torch.stack([item['right_img'] for item in batch])
    images_r_p = torch.stack([item['right_img_previous'] for item in batch])
    # calib_left = torch.stack([item['calib'] for item in batch], dim=0)  # a 12-value each row of P
    
    batch_dict = {
        "left_img": images_l.to(dtype=torch.float32),
        "left_img_previous": images_l_p.to(dtype=torch.float32),
        "right_img": images_r.to(dtype=torch.float32),
        "right_img_previous": images_r_p.to(dtype=torch.float32),
        # "calib": calib_left.to(dtype=torch.float32)
    }

    if "label" in batch[0].keys():
        batch_dict['assignment'] = torch.stack([item['assignment'] for item in batch])
        batch_dict['disparity'] = torch.stack([item['depth'] for item in batch])
       #  batch_dict["assignment_bev"] = torch.stack([item['assignment_bev'] for item in batch])
        
        max_objects = batch[0]['valid_obj'].shape[0] # from dataset statistics
        feature_number = (7   # bbox 3d
                          + 1  # category
                          + 3  # i, j, k of closest voxel
                          + 1)  # valid mask
        
        labels = torch.zeros((len(images_l), max_objects, feature_number), dtype=torch.float32)
        
        for batch_num, item in enumerate(batch):
            label_p_f = item["label"]
            labels[batch_num, :, 8:11] = item['center_voxel'].to(dtype=torch.float32)
            labels[batch_num, :, 11] = item['valid_obj'].to(dtype=torch.float32)
            for idx, obj in enumerate(label_p_f):
                bbox3d = torch.as_tensor(obj['bbox3d'], dtype=torch.float32)
                bbox3d = bbox3d[[1, 0, 2, 3, 4, 5, 6]]
                labels[batch_num, idx, 0:7] = bbox3d
                # labels[batch_num, idx, 7:12] = obj['bbox_bev'].to(dtype=torch.float32)
                labels[batch_num, idx, 7] = obj['category'].to(dtype=torch.float32)
            
        batch_dict['label'] = labels
        
    return batch_dict
        
    
    
    
    
    
    
    
    
