#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 18:56:11 2025

@author: arash
"""
import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils.eval_utils as ev

class Evaluator():
    def __init__(self, cfg, model, dataset, collate_fn):
        
        self.cfg = cfg
        self.batch_size = self.cfg.eval.batch_size
        self.num_workers = self.cfg.eval.num_workers
        self.device = self.cfg.device[0]
        self.model = model
        self.save_dir = self.cfg.eval.save_dir
        self.save_dir_GT = self.cfg.eval.save_dir_gt
        self.kernel = self.cfg.eval.local_maxima_kernel
        self.threshold = self.cfg.eval.score_th
        self.k = self.cfg.eval.topk
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                     collate_fn = collate_fn, num_workers=self.num_workers)


    def out_preparation(self, obj_out):
        obj, classes, offset, dims, yaw = obj_out
        
        scores = F.sigmoid(obj).to('cpu')
        
        probs = F.softmax(classes, dim=1).to('cpu')
        max_probs, max_indices = torch.max(probs, dim=1)
        
        grid = self.cfg.grid_forward[0].permute(3, 0, 1, 2).unsqueeze(0)
        centers = offset.to('cpu')
        scales = torch.tensor(self.cfg.grid_unc, dtype=centers.dtype, device=centers.device).view(1, 3, 1, 1, 1)
        centers = centers * scales
        centers = centers + grid

        dims = dims.to('cpu')
        yaw = math.pi * yaw.to('cpu')

        return scores, max_probs, max_indices, centers, dims, yaw

    def create_per_sample_out(self, obj, cls_prob, cls_id, cnt, dim, yaw, sample_id):
        # obj: [1,1,W,H,D]
        indices_sorted, scores_sorted = ev.local_maximum_3d(objectness=obj, kernel=self.kernel,
                                                            score_thr=self.threshold, topk=self.k)
        # TODO: Non-max supression alternative
        #indices_sorted, scores_sorted = ev.nms_python(objectness=obj, kernel=self.kernel,
                                                            #score_thr=self.threshold, topk=self.k)
    
        # Prepare output path
        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, f"{int(sample_id):06d}.txt")
    
        if indices_sorted.numel() == 0:
            open(out_path, "w").close()
            return
    
        b_idx, c_idx, w_idx, h_idx, d_idx = indices_sorted.T  # should have [N]
    
        cls_ids = cls_id[0, w_idx, h_idx, d_idx]          # [N]
        cls_scores = cls_prob[0, cls_ids, w_idx, h_idx, d_idx]  # [N] # TODO: shoudl I merge?
    
        centers = cnt[b_idx, :, w_idx, h_idx, d_idx].T        # [N, 3] (x,y,z)
        dims    = dim[b_idx, :, w_idx, h_idx, d_idx].T        # [N, 3] (h,w,l)
        yaws    = yaw[b_idx, 0, w_idx, h_idx, d_idx]          # [N]
        scores  = scores_sorted * cls_scores                  # [N]
    
        P2 = self.cfg.camera.P_l[0]
    
        lines = []
        for i in range(centers.shape[0]):
            if self.cfg.eval.estimate_2d:
                lines.append(ev.create_prediction_line_est2d(centers[i], dims[i],
                                                             yaws[i], scores[i], cls_ids[i], P2,
                                                             self.cfg.eval.class_names))
            else:
                raise NotImplementedError("An accurate 2d box prediction is not provided yet!")
    
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
            
    def modify_per_sample_gt(self, label_path: str):
        """
        Read a KITTI-style GT label file and write a remapped version into
        self.save_dir_GT, such that classes match the model's reduced set:

            - 'Car', 'Van'              -> 'Car'
            - 'Truck'                   -> 'Truck'
            - 'Pedestrian'              -> 'Person'
            - 'Cyclist'                 -> 'Cyclist'
            - 'DontCare'                -> 'DontCare' (kept for eval)
            - all other classes         -> dropped

        All other fields (truncation, occlusion, 2D bbox, 3D box) are left untouched.
        """
        os.makedirs(self.save_dir_GT, exist_ok=True)

        if not isinstance(label_path, str):
            label_path = str(label_path)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"GT label file not found: {label_path}")

        # Output file name mirrors original KITTI naming (e.g. 000123.txt)
        base_name = os.path.basename(label_path)
        out_path = os.path.join(self.save_dir_GT, base_name)

        with open(label_path, "r") as f:
            lines = f.readlines()

        remapped_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            cls_ = fields[0]
            
            # remove out of the range objects
            if getattr(self.cfg, "short_grid_range", False) and cls_ != "DontCare":
                try:
                    x = float(fields[11])
                    y = float(fields[12])
                    z = float(fields[13])
                except (IndexError, ValueError):
                    continue
            
                # Drop objects outside the defined 3D range
                if abs(x) > self.cfg.grid_size[0] / 2.0:  # beyond left/right limits
                    continue
                if y < self.cfg.H_min or y > self.cfg.H_max:   # outside vertical range
                    continue
                if z < 0 or z > self.cfg.grid_size[2]:         # outside forward depth range
                    continue

            # Map original KITTI classes to your reduced set
            if cls_ in ("Car", "Van"):
                new_cls = "Car"
            elif cls_ == "Truck":
                new_cls = "Truck"
            elif cls_ in ("Pedestrian"):
                new_cls = "Person"   # matches cfg.eval.class_names[2]
            elif cls_ == "Cyclist":
                new_cls = "Cyclist"
            elif cls_ == "DontCare":
                new_cls = "DontCare"
            else:
                # Classes not modeled by your network (e.g. 'Tram', 'Misc', 'Motorcyclist')
                # are dropped from the GT for this evaluation.
                continue

            fields[0] = new_cls
            remapped_lines.append(" ".join(fields))

        # If nothing remains, write an empty file (valid for the evaluator)
        with open(out_path, "w") as f:
            if remapped_lines:
                f.write("\n".join(remapped_lines))
      
    def generate_labels(self):
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Modified label creation.")
        
        for batch_idx, batch in pbar:
            labels_path = batch["label_path"] # a list of batch size
            
            for label_path in labels_path:
                self.modify_per_sample_gt(label_path)
                
            pbar.set_postfix({'Nothing': "Nothing"})
      
    def generate_predictions(self):
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Prediction on dataset.")
        
        for batch_idx, batch in pbar:
                        
            batch["left_img"] = batch["left_img"].to(self.device)
            batch["right_img"] = batch["right_img"].to(self.device)
            samples_idx = batch['id']
            
            with torch.no_grad():
                outputs, _ = self.model(batch["left_img"], batch["right_img"])
              
            
            scores, max_probs, max_indices, centers, dims, yaw = self.out_preparation(outputs)
            
            for b, sample_id in enumerate(samples_idx):
                obj_b   = scores[b:b+1]      # [1,1,W,H,D]
                prob_b  = max_probs[b:b+1]   # per-voxel class prob for chosen class
                clsid_b = max_indices[b]     # [W,H,D]
                cnt_b   = centers[b:b+1]
                dim_b   = dims[b:b+1]
                yaw_b   = yaw[b:b+1]
    
                self.create_per_sample_out(obj=obj_b,
                                        cls_prob=prob_b,
                                        cls_id=clsid_b,
                                        cnt=cnt_b,
                                        dim=dim_b,
                                        yaw=yaw_b,
                                        sample_id=sample_id)
                
    
    def eval(self):
        raise NotADirectoryError("must be completed")