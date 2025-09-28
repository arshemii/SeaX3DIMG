#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 17:38:07 2025

@author: arash
"""

import torch
from scipy.optimize import linear_sum_assignment
import utils.eval_utils as eu

class Evaluate:
    def __init__(self, cfg, prediction, gt_all, grid):
        super(Evaluate, self).__init__()
        """
        prediction is a list of all batches:
            each element is a torch.tensor on CPU with (num_batch, 10, res_w, res_z)
        
        gt_all is a list where each element is calles gt:
            gt is is a list (length is num_batch) where for each gt in gt[index]:
                gt['category'] = object class (zero to num_classes-1 and -1 for not important objects)
                gt['bbox3d'] = order is: h, w, l, cx, cy, cz, yaw
                and mut be present:
                    gt['bbox_bev'] order is w, l, cx, cz, yaw
            
        grid is:
            the grid with x, y, z of each voxel to match prediction with gt objects
        """
        self.cfg = cfg
        
        self.grid = eu.grid3d_to_grid2d(grid)  # TODO: should I move to CPU? No. I will use GPU!
        self.prediction = prediction
        self.gt_all = gt_all
        
        
    def _pred_preparation(self, preds):
        # only for a batch of detection
        
        # Normalize objectness as probability
        preds[:, 0, :, :] = torch.sigmoid(preds[:, 0, :, :])
        
        # Transform class logits to probabilities
        preds[:, 1:5, :, :] = torch.softmax(preds[:, 1:5, :, :], dim=1)
        
        # convert classes to max class index and max class prob
        preds = eu.compress_tensor(preds)
        
        # convert prediction to have absolute detection centers instead of offsets
        preds = eu.add_voxel_centers(preds, self.grid)
        
        # Drop out detectionsfor low detection scores
        preds_list = eu.filter_detections(preds, self.cfg.eval.objectness_threshold)
        
        # drop detections out of the range limit (to evaluate the model in different ranges)
        if self.cfg.eval.range_limit:
            preds_list = eu.drop_far_dets(preds_list, self.cfg.eval.range_limit)
            
        return preds_list    
        
    def _gt_preparation(self, gt):
        # only for a batch of detection
        # drop objects out of the range limit (to evaluate the model in different ranges)
        if self.cfg.eval.range_limit:
            gt = eu.drop_far_gts(gt, self.cfg.eval.range_limit)
        
        return gt
    
    def hungarian_matching(self, prediction, gt_list, threshold):
        """
        prediction : a tensor of shape N, 8
        gt_list : A list of M objects each with bbox_bev and category

        Returns: # TODO
        """
        device = prediction.device
        dtype = torch.float32
        
        # a tensor for IoU calculation per each pred and gt
        IoUs = torch.zeros(len(prediction), len(gt_list), dtype=dtype, device = device)
        
        # class mismatch cost
        class_cost = torch.zeros(len(prediction), len(gt_list), device = prediction.device())
        
        for m in range(len(gt_list)):
            bbox_bev_gt = torch.tensor(gt_list[m]['bbox_bev'], dtype=dtype, device=device) # tensor w, l, cx, cz, yaw
            bbox_bev_gt = bbox_bev_gt[[2, 3, 0, 1, 4]] # cx, cz, w, l, yaw
            
            IoUs[:, m] = eu.bev_iou(prediction[:, 3:8], bbox_bev_gt)
            
            if self.cfg.eval.is_class_cost:
                for n in range(len(prediction)):
                    if int(prediction[n, 1]) == int(gt_list[m]["category"]):
                        class_cost[n, m] = 0.0
                    else:
                        class_cost[n, m] = 1.0
        
        if self.cfg.eval.is_class_cost:
            # total cost with weights for each cost term
            Cost = self.cfg.eval.match_cost_dist[0] * (1.0 - IoUs) +\
                self.cfg.eval.match_cost_dist[1] * class_cost
        else:
            Cost = 1.0 - IoUs
        
        # hungarian
        pred_indices, gt_indices = linear_sum_assignment(Cost.cpu().numpy())
        
        matches = []
        for p, g in zip(pred_indices, gt_indices):
            if IoUs[p, g] >= threshold:
                matches.append((p, g))   # prediction p matched with GT g
            
        matched_preds = {p for p, _ in matches}
        matched_gts   = {g for _, g in matches}
        
        unmatched_preds = [i for i in range(len(prediction)) if i not in matched_preds]
        unmatched_gts   = [j for j in range(len(gt_list)) if j not in matched_gts]
        
        
        return matched_gts, unmatched_preds, unmatched_gts

    def eval_loop(self):
        
        assert len(self.gt_all) == len(self.prediction)
        
        num_total_obj = 0
        
        FP_all_iou = [0 for i in range(len(self.cfg.eval.iou_list))]
        FN_all_iou = [0 for i in range(len(self.cfg.eval.iou_list))]
        TP_all_iou = [0 for i in range(len(self.cfg.eval.iou_list))]
        
        for idx, gt_batch in enumerate(self.gt_all):
            # find preds and gts for a single batch of data
            self.pred = self._pred_preparation(self.prediction[idx])
            self.gt = self._gt_preparation(gt_batch)
            
            for i in range(self.cfg.num_batch):
                # preds and gt of a sample in a batch
                
                FP = 0
                FN = 0
                TP = 0
                
                num_preds = len(self.pred[i])
                num_gts = len(self.gt[i])
                
                num_total_obj += num_gts
                
                if num_preds == 0:
                    if num_gts == 0:
                        continue
                    else:
                        # we have GTs that are not detected!
                        FN += num_gts
                else:
                    if num_gts == 0:
                        # we have objects wrongly have been detected!
                        FP += num_preds
                    else:
                        # matching based on each  IoU threshold
                        for iou_idx, iou_th in enumerate(self.cfg.eval.iou_list):
                            # we have some detections and some GTs
                            matches, unmatched_preds_idx, unmatched_gts_idx = self.hungarian_matching(self.pred[i], self.gt[i], iou_th)
                                
                            TP_all_iou[iou_idx] += TP + len(matches)
                            FP_all_iou[iou_idx] += FP + len(unmatched_preds_idx)
                            FN_all_iou[iou_idx] += FN + len(unmatched_gts_idx)
        
        return num_total_obj, FP_all_iou, FN_all_iou, TP_all_iou