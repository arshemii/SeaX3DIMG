#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 18:56:11 2025

@author: arash
"""

# evaluator.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import utils.eval_utils as eu

class Evaluate:
    """
    Evaluation helper that:
      - prepares predictions (sigmoid/softmax -> compress -> add voxel centers -> filter),
      - runs Hungarian matching per-sample for counting TP/FP/FN per IoU threshold (class-aware),
      - builds per-class detections and GTs and computes AP (per IoU),
      - returns per-class stats and mAP per IoU.
    Config expectations (self.cfg.eval):
      - objectness_threshold: float
      - range_limit: float or None
      - iou_list: list of IoU thresholds (e.g. [0.5])
      - num_batch: number of samples per batch tensor
      - require_class_match: bool (True forces predicted class == GT class to count TP)
      - match_cost_dist (unused here; Hungarian uses IoU + optional class constraint)
    """

    def __init__(self, cfg, prediction_batches, gt_all_batches, grid, oob_mask_valid):
        """
        prediction_batches: list of batch tensors; each tensor shaped [B, 10, W, D] (CPU or GPU)
        gt_all_batches: list parallel to prediction_batches; each item is a list of length B,
                        where each entry is a list of GT dicts for that sample
        grid: full voxel grid [3, W, H, D] or similar (we expect grid3d_to_grid2d to turn it to [W,D,2])
        """
        self.cfg = cfg
        self.prediction_batches = prediction_batches
        self.gt_all_batches = gt_all_batches
        self.num_c = self.cfg.model.num_class
        self.oo_fov = oob_mask_valid.cpu()

        # convert 3D grid -> BEV xy centers [W, D, 2]
        # NOTE: keep grid on CPU because eu.bev_iou uses shapely (CPU)
        self.grid_3d = grid.cpu()
        bev_grid = eu.grid3d_to_grid2d(grid)
        self.grid = bev_grid.cpu() if isinstance(bev_grid, torch.Tensor) else bev_grid

    # -------------------------
    # Prediction / GT helpers
    # -------------------------
    def _pred_preparation(self, batch_preds):
        """
        Input:
            batch_preds: tensor [B, 10, W, D] (logits):
                Order in second dim: cl1, cl2, cl3, cl4, Objectness, cx_offset, cz_offset, w, l, yaw
        Output:
            preds_list: python list length B, each element is tensor [N, 8] detections:
                Order in second dim: objectness, class_idx, class_prob, cx_abs, cz_abs, w, l, yaw
        """
        preds = batch_preds.clone()  # avoid in-place changes to original tensors

        # We will run IoUs on CPU using eu.bev_iou; compress/add_voxel_centers assume tensors
        preds = preds.to(self.cfg.eval.eval_device[0])

        # objectness -> sigmoid
        preds[:, self.num_c:self.num_c+1, :, :] = torch.sigmoid(preds[:, self.num_c:self.num_c+1, :, :])

        # class logits [1:5]
        preds[:, 1:5, :, :] = torch.softmax(preds[:, 0:self.num_c, :, :], dim=1)

        # compress: [B,10,W,D] -> [B,8,W,D]
        preds = eu.compress_tensor(preds, self.num_c)
        # oder: objectness, class_idx, class_prob, regressed

        # add absolute voxel centers (grid must be [W,D,2])
        preds = eu.add_voxel_centers(preds, self.grid)

        # filter by objectness threshold -> list of [N,8] per sample
        preds_list = eu.filter_detections(preds, self.cfg.eval.objectness_threshold)

        # optionally drop far detections
        if self.cfg.eval.range_limit:
            preds_list = eu.drop_far_dets(preds_list, self.cfg.eval.range)

        return preds_list

    def _gt_preparation(self, batch_gt):
        """
        Input:
            batch_gt: list length B, each element is a list of gt dicts for that sample
        Output:
            prepared_gt: same shape, but optionally filtered by range_limit
        """
        batch_gt = eu.mark_nonvalid_obj(self.grid_3d, batch_gt, self.oo_fov, self.cfg.grid_unc)
        # all oo fov and ignored are marked as invalid
        
        if self.cfg.eval.range_limit:
            return eu.mark_far_gts(batch_gt, self.cfg.eval.range)
        return batch_gt


    def hungarian_matching(self, preds: torch.Tensor, gt_valid: torch.Tensor, iou_threshold: float):
        """
        preds: [N, 8]  (objectness, class_idx, class_prob, cx, cz, w, l, yaw)
        gt_tensor: [num_objects, 14]  (includes padding; only valid if gt_tensor[:, -1] == 1)
           - bbox fields: gt_tensor[:, 7:12] -> (w, l, cx, cz, yaw)
           - category: gt_tensor[:, 12]
           - flag: gt_tensor[:, -1] == 1.0 means valid object
    
        Returns:
            matches: list of (pred_idx, gt_idx)
            unmatched_pred_indices: list of pred indices
            unmatched_gt_indices: list of gt indices
        """
        N = len(preds)
        M = len(gt_valid)
    
        # Early return if no predictions or no ground truth
        if N == 0 or M == 0:
            return [], list(range(N)), list(range(M))
    
        device = preds.device
        dtype = torch.float32
    
        IoUs = torch.zeros((N, M), dtype=dtype, device=device)
    
        # Prediction boxes: [cx, cz, w, l, yaw]
        pred_boxes = preds[:, 3:8].to(device, dtype=dtype)
    
        # GT boxes: reorder from [w, l, cx, cz, yaw] to [cx, cz, w, l, yaw]
        gt_boxes = gt_valid[:, 7:12][:, [2, 3, 0, 1, 4]].to(device, dtype=dtype)
    
        # Compute IoU for all pairs (pred, gt)
        for m in range(M):
            gt_box = gt_boxes[m]
            IoUs[:, m] = eu.bev_iou(pred_boxes, gt_box)
    
        # Cost = 1 - IoU
        cost = (1.0 - IoUs).cpu().numpy()
    
        # Solve Hungarian
        pred_indices, gt_indices = linear_sum_assignment(cost)
    
        # Filter matches by IoU threshold
        matches = []
        matched_pred_set = set()
        matched_gt_set = set()
        for p, g in zip(pred_indices, gt_indices):
            iou_val = float(IoUs[p, g].item())
            if iou_val < iou_threshold:
                continue
            matches.append((int(p), int(g)))
            matched_pred_set.add(int(p))
            matched_gt_set.add(int(g))
    
        unmatched_preds = [i for i in range(N) if i not in matched_pred_set]
        unmatched_gts = [j for j in range(M) if j not in matched_gt_set]
    
        return matches, unmatched_preds, unmatched_gts

    # -------------------------
    # Utility: compute AP for one class across dataset (greedy score-based matching)
    # -------------------------
    def _compute_ap_for_class(self, detections, gt_by_image, iou_th):
        """
        Args:
            detections: list of dict {image_id: int, score: float, bbox: [cx,cz,w,l,yaw] (python list or np)}
                This list should contain only detections predicted as this class (all images).
            gt_by_image: dict image_id -> list of gt bbox (each [cx,cz,w,l,yaw]) for this class
            iou_th: float

        Returns:
            ap: float (average precision)
            precision, recall, (optional arrays) - we return AP only here to keep it concise
        """
        if len(detections) == 0:
            # Wrongly no detection means AP == 0
            return 0.0

        # How many times class C is present in an object over all samples
        n_gt_c = sum(len(v) for v in gt_by_image.values())

        # sort detections by score descending
        detections_sorted = sorted(detections, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(detections_sorted), dtype=np.float32)
        fp = np.zeros(len(detections_sorted), dtype=np.float32)

        # to check if a gt object is matched with a higher score preds:
        matched_gt_flags = {img: np.zeros(len(gt_by_image.get(img, [])), dtype=bool) for img in gt_by_image.keys()}

        for i, det in enumerate(detections_sorted):
            img = det['image_id']
            bbox_det = torch.tensor(det['bbox'], dtype=torch.float32)
            gt_list = gt_by_image.get(img, [])

            if len(gt_list) == 0:
                # no GT of this class in this image -> FP
                fp[i] = 1.0
                continue

            # compute IoU of this detection with all GTs in this image
            ious = []
            for gt_idx, gt_bbox in enumerate(gt_list):
                gt_t = torch.tensor(gt_bbox, dtype=torch.float32)
                iou_val = float(eu.bev_iou(bbox_det.unsqueeze(0), gt_t).item())
                ious.append(iou_val)
            ious = np.array(ious, dtype=np.float32)

            # find best match
            best_idx = ious.argmax() if ious.size > 0 else -1
            best_iou = ious[best_idx] if ious.size > 0 else 0.0

            if best_iou >= iou_th:
                if not matched_gt_flags[img][best_idx]:
                    # TP: mark this gt matched
                    tp[i] = 1.0
                    matched_gt_flags[img][best_idx] = True
                else:
                    # this gt already matched -> duplicate detection => FP
                    fp[i] = 1.0
            else:
                # FL
                fp[i] = 1.0

        # cumulative sums
        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)

        if n_gt_c == 0:
            recall = tp_cum * 0.0
        else:
            recall = tp_cum / float(n_gt_c)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)

        # AP - interpolated precision
        # make precision monotonically decreasing
        mpre = np.concatenate(([0.0], precision, [0.0]))
        mrec = np.concatenate(([0.0], recall, [1.0]))
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        # integrate area under curve
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)


    def evaluate(self):
        """
        Return:
            mAP
            AP per class
            recall and Precision per class
        """
        iou_list = self.cfg.eval.iou_list # default is: [0.10, 0.25, 0.50, 0.75, 0.90]

        class_ids = list(range(self.cfg.model.num_class))
        metrics_per_iou = {iou: {'per_class_counts': {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in class_ids},
                                 'detections_by_class': {c: [] for c in class_ids},
                                 'gt_by_class': {c: defaultdict(list) for c in class_ids}}
                           for iou in iou_list}

        # Image id in global scope
        global_image_id = 0

        assert len(self.prediction_batches) == len(self.gt_all_batches), "prediction and gt batch count mismatch"
        
        for batch_idx, (batch_preds, batch_gts) in tqdm(
            enumerate(zip(self.prediction_batches, self.gt_all_batches)),
            total=len(self.prediction_batches), desc="Collectinbg statistics:"):
            
            # prepare batch-level preds and gts
            prepared_preds_list = self._pred_preparation(batch_preds)  # list length B of [N,8] tensors
            prepared_gt_list = self._gt_preparation(batch_gts)  # list length B of tensors B, 18, 14

            # iterate samples in this batch
            for sample_idx in range(self.cfg.num_batch):
                preds = prepared_preds_list[sample_idx]  # [N,8] or empty tensor (0,8)
                gts_full = prepared_gt_list[sample_idx]       # tensor 18, 14
                valid_mask = gts_full[:, -1] == 1.0
                gts = gts_full[valid_mask]

                # For each IoU threshold, run class-aware Hungarian to get TP/FP/FN per class
                for iou_th in iou_list:
                    matches, unmatched_pred_idx, unmatched_gt_idx = self.hungarian_matching(
                        preds, gts, iou_threshold=iou_th)

                    # Count matches per class (TP)
                    for p_idx, g_idx in matches:
                        gt_cls = int(gts[g_idx][-2])
                        pred_cls = int(preds[p_idx, 1].item())
                        if pred_cls == gt_cls:
                            metrics_per_iou[iou_th]['per_class_counts'][gt_cls]['TP'] += 1
                        else:
                            # Wrong class → penalize both preds and GT
                            # TODO: what is a common practice here? which to penalize?
                            metrics_per_iou[iou_th]['per_class_counts'][pred_cls]['FP'] += 1
                            metrics_per_iou[iou_th]['per_class_counts'][gt_cls]['FN'] += 1

                    # Unmatched preds -> FP per predicted class
                    for p_idx in unmatched_pred_idx:
                        pred_cls = int(preds[p_idx, 1].item())
                        metrics_per_iou[iou_th]['per_class_counts'][pred_cls]['FP'] += 1

                    # Unmatched gts -> FN per gt class
                    for g_idx in unmatched_gt_idx:
                        gt_cls = int(gts[g_idx][-2])
                        metrics_per_iou[iou_th]['per_class_counts'][gt_cls]['FN'] += 1

                # Collect detections and GTs (for AP computation) across IoUs (AP computed per IoU later)
                if preds.numel() > 0:
                    scores = (preds[:, 0] * preds[:, 2]).cpu().numpy()  # objectness * class_prob
                    boxes = preds[:, 3:8].cpu().numpy()  # cx, cz, w, l, yaw
                    pred_classes = preds[:, 1].long().cpu().numpy()
                    for i_det in range(preds.shape[0]):
                        c = int(pred_classes[i_det])
                        entry = {'image_id': global_image_id, 'score': float(scores[i_det]), 'bbox': boxes[i_det].tolist()}
                        for iou_th in iou_list:
                            metrics_per_iou[iou_th]['detections_by_class'][c].append(entry)

                for g_idx in range(len(gts)):
                    c = int(gts[g_idx][-2])
                    bbox_bev = gts[g_idx][7:12]
                    bbox_reordered = [bbox_bev[2], bbox_bev[3], bbox_bev[0], bbox_bev[1], bbox_bev[4]]
                    for iou_th in iou_list:
                        metrics_per_iou[iou_th]['gt_by_class'][c][global_image_id].append(bbox_reordered)

                global_image_id += 1

        print("-------------------Start evaluation---------------------")    

        # AP per class per IoU threshold
        results = {}
        for iou_th in iou_list:
            per_class_results = {}
            aps = []
            for c in class_ids:
                counts = metrics_per_iou[iou_th]['per_class_counts'][c]
                TP = counts['TP']; FP = counts['FP']; FN = counts['FN']
                prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                # compute AP using all detections of this class across dataset
                dets_for_class = metrics_per_iou[iou_th]['detections_by_class'][c]
                gt_by_image_for_class = metrics_per_iou[iou_th]['gt_by_class'][c]
                ap = self._compute_ap_for_class(dets_for_class, gt_by_image_for_class, iou_th)

                per_class_results[c] = {
                    'TP': TP, 'FP': FP, 'FN': FN,
                    'precision': float(prec), 'recall': float(rec),
                    'AP': float(ap)
                }
                aps.append(ap)

            # mAP: mean of APs over classes (exclude classes with zero GT? here we include all classes)
            mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0
            results[iou_th] = {
                'per_class': per_class_results,
                'mAP': mAP
            }
        return results


def evaluate_model(model, dataset, oob_mask_valid, collate_fn, grid, cfg, debug = False):
    
    device_p = cfg.device[0] # prediction device
    device_e = cfg.eval.eval_device[0] # evaluation device
    
    predictions = []
    gt_all = []
    
    dataloader = DataLoader(dataset, batch_size=cfg.num_batch, shuffle=True,
                                 collate_fn=collate_fn, num_workers=cfg.num_worker)
    
    len_dl = len(dataloader)
    
    iddx = len_dl // 40 if debug else len_dl
    
    pbar_p = tqdm(enumerate(dataloader), total = len_dl, desc=f"Prediction on dataset (stop at {iddx}):")
    for batch_idx, batch in pbar_p:
    
        if debug:
            if batch_idx >= iddx:
                break
            
        with torch.no_grad():
            batch["left_img"] = batch["left_img"].to(device_p)
            batch["left_img_previous"] = batch["left_img_previous"].to(device_p)
            batch["right_img"] = batch["right_img"].to(device_p)
            batch["label"] = batch["label"].to(device_e)
                    
            gt_all.append(batch["label"])
            
            temporal_l = model.create_memory(batch["left_img_previous"])
            outputs = model(batch["left_img"], batch["right_img"], temporal_l)[0]
            outputs = outputs.detach().cpu()
            predictions.append(outputs)
            
        del batch["left_img"], batch["left_img_previous"], batch["right_img"], temporal_l
        torch.cuda.empty_cache()

    
    evaluator = Evaluate(cfg, predictions, gt_all, grid, oob_mask_valid)
    results = evaluator.evaluate()
    
    return results
    
        
    
    
    

