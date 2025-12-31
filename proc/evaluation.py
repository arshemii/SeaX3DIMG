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
        if dataset is not None:
            assert collate_fn is not None
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,                 # better for eval
                collate_fn=collate_fn,
                num_workers=self.num_workers)

    def out_preparation(self, obj_out):
        obj, classes, offset, dims, yaw = obj_out
        
        scores = torch.sigmoid(obj).to("cpu")  # F.sigmoid is deprecated
        
        probs = F.softmax(classes, dim=1).to("cpu")     # [B, C, W, H, D]
        max_probs, max_indices = torch.max(probs, dim=1)  # [B, W, H, D]

        centers = offset.to("cpu")                      # [B, 3, W, H, D]
        grid = self.cfg.grid_forward[0].permute(3, 0, 1, 2).unsqueeze(0)
        grid = grid.to(centers.device)                  # make sure same device

        scales = torch.tensor(
            self.cfg.grid_unc, dtype=centers.dtype, device=centers.device
        ).view(1, 3, 1, 1, 1)

        centers = centers * scales + grid               # world centers

        dims = dims.to("cpu")                           # [B, 3, W, H, D]
        yaw = math.pi * yaw.to("cpu")                   # [B, 1, W, H, D]

        return scores, max_probs, max_indices, centers, dims, yaw

    def create_per_sample_out(self, obj, cls_prob, cls_id, cnt, dim, yaw, sample_id):
        # TODO: another way of filtering
        # obj: [1,1,W,H,D]
        indices_sorted, scores_sorted = ev.local_maximum_3d(
            objectness=obj,
            kernel=self.kernel,
            score_thr=self.threshold,
            topk=self.k
        )

        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, f"{int(sample_id):06d}.txt")
    
        if indices_sorted.numel() == 0:
            open(out_path, "w").close()
            return
    
        b_idx, c_idx, w_idx, h_idx, d_idx = indices_sorted.T  # [N]

        # cls_id: [W, H, D], cls_prob: [1, W, H, D]
        cls_ids    = cls_id[w_idx, h_idx, d_idx]                 # [N]
        cls_scores = cls_prob[0, w_idx, h_idx, d_idx]            # [N]

        centers = cnt[b_idx, :, w_idx, h_idx, d_idx]           # [N, 3]
        dims    = dim[b_idx, :, w_idx, h_idx, d_idx]           # [N, 3]
        yaws    = yaw[b_idx, 0, w_idx, h_idx, d_idx]             # [N]
        scores  = scores_sorted * cls_scores                     # [N]

        P2 = self.cfg.camera.P_l[0]
    
        lines = []
        for i in range(centers.shape[0]):
            if self.cfg.eval.estimate_2d:
                lines.append(
                    ev.create_prediction_line_est2d(
                        centers[i], dims[i],
                        yaws[i], scores[i], cls_ids[i], P2,
                        self.cfg.eval.class_names
                    )
                )
            else:
                raise NotImplementedError("An accurate 2D box prediction is not provided yet!")
    
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    def modify_per_sample_gt(self, label_path: str):
        os.makedirs(self.save_dir_GT, exist_ok=True)

        if not isinstance(label_path, str):
            label_path = str(label_path)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"GT label file not found: {label_path}")

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
            
            if getattr(self.cfg, "short_grid_range", False) and cls_ != "DontCare":
                try:
                    x = float(fields[11])
                    y = float(fields[12])
                    z = float(fields[13])
                except (IndexError, ValueError):
                    continue

                if abs(x) > self.cfg.grid_size[0] / 2.0:
                    continue
                if y < self.cfg.H_min or y > self.cfg.H_max:
                    continue
                if z < 0 or z > self.cfg.grid_size[2]:
                    continue
            
            if cls_ in ("Car", "Van"):
                new_cls = "Car"
            elif cls_ == "Truck":
                new_cls = "Truck"
            elif cls_ == "Pedestrian":
                new_cls = "Person"
            elif cls_ == "Cyclist":
                new_cls = "Cyclist"
            else:
                new_cls = "DontCare"

            fields[0] = new_cls
            remapped_lines.append(" ".join(fields))

        with open(out_path, "w") as f:
            if remapped_lines:
                f.write("\n".join(remapped_lines))

    def generate_labels(self):
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                    desc="Modified label creation.")
        
        for batch_idx, batch in pbar:
            labels_path = batch["label_path"]  # list[str] of length B
            for label_path in labels_path:
                self.modify_per_sample_gt(label_path)
            pbar.set_postfix({'status': "OK"})

    def generate_predictions(self):
        self.model.eval()
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                    desc="Prediction on dataset.")
        
        for batch_idx, batch in pbar:
            batch["left_img"] = batch["left_img"].to(self.device)
            batch["right_img"] = batch["right_img"].to(self.device)
            samples_idx = batch['id']
            
            with torch.no_grad():
                outputs, _ = self.model(batch["left_img"], batch["right_img"])
            
            scores, max_probs, max_indices, centers, dims, yaw = self.out_preparation(outputs)
            
            for b, sample_id in enumerate(samples_idx):
                obj_b   = scores[b:b+1]      # [1,1,W,H,D]
                prob_b  = max_probs[b:b+1]   # [1,W,H,D]
                clsid_b = max_indices[b]     # [W,H,D]
                cnt_b   = centers[b:b+1]
                dim_b   = dims[b:b+1]
                yaw_b   = yaw[b:b+1]
    
                self.create_per_sample_out(
                    obj=obj_b,
                    cls_prob=prob_b,
                    cls_id=clsid_b,
                    cnt=cnt_b,
                    dim=dim_b,
                    yaw=yaw_b,
                    sample_id=sample_id
                )
                
    def please_evaluate(self):
        
        import sys
        sys.path.append("proc/kitti-object-eval-python")
        import kitti_common as kitti
        from eval import get_official_eval_result
        
        det_path = self.cfg.eval.save_dir
        gt_path  = self.cfg.eval.save_dir_gt
        split_file = self.cfg.eval.split_eval

        with open(split_file, 'r') as f:
            lines = f.readlines()
        val_image_ids = [int(line) for line in lines]
        
        dt_annos = kitti.get_label_annos(det_path)                 # all preds
        gt_annos = kitti.get_label_annos(gt_path, val_image_ids)   # GT for split
        
        for cls_id, cls_name in enumerate(self.cfg.eval.class_names):
            print(f"Evaluation on {cls_name} ...............")
            print(get_official_eval_result(gt_annos, dt_annos, cls_id))