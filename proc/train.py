#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 19:41:22 2025

@author: arash
"""

import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from torch.amp import autocast, GradScaler
scaler = GradScaler()

def save_checkpoint(state, filename):
    torch.save(state, filename)
    
def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint.get('epoch', 0)

def interpolate_weights(w1, w2, alpha):
    """Linearly interpolate between two lists of weights."""
    return [(1 - alpha) * a + alpha * b for a, b in zip(w1, w2)]

class Trainer:
    def __init__(self, cfg, model, dataset, collate_fn,
                 optimizer, scheduler, loss_fn, resume_checkpoint=None):
        
        self.cfg = cfg
        self.model = model
        self.device = self.cfg.device[0]
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_epochs = self.cfg.dev.num_epoch
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.batch_size = self.cfg.num_batch
        self.num_workers = self.cfg.num_worker
        
        self.missed_dict = {k: 0 for k in ['cnt_disp', 'cnt_obj', 'cnt_cls', 'cnt_cntr', 'cnt_dim', 'cnt_yaw']}
        self.stage_epochs = self.cfg.loss.stage_epochs  # boundaries for transitions (example)
        
        self.checkpoint_dir = self.cfg.model.sx3d.checkpoint_3d
        self.log_dir = self.cfg.log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.start_epoch = 0
        if resume_checkpoint:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
            if self.cfg.dev.continue_training:
                self.start_epoch = load_checkpoint(resume_checkpoint, self.model, self.optimizer)
        
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                     collate_fn=self.collate_fn, num_workers=self.num_workers)
    
    def _print_train_stats(self, epoch, avg_loss, train_time):
        print(f"Training epoch {epoch} with loss {avg_loss:.2f} in {train_time:.2f}")
    
    def update_loss_weights(self, epoch):
        """
        Smoothly update self.loss_weights based on epoch using linear interpolation
        between consecutive weight schedules.
        """
        w = self.cfg.loss.w_schedule  # list of lists
        
    
        if epoch <= self.stage_epochs[0]:
            self.loss_weights = w[0]
        elif epoch < self.stage_epochs[1]:
            alpha = (epoch - self.stage_epochs[0]) / (self.stage_epochs[1] - self.stage_epochs[0])
            self.loss_weights = interpolate_weights(w[0], w[1], alpha)
        elif epoch < self.stage_epochs[2]:
            alpha = (epoch - self.stage_epochs[1]) / (self.stage_epochs[2] - self.stage_epochs[1])
            self.loss_weights = interpolate_weights(w[1], w[2], alpha)
        elif epoch < self.stage_epochs[3]:
            alpha = (epoch - self.stage_epochs[2]) / (self.stage_epochs[3] - self.stage_epochs[2])
            self.loss_weights = interpolate_weights(w[2], w[3], alpha)
        elif epoch < self.stage_epochs[4]:
            alpha = (epoch - self.stage_epochs[3]) / (self.stage_epochs[4] - self.stage_epochs[3])
            self.loss_weights = interpolate_weights(w[3], w[4], alpha)
        else:
            self.loss_weights = w[4]
    
    def train_epoch(self, epoch):
        running_loss = 0.0
        avg_loss = 0.0
        
        self.update_loss_weights(epoch)
        
        print("---------------------------------------------------------------")
        print(f"Epoch {epoch} using loss weights: {self.loss_weights}")
        
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
                        
            batch["left_img"] = batch["left_img"].to(self.device)
            batch["right_img"] = batch["right_img"].to(self.device)
            if self.cfg.model.sx3d.memory:
                batch["left_img_previous"] = batch["left_img_previous"].to(self.device)
                batch["right_img_previous"] = batch["right_img_previous"].to(self.device)
            
            self.optimizer.zero_grad()
            
            #create temporal memory for both left and right image from t - dt
            with autocast(device_type='cuda'):
                # model forward
                if self.cfg.model.sx3d.memory:
                    temporal = self.model(batch["left_img_previous"], batch["right_img_previous"],
                                         None, create_memory = True)
                    del batch["left_img_previous"], batch["right_img_previous"]
                    if self.cfg.loss.aux_loss:
                        outputs, disp, _ = self.model(batch["left_img"], batch["right_img"],
                                                      temporal, create_memory = False)
                    else:
                        outputs, _ = self.model(batch["left_img"], batch["right_img"],
                                                temporal, create_memory = False)
                        disp = None
                        del temporal
                        
                else:
                    if self.cfg.loss.aux_loss:
                        outputs, disp = self.model(batch["left_img"], batch["right_img"],
                                                   None, create_memory = False)
                    else:
                        outputs = self.model(batch["left_img"], batch["right_img"],
                                             None, create_memory = False)
                        disp = None
                
                # model predictions
                obj = outputs[0]
                dim = outputs[1]
                centerx = outputs[2]
                cls_logits = outputs[3]
                yaw = outputs[4]
                
                del batch["left_img"], batch["right_img"], outputs
                assert "label" in batch.keys()
                
                batch["label"] = batch["label"].to(self.device)
                batch['assignment'] = batch['assignment'].to(self.device)
                if self.cfg.loss.aux_loss:
                    batch["disparity"] = batch["disparity"].to(self.device)
                
                # Loss calculation
                loss = {}
                if self.cfg.loss.aux_loss:
                    loss['disparity_loss'], cnt_disp = self.loss_fn.disparity_loss(disp, batch["disparity"])
                    self.missed_dict['cnt_disp'] += cnt_disp
                    del disp, batch["disparity"]
                    
                loss['obj_conf'], cnt_obj = self.loss_fn.object_conf_loss(obj, batch['assignment'])
                self.missed_dict['cnt_obj'] += cnt_obj
                loss['cls_loss'], cnt_cls = self.loss_fn.classification_loss(cls_logits, obj, batch['assignment'], batch["label"])
                self.missed_dict['cnt_cls'] += cnt_cls
                del cls_logits, obj
                
                loss['center_loss'], cnt_center = self.loss_fn.center_loss(centerx, batch['assignment'], batch["label"])
                self.missed_dict['cnt_cntr'] += cnt_center 
                del centerx
                
                loss['dim_loss'], cnt_dim = self.loss_fn.dimension_loss(dim, batch['assignment'], batch["label"])
                self.missed_dict['cnt_dim'] += cnt_dim
                del dim
                
                # yaw angle loss
                loss['yaw_angle_loss'], cnt_yaw = self.loss_fn.yaw_loss(yaw, batch['assignment'], batch["label"])
                self.missed_dict['cnt_yaw'] += cnt_yaw
                del yaw
                
                loss['total'] = self.loss_weights[0] * loss['obj_conf'] + \
                    self.loss_weights[1] * loss['cls_loss'] + \
                        self.loss_weights[2] * loss['center_loss'] + \
                            self.loss_weights[3] * loss['dim_loss'] + \
                                self.loss_weights[4] * loss['yaw_angle_loss']
                                
                if self.cfg.loss.aux_loss:
                    loss['total'] += self.loss_weights[5] * loss['disparity_loss']
                    
                
                if self.cfg.loss.debug:
                    print(f"loss weighs are: {self.loss_weights}")
                    for key in loss.keys():
                        print(f"The {key} value is: {loss[key]}")
                    if batch_idx % 20 == 0:
                        for k in self.missed_dict.keys():
                            print(f"Instables in {k} are: {self.missed_dict[k]}")

                del batch["label"], batch['assignment']
                
                
            
            scaler.scale(loss['total']).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # Clear unused memory to reduce fragmentation (ChatGPT)
            torch.cuda.empty_cache()
            
            running_loss += loss['total'].item()
            avg_loss = running_loss / (batch_idx + 1)
            
            pbar.set_postfix({'loss': f"{avg_loss:.3f}", 'batch': f"{batch_idx+1}/{len(self.dataloader)}, Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB, Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB"})
            
        self.scheduler.step()
        
        avg_epoch_loss = running_loss / len(self.dataloader)
        return avg_epoch_loss

    
    def save_epoch_log(self, epoch, loss, train_time):
        log = {
            'total epoch': self.num_epochs,
            'start epoch': self.start_epoch,
            'current epoch': epoch,
            'loss': loss,
            'learning rate start': self.cfg.dev.lr,
            'Weight decay': self.cfg.dev.weight_decay,
            'T max': self.cfg.dev.t_max,
            'Eta min': self.cfg.dev.eta_min,
            'train_time_sec': train_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
                
        log_filename = os.path.join(self.log_dir, f'epoch_{epoch}_log.json')
        with open(log_filename, 'w') as f:
            json.dump(log, f, indent=4)
    
    def train(self):
        
        print("\n" + "="*60)
        print("  ------------------------Training Started ------------------------  ")
        print("="*60)
        print(f" Grid resolution :          {self.cfg.grid_resolution}")
        print(f" Input image size:          {self.cfg.model.in_size}")
        print(f" Backbone       :           {self.cfg.model.back.name}")
        print("="*60 + "\n")
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            start_time = time.time()
            
            avg_loss = self.train_epoch(epoch)
            
            train_time = time.time() - start_time
            
            self._print_train_stats(epoch, avg_loss, train_time)
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            
            print(f"Saving checkpoints of epoch: {epoch} ...")
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict()
            }, checkpoint_path)
            
            checkpoint_path_prev = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch-2}.pth')
            if os.path.exists(checkpoint_path_prev):
                print(f"Removing checkpoints of epoch: {epoch - 2} ...")
                os.remove(checkpoint_path_prev)
            
            # Save epoch logs
            self.save_epoch_log(epoch, avg_loss, train_time)
