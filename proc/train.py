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
        
        self.heads_for_loss = [o for o in self.cfg.loss.heads if o not in self.cfg.loss.freezed_output]
        
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
    
    
    def train_epoch(self, epoch):
        
        running_loss = 0.0
        avg_loss = 0.0
        
        if len(self.cfg.loss.heads) > 1:
            loss_track_obj = 0.0
            avg_loss_track_obj = 0.0
            loss_track_cls = 0.0
            avg_loss_track_cls = 0.0
            loss_track_cnt = 0.0
            avg_loss_track_cnt = 0.0
            loss_track_dim = 0.0
            avg_loss_track_dim = 0.0
            loss_track_yaw = 0.0
            avg_loss_track_yaw = 0.0
        
        print(f"Active: {self.cfg.loss.heads}, freezed: {self.cfg.loss.freezed_output}")
        print("From stage 5, the losses are: Dim: 0.00084, Center: 0.10376, Class: 0.00652, Obj: 0.00028")
        print("---------------------------------------------------------------")
        
        w_prev = [self.cfg.loss.w_total_previous[0],
                  self.cfg.loss.w_total_previous[1],
                  self.cfg.loss.w_total_previous[2],
                  self.cfg.loss.w_total_previous[3]]    # objecness, classification, center, dim
        w_yaw = self.cfg.loss.w_yaw

        
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Stage 6, Epoch {epoch}")
        
        for batch_idx, batch in pbar:
                        
            batch["left_img"] = batch["left_img"].to(self.device)
            batch["right_img"] = batch["right_img"].to(self.device)
            
            with autocast(device_type='cuda'):
                if self.cfg.loss.heads == ['disp']:
                    disp = self.model(batch["left_img"], batch["right_img"])
                else:
                    outputs, disp = self.model(batch["left_img"], batch["right_img"])
                    if 'disp' in self.cfg.loss.freezed_output:
                        del disp

                del batch["left_img"], batch["right_img"]
                assert "label" in batch.keys()
                
                loss = {'total': 0.0}
                if 'disp' in self.heads_for_loss:
                    batch["disparity"] = batch["disparity"].to(self.device)
                    loss['disp'], _ = self.loss_fn.disparity_loss(disp, batch["disparity"])
                    del disp, batch["disparity"]
                    
                if 'obj_head' in self.heads_for_loss:
                    batch['assignment'] = batch['assignment'].to(self.device)
                    loss['obj_head'], _ = self.loss_fn.object_conf_loss(outputs[0], batch['assignment'])
                    
                if 'cls_head' in self.heads_for_loss:
                    batch["label"] = batch["label"].to(self.device)
                    loss['cls_head'], _ = self.loss_fn.classification_loss(outputs[1], outputs[0],
                                                                          batch['assignment'], batch["label"])
                
                if 'cnt_head' in self.heads_for_loss:
                    loss['cnt_head'], _ = self.loss_fn.center_loss(outputs[2], batch['assignment'], batch["label"])

                if 'dim_head' in self.heads_for_loss:
                    loss['dim_head'], _ = self.loss_fn.dimension_loss(outputs[3], batch['assignment'], batch["label"])
                    
                if 'yaw_head' in self.heads_for_loss:
                    loss['yaw_head'], _ = self.loss_fn.yaw_loss(outputs[4], batch['assignment'], batch["label"])
                    
                
                del batch["label"], batch['assignment']
                                    
                if self.cfg.loss.heads == ['disp']:
                    loss['total'] = loss['disp']
                else:
                    del outputs
                    for idx, loss_t in enumerate(self.heads_for_loss[:-1]):
                        loss['total'] += (w_prev[idx] * loss[loss_t])

                    loss['total'] = loss['total'] + \
                                    (w_yaw * loss[self.heads_for_loss[-1]])
              
         
            scaler.scale(loss['total']).backward()

            if (batch_idx + 1) % self.cfg.dev.grad_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            running_loss += loss['total'].item()
            avg_loss = running_loss / (batch_idx + 1)

            if len(self.cfg.loss.heads) == 1:
                per_batch_loss = loss['total'].item()
                pbar.set_postfix({'loss': f"{avg_loss:.5f}",
                                  'Disp Loss PB:': f"{per_batch_loss:.5f}",
                                  'batch': f"{batch_idx+1}/{len(self.dataloader)}"})
            else:
                loss_track_obj += loss['obj_head'].item()
                loss_track_cnt += loss['cnt_head'].item()
                loss_track_cls += loss['cls_head'].item()
                loss_track_dim += loss['dim_head'].item()
                loss_track_yaw += loss['yaw_head'].item()
                avg_loss_track_cnt = loss_track_cnt / (batch_idx + 1)
                avg_loss_track_obj = loss_track_obj / (batch_idx + 1)
                avg_loss_track_cls = loss_track_cls / (batch_idx + 1)
                avg_loss_track_dim = loss_track_dim / (batch_idx + 1)
                avg_loss_track_yaw = loss_track_yaw / (batch_idx + 1)
                pbar.set_postfix({'loss': f"{avg_loss:.4f}",
                		  'Yaw loss': f"{avg_loss_track_yaw:.5f}",
                                  'Dim loss': f"{avg_loss_track_dim:.5f}",
                                  'Center loss': f"{avg_loss_track_cnt:.5f}",
                                  'Class loss': f"{avg_loss_track_cls:.5f}",
                                  'Obj loss': f"{avg_loss_track_obj:.5f}",
                                  'batch': f"{batch_idx+1}/{len(self.dataloader)}"})
                
            
        if (batch_idx + 1) % self.cfg.dev.grad_steps != 0:
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
        self.scheduler.step()
        # TODO: inside loop or each epoch?
        torch.cuda.empty_cache()
        avg_epoch_loss = running_loss / len(self.dataloader)
        return avg_epoch_loss
    
    def train(self):
        
        print("\n" + "="*60)
        print("  ------------------------Training Started ------------------------  ")
        print("="*60)
        print(f" Grid resolution :          {self.cfg.grid_resolution}")
        print(f" Input image size:          {self.cfg.model.in_size}")
        print(f" Backbone       :           {self.cfg.model.back.name}")
        print("-"*60 + "\n")
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
            
            checkpoint_path_prev = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch-3}.pth')
            if os.path.exists(checkpoint_path_prev):
                print(f"Removing checkpoints of epoch: {epoch - 3} ...")
                os.remove(checkpoint_path_prev)
        
