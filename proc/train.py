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

class Trainer:
    def __init__(self, cfg, model, dataset, collate_fn,
                 optimizer, scheduler, loss_fn, resume_checkpoint=None):
        
        self.cfg = cfg
        self.device = self.cfg.device[0]
        self.model = model.to(self.device)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_epochs = self.cfg.dev.num_epoch
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.batch_size = self.cfg.num_batch
        self.num_workers = self.cfg.num_worker
        
        self.checkpoint_dir = self.cfg.model.sx3d.checkpoint_bev
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
        # TODO: use method print_metrics from metric_module object
        print(f"Training epoch {epoch} with loss {avg_loss:.2f} in {train_time:.2f}")
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        avg_loss = 0.0
                
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
                        
            # batch["calib"] = batch["calib"].to(self.device)
            batch["left_img"] = batch["left_img"].to(self.device)
            batch["left_img_previous"] = batch["left_img_previous"].to(self.device)
            batch["right_img"] = batch["right_img"].to(self.device)
            batch["label"] = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            #create temporal memory for both left and right image from t - dt
            with autocast(device_type='cuda'):
                temporal_l = self.model.create_memory(batch["left_img_previous"])
                outputs = self.model(batch["left_img"], batch["right_img"], temporal_l)[0]
                
                # TODO: reduce memory oh
                del temporal_l
                
                assert "label" in batch.keys()
                loss = self.loss_fn(outputs, batch["label"])
                
                # TODO: reduce overhead
                del outputs
                
            # TODO: must be removed
            if torch.isnan(loss['total']) or loss['total'].item() == 0.0:
                # Clear unused memory to reduce fragmentation (ChatGPT)
                torch.cuda.empty_cache()
                print(f"==> Skipping optimizer step — Loss is {loss['total'].item()}")
                continue
            
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
