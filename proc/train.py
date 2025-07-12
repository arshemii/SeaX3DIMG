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
    def __init__(self, cfg, model, dataset, grid, collate_fn, metric_module,
                 optimizer, scheduler, loss_fn, resume_checkpoint=None):
        
        self.cfg = cfg
        self.device = self.cfg.device[0]
        self.model = model.to(self.device)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.metric_module = metric_module
        self.num_epochs = self.cfg.num_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grid = grid
        self.scheduler = scheduler
        assert self.grid.shape[-1] == 0

        self.batch_size = self.cfg.num_batch
        self.num_workers = self.cfg.num_worker
        self.checkpoint_dir = self.cfg.model.sx3d.checkpoint
        self.log_dir = self.cfg.log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.start_epoch = 0
        if resume_checkpoint:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
            self.start_epoch = load_checkpoint(resume_checkpoint, self.model, self.optimizer)
        
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                     collate_fn=self.collate_fn, num_workers=self.num_workers)
    
    def _print_train_stats(epoch, avg_loss, metrics, train_time):
        # TODO: use method print_metrics from metric_module object
        print(f"Training epoch {epoch} with loss {avg_loss:.2f} in {train_time():.2f}")
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        avg_loss = 0.0
        if self.metric_module:
            if epoch % 4 == 0:
                eval_pair = []
                
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
            
            for key in ["left_img", "left_img_previous", "right_img", "calib"]:
                if key == "calib":
                    batch["calib"] = batch["calib"].to(self.device)
                elif key == "label":
                    for sample in batch["label"]:
                        for label in sample:
                            for k in label.keys():
                                if k in ["category", "bbox3d", "bbox2d"]:
                                    label[k] = label[k].to(self.device)
                else:
                    batch[key] = batch[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            #create temporal memory for both left and right image from t - dt
            with autocast():
                temporal_l = self.model.create_memory(batch["left_img_previous"])
                outputs = self.model(batch["left_img"], batch["right_img"], temporal_l, batch["calib"])[0]
                if self.eval_in_training:
                    if epoch % 4 == 0:
                        eval_pair.append([outputs, batch["label"]])
            
                loss = self.loss_fn(outputs, batch["label"], self.grid)
                
            scaler.scale(loss['total']).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            running_loss += loss['total'].item()
            avg_loss = running_loss / (batch_idx + 1)
            
            pbar.set_postfix({'loss': f"{avg_loss:.2f}", 'batch': f"{batch_idx+1}/{len(self.dataloader)}"})
            
        self.scheduler.step()
        
        if self.metric_module:
            if epoch % 4 == 0:
                metric_values = self.metric_module.eval_from_prediction(eval_pair, self.grid)
        else:
            metric_values = None
        
        avg_epoch_loss = running_loss / len(self.dataloader)
        return avg_epoch_loss, metric_values
    
    def evaluate(self):
        raise NotImplementedError("Target must be list for integration")
    
    def save_epoch_log(self, epoch, loss, metrics, train_time):
        log = {
            'epoch': epoch,
            'loss': loss,
            'train_time_sec': train_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        if self.metric_module:
            if epoch % 4 == 0:
                log['metrics'] = metrics
                
        log_filename = os.path.join(self.log_dir, f'epoch_{epoch}_log.json')
        # TODO: why opening here?
        with open(log_filename, 'w') as f:
            json.dump(log, f, indent=4)
    
    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            start_time = time.time()
            
            avg_loss, _ = self.train_epoch(epoch)
            
            if self.metric_module:
                metrics = self.evaluate()
            else:
                metrics = None
            
            train_time = time.time() - start_time
            
            self._print_train_stats(epoch, avg_loss, metrics, train_time)
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict()
            }, checkpoint_path)
            
            # Save epoch logs
            self.save_epoch_log(epoch, avg_loss, metrics, train_time)
