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

def save_checkpoint(state, filename):
    torch.save(state, filename)
    
def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint.get('epoch', 0)

class Trainer:
    def __init__(self, cfg, model, dataset, collate_fn, metric_module,
                 print_fn, device, optimizer, loss_fn, resume_checkpoint=None):
        
        self.cfg = cfg
        self.model = model.to(device)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.metric_module = metric_module
        self.print_fn = print_fn
        self.device = device
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
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
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        metric_values = []
        
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
            inputs, targets = batch
            
            # input has four images (only in training):
                # left pair: img in t-dt and img in t
                # right pair: img in t-dt and img in t
            inputs = [t.to(self.device) for t in inputs]
            
            # target is a list
            if isinstance(targets, dict):
                raise NotImplementedError("Target must be list for integration")
            elif isinstance(targets, tuple):
                raise NotImplementedError("Target must be list for integration")
            elif isinstance(targets, list):
                targets = [t.to(self.device) for t in targets]
            else:
                raise NotImplementedError("Target must be list for integration")
            
            self.optimizer.zero_grad()
            
            #create temporal memory for both left and right image from t - dt
            temporal_l, temporal_r = self.model.creat_memory(inputs[1], inputs[3])
            outputs = self.model(inputs[0], inputs[2], temporal_l, temporal_r)
            
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'batch': f"{batch_idx+1}/{len(self.dataloader)}"})
            
            # Optionally, accumulate intermediate metrics on the fly
            # e.g. metric_values.append(self.metric_module(outputs, targets))
        
        avg_epoch_loss = running_loss / len(self.dataloader)
        return avg_epoch_loss, metric_values
    
    def evaluate(self):
        self.model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                elif isinstance(targets, (list, tuple)):
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets)
        
        # Assuming your metric_module takes lists of outputs and targets
        metrics = self.metric_module(all_outputs, all_targets)
        return metrics
    
    def save_epoch_log(self, epoch, loss, metrics, train_time):
        log = {
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'train_time_sec': train_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        log_filename = os.path.join(self.log_dir, f'epoch_{epoch}_log.json')
        with open(log_filename, 'w') as f:
            json.dump(log, f, indent=4)
    
    def train(self, num_epochs):
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            start_time = time.time()
            
            avg_loss, _ = self.train_epoch(epoch)
            
            metrics = self.evaluate()
            
            train_time = time.time() - start_time
            
            self.print_fn(epoch, avg_loss, metrics, train_time)
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict()
            }, checkpoint_path)
            
            # Save epoch logs
            self.save_epoch_log(epoch, avg_loss, metrics, train_time)