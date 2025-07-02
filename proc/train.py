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
    def __init__(self, cfg, model, dataset, grid, collate_fn, metric_module,
                 device, optimizer, loss_fn, resume_checkpoint=None):
        
        self.cfg = cfg
        self.model = model.to(device)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.metric_module = metric_module
        self.num_epochs = self.cfg.num_epochs
        self.device = device
        self.eval_in_training = self.cfg.eval_in_training
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grid = grid
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
        # TODO
        raise NotImplementedError("Print statistics")
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        avg_loss = 0.0
        if self.eval_in_training:
            if epoch % 4 == 0:
                eval_pair = []
                
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
                # TODO: does it applicable to use .to method for device for a list???
                targets = [t.to(self.device) for t in targets]
            else:
                raise NotImplementedError("Target must be list for integration")
            
            self.optimizer.zero_grad()
            
            #create temporal memory for both left and right image from t - dt
            temporal_l, temporal_r = self.model.creat_memory(inputs[1], inputs[3])
            outputs = self.model(inputs[0], inputs[2], temporal_l, temporal_r)[0]
            if self.eval_in_training:
                if epoch % 4 == 0:
                    eval_pair.append([outputs, targets])
            
            loss = self.loss_fn(outputs, targets, self.grid)
            loss['total'].backward()
            self.optimizer.step()
            
            running_loss += loss['total'].item()
            avg_loss = running_loss / (batch_idx + 1)
            
            pbar.set_postfix({'loss': f"{avg_loss:.2f}", 'batch': f"{batch_idx+1}/{len(self.dataloader)}"})
        
        
        if self.eval_in_training:
            if epoch % 4 == 0:
                metric_values = self.metric_module.eval_from_prediction(eval_pair, self.grid)
        else:
            metric_values = None
        
        avg_epoch_loss = running_loss / len(self.dataloader)
        return avg_epoch_loss, metric_values
    
    def evaluate(self):
        self.model.eval()
        eval_pair = []
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, targets = batch
                inputs = [t.to(self.device) for t in inputs]
                # target is a list
                if isinstance(targets, dict):
                    raise NotImplementedError("Target must be list for integration")
                elif isinstance(targets, tuple):
                    raise NotImplementedError("Target must be list for integration")
                elif isinstance(targets, list):
                    # TODO: does it applicable to use .to method for device for a list???
                    targets = [t.to(self.device) for t in targets]
                else:
                    raise NotImplementedError("Target must be list for integration")
                
                temporal_l, temporal_r = self.model.creat_memory(inputs[1], inputs[3])
                outputs = self.model(inputs[0], inputs[2], temporal_l, temporal_r)[0]
                # TODO: why only in evaluation to use .cpu for output?
                eval_pair.append([outputs.cpu(), targets])
        
        # Assuming your metric_module takes lists of outputs and targets
        metrics = self.metric_module.eval_from_prediction(eval_pair, self.grid)
        return metrics
    
    def save_epoch_log(self, epoch, loss, metrics, train_time):
        log = {
            'epoch': epoch,
            'loss': loss,
            'train_time_sec': train_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        if self.eval_in_training:
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
            
            metrics = self.evaluate()
            
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
