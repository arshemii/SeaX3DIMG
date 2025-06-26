#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:14:47 2025

@author: arash
"""
import torch.nn as nn


class criterion_3dbox(nn.Module):
    def __init__(self):
        super(criterion_3dbox, self).__init__()
        
        
        
        
    def forward(self, output, target):
        