#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note: all the inputs to these modules are in shape:
    (1, 259, 100, 30, 70)
    where we try to keep the spatial dimension the same since they are position clues
"""
import torch.nn as nn
import torch.nn.functional as F


def convbn_3d(in_planes, out_planes, kernel_size, stride, 
              pad, dilation=1,gn=False, groups=8):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, 
                                   padding=pad, dilation=dilation, stride=stride, bias=False),
                                    nn.BatchNorm3d(out_planes) if not gn else nn.GroupNorm(groups, out_planes))

class head_3d_detection(nn.Module):
    def __init__(self, cfg, gn=True, debug = False):
        super(head_3d_detection, self).__init__()
        
        self.cfg = cfg
        self.inplanes = self.cfg.model.head.inplanes
        self.num_classes = self.cfg.model.num_class
        
        self.debug = debug
        
        self.conv1 = nn.Sequential(convbn_3d(self.inplanes, self.inplanes*2, kernel_size=3, 
                                             stride=2, pad=1, gn=gn),
                                           nn.ReLU(inplace=True))
        
        self.conv2 = convbn_3d(self.inplanes*2, self.inplanes*2,  kernel_size=3, 
                               stride=1, pad=1, gn=gn)

        self.conv3 = nn.Sequential(convbn_3d(self.inplanes*2, self.inplanes*2, kernel_size=3, 
                                             stride=2, pad=1, gn=gn),
                                               nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(self.inplanes*2, self.inplanes*2, kernel_size=3, 
                                             stride=1, pad=1, gn=gn),
                                           nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(self.inplanes*2, self.inplanes*2, kernel_size=3, 
                                                      padding=1, output_padding=1, stride=2,bias=False),
                                                       nn.BatchNorm3d(self.inplanes*2) if not gn else nn.GroupNorm(32, self.inplanes * 2))

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(self.inplanes*2, self.inplanes, kernel_size=3, 
                                                      padding=1, output_padding=1, stride=2,bias=False),
                                                       nn.BatchNorm3d(self.inplanes) if not gn else nn.GroupNorm(32, self.inplanes))


        assert len(self.cfg.loss.heads) > 1
        self.head = self.cfg.loss.heads[1:]
        
        self.head_modules = nn.ModuleDict()
        
        assert 'obj_head' in self.head
        self.head_modules['obj_head'] = nn.Conv3d(self.inplanes, 1, kernel_size=1)
        
        if 'cls_head' in self.head:
            self.head_modules['cls_head'] = nn.Conv3d(self.inplanes, self.num_classes, kernel_size=1)
        if 'cnt_head' in self.head:
            self.head_modules['cnt_head'] = nn.Sequential(nn.Conv3d(self.inplanes, 3, kernel_size=1), nn.Tanh())
        if 'dim_head' in self.head:
            self.head_modules['dim_head'] = nn.Sequential(nn.Conv3d(self.inplanes, 3, kernel_size=1), nn.Softplus())
        if 'yaw_head' in self.head:
            self.head_modules['yaw_head'] = nn.Sequential(nn.Conv3d(self.inplanes, 1, kernel_size=1), nn.Tanh())

    def forward(self, x, mode = 'eval'):
        
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        pre = F.relu(pre, inplace=True)
        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16
        post = F.relu(self.conv5(out) + pre, inplace=True)
        out = self.conv6(post)  # in:1/8 out:1/4
        
        head_outs = []
        for hd in self.head:
            # self.head order is consistent with head_module order
            head_outs.append(self.head_modules[hd](out))
        return tuple(head_outs)


 
