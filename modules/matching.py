#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:16:03 2025

@author: arash
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools

# def hyp_up(hyp, scale=1, tile_scale=1):
#     if scale != 1:
#         d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=False)
#         p = F.interpolate(hyp[:, 1:], scale_factor=scale)
#         hyp = torch.cat((d, p), dim=1)
#     if tile_scale != 1:
#         d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], tile_scale, tile_expand=True)
#         p = F.interpolate(hyp[:, 1:], scale_factor=tile_scale)
#         hyp = torch.cat((d, p), dim=1)
#     return hyp

def make_cost_volume_v2(left, right, max_disp):
    d_range = torch.arange(max_disp, device=left.device)
    d_range = d_range.view(1, 1, -1, 1, 1)

    x_index = torch.arange(left.size(3), device=left.device)
    x_index = x_index.view(1, 1, 1, 1, -1)

    x_index = torch.clip(4 * x_index + 1 - d_range, 0, right.size(3) - 1).repeat(
        right.size(0), right.size(1), 1, right.size(2), 1
    )
    right = torch.gather(
        right.unsqueeze(2).repeat(1, 1, max_disp, 1, 1), dim=-1, index=x_index
    )
    
    return left.unsqueeze(2) - right

def same_padding_conv(x, w, b, s):
    out_h = math.ceil(x.size(2) / s[0])
    out_w = math.ceil(x.size(3) / s[1])

    pad_h = max((out_h - 1) * s[0] + w.size(2) - x.size(2), 0)
    pad_w = max((out_w - 1) * s[1] + w.size(3) - x.size(3), 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    x = F.conv2d(x, w, b, stride=s)
    return x

@functools.lru_cache()
@torch.no_grad()
def make_warp_coef(scale, device):
    center = (scale - 1) / 2
    index = torch.arange(scale, device=device) - center
    coef_y, coef_x = torch.meshgrid(index, index)
    coef_x = coef_x.reshape(1, -1, 1, 1)
    coef_y = coef_y.reshape(1, -1, 1, 1)
    return coef_x, coef_y

def disp_up(d, dx, dy, scale, tile_expand):
    n, _, h, w = d.size()
    # a 2d grid coordinates cenetered at zero with scale*scale points
    # used for sampling
    coef_x, coef_y = make_warp_coef(scale, d.device) # (1, 16, 1, 1)
    
    # find the more precise indexes for minimum disparity
    if tile_expand:
        d = d + coef_x * dx + coef_y * dy # 1, 16, h/4, w/4
    else:
        d = d * scale + coef_x * dx * 4 + coef_y * dy * 4

    d = d.reshape(n, 1, scale, scale, h, w) # 1, 1, 4, 4, h/4, w/4
    d = d.permute(0, 1, 4, 2, 5, 3)  # 1, 1, h/4, 4, w/4, 4
    d = d.reshape(n, 1, h * scale, w * scale) # 1, 1, 128, 240 --> upsampled min disparity
    return d

def warp_and_aggregate(hyp, left, right):
    scale = left.size(3) // hyp.size(3)
    assert scale == 4
    
    # inputs of disp_up: d, zero, zero where d is index on min disparity in channels
    # outputs min disparity index in upsampled version
    d_expand = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=True)
    d_range = torch.arange(right.size(3), device=right.device)
    d_range = d_range.view(1, 1, 1, -1) - d_expand # 1, 1, h, w
    d_range = d_range.repeat(1, right.size(1), 1, 1) # 1, cin, h, w

    cost = [torch.sum(torch.abs(left), dim=1, keepdim=True)] # cost sum, 1, 1, h, w
    for offset in [1, 0, -1]:
        index_float = d_range + offset # 1, cin, h, w
        index_long = torch.floor(index_float).long()
        index_left = torch.clip(index_long, min=0, max=right.size(3) - 1)
        index_right = torch.clip(index_long + 1, min=0, max=right.size(3) - 1)
        index_weight = index_float - index_left

        right_warp_left = torch.gather(right, dim=-1, index=index_left.long())
        right_warp_right = torch.gather(right, dim=-1, index=index_right.long())
        right_warp = right_warp_left + index_weight * (
            right_warp_right - right_warp_left
        ) 
        cost.append(torch.sum(torch.abs(left - right_warp), dim=1, keepdim=True)) # index cost sum, 1, 1, h, w (displacement)
    cost = torch.cat(cost, dim=1) # 1, 2, h, w

    n, c, h, w = cost.size()
    cost = cost.reshape(n, c, h // scale, scale, w // scale, scale)
    cost = cost.permute(0, 3, 5, 1, 2, 4)
    cost = cost.reshape(n, scale * scale * c, h // scale, w // scale)
    return cost

class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.conv(input)
        x = x + input
        x = self.relu(x)
        return x

class LevelInit(nn.Module):
    def __init__(self, cin, max_disp, cref=16):
        super(LevelInit, self).__init__()
        self.max_disp = max_disp
        self.conv_reduce = nn.Conv2d(cin, 16, 4)
        self.conv_em = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv_hyp = nn.Sequential(
            nn.Conv2d(cref + 1, 13, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, l, r, ref=None):
        lt = F.conv2d(l, self.conv_reduce.weight, self.conv_reduce.bias, stride=(4, 4))
        lt = self.conv_em(lt)

        rt = same_padding_conv(r, self.conv_reduce.weight, self.conv_reduce.bias, s=(4, 1))
        rt = self.conv_em(rt)
        """
        left feature map is downsampled to 1/4 in 1/4 but right to 1/4 in 1 (width the same)
        then, corresponding points from w width level of right feature map are sampled and w/4 points are gathered
        Output cv is a (1, 16, disparity_level, h/4, w/4) showing that for each channel, each point has n(disparity_level) feature by subtracting
        left feature by disparity-shifted rigth feature map
        """

        cv = make_cost_volume_v2(lt, rt, self.max_disp)
        cv = torch.norm(cv, p=1, dim=1) # L1 norm along disparity --> (1, disparity_level, h/4, w/4)
        cv_min, d = torch.min(cv, dim=1, keepdim=True)
        d = d.float() # at each pixel which disparity level (index) is minimum

        if ref is None:
            ref = lt
        p = torch.cat((cv_min, ref), dim=1) # must be the main fature + min disparity with c_in + 1 channels (1, 16 + 1, h/4, w/4)
        p = self.conv_hyp(p) # (1, 13, h/4, w/4)
        p = torch.cat((d, torch.zeros_like(d), torch.zeros_like(d), p), dim=1)
        # P is some features extracted
        # cv is the disparity
        # p and cv has the same shape (1, 16, h/4, w/4)
        return p, cv

class LevalProp(nn.Module):
    def __init__(self, h_size=2):
        super(LevalProp, self).__init__()
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d(64 * h_size, 16 * h_size, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32 * h_size, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.res_block = nn.Sequential(ResBlock(32), ResBlock(32))
        self.convn = nn.Conv2d(32, 17 * h_size, 3, 1, 1)
        self.h_size = h_size

    def forward(self, hyps, l, r):
        # hyps is some features extracted shape (1, 16, h/4, w/4)
        # l and r are tensors of shape (1, cin, h, w)
        # hyps has one element in this setup
        cost = [warp_and_aggregate(h, l, r) for h in hyps]
        cost = torch.cat(cost, dim=1)
        x = self.conv_neighbors(cost)
        hyps = torch.cat(hyps, dim=1)
        x = torch.cat((hyps, x), dim=1)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)

        dh = x[:, : 16 * self.h_size]
        w = x[:, 16 * self.h_size :]
        return hyps + dh, w


class Level(nn.Module):
    def __init__(self, cin, max_disp, h_size, cref=16):
        super(Level, self).__init__()
        self.h_size = h_size
        self.init = LevelInit(cin, max_disp, cref)
        self.prop = LevalProp(self.h_size)

    def forward(self, l, r, h=None, ref=None):
        hi, cv = self.init(l, r, ref)
        # hi is some features extracted
        # cv is the disparity
        if self.h_size == 1:
            h, w = self.prop([hi], l, r)
            return h, cv, hi[:, :1], w
        else:
            h, w = self.prop([hi, h], l, r)
            h0 = h[:, :16]
            h1 = h[:, 16:]
            w0 = w[:, :1]
            w1 = w[:, 1:]
            h = torch.where(w0 > w1, h0, h1)
            return h, cv, hi[:, :1], [[w0, h0[:, :1]], [w1, h1[:, :1]]]\
        
        
# class Refine(nn.Module):
#     def __init__(self, cin, cres, dilations):
#         super().__init__()
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(cin + 16, cres, 1),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(cres, cres, 3, 1, 1),
#             nn.LeakyReLU(0.2),
#         )
#         self.res_block = []
#         for d in dilations:
#             self.res_block += [ResBlock(cres, d)]
#         self.res_block = nn.Sequential(*self.res_block)
#         self.convn = nn.Conv2d(cres, 16, 3, 1, 1)

#     def forward(self, hpy, left):
#         x = torch.cat((left, hpy), dim=1)
#         x = self.conv1x1(x)
#         x = self.conv1(x)
#         x = self.res_block(x)
#         x = self.convn(x)
#         return hpy + x