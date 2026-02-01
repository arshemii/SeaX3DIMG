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
        # reduce left features (same reduction pattern you used)
        self.conv_reduce = nn.Conv2d(cin, 16, kernel_size=4, stride=4)
        self.conv_em = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 1),
            nn.LeakyReLU(0.2),
        )
        # hypothesis extractor (kept similar)
        self.conv_hyp = nn.Sequential(
            nn.Conv2d(cref + 1, 13, 1),
            nn.LeakyReLU(0.2),
        )
        # cost refinement: small 3D conv block to smooth cost volume
        # input 1 channel (L1-norm cost), output 1 channel refined cost
        self.cost_refine = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, l, r, ref=None):
        # l: (B, cin, H, W)  r: (B, cin, H, W)
        lt = F.conv2d(l, self.conv_reduce.weight, self.conv_reduce.bias, stride=(4, 4))
        lt = self.conv_em(lt)

        # same_padding_conv behavior replaced by explicit padding conv2d to match earlier code:
        # replicate same behavior: downsample r in height but not width (stride=(4,1) + pad)
        # we emulate your same_padding_conv by explicit padding calculation
        # pad for stride (4,1) with kernel 4x4
        pad_h = (4 - 1) // 2
        pad_w = (4 - 1) // 2
        rt = F.pad(r, (pad_w, pad_w, pad_h, pad_h))
        rt = F.conv2d(rt, self.conv_reduce.weight, self.conv_reduce.bias, stride=(4, 1))
        rt = self.conv_em(rt)

        # cost volume: (B, D, H4, W4)
        #lt = F.normalize(lt, p=2, dim=1)
        #rt = F.normalize(rt, p=2, dim=1)
        cv = make_cost_volume_v2(lt, rt, self.max_disp)  # (B, C, D, H4, W4) with C = channels difference
        # collapse channel difference to cost per disparity by L1-norm across feature channels
        cv = torch.norm(cv, p=1, dim=1, keepdim=True)  # (B, 1, D, H4, W4)

        # refine with 3D conv
        cv_refined = self.cost_refine(cv)  # (B, 1, D, H4, W4)
        cv_ref = cv_refined.squeeze(1)     # (B, D, H4, W4)

        # soft-argmin for differentiable disparity (subpixel)
        # convert costs to negative log-likelihood style: lower cost -> higher prob
        prob = torch.softmax(-cv_ref, dim=1)  # (B, D, H4, W4)

        disp_levels = torch.arange(self.max_disp, dtype=cv_ref.dtype, device=cv_ref.device).view(1, -1, 1, 1)
        disp = torch.sum(prob * disp_levels, dim=1, keepdim=True)  # (B,1,H4,W4) subpixel disparity index

        # confidence: use negative entropy (higher -> more confident)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1, keepdim=True)  # (B,1,H4,W4)
        # normalize confidence into [0,1] by a small squashing (optional)
        conf = torch.sigmoid((1.0 - entropy))  # (B,1,H4,W4) -- higher for less entropy

        if ref is None:
            ref = lt
        # preserve previous behavior: p = concat(cv_min/ref) but now use cv_min as min cost across disparity
        cv_min, _ = torch.min(cv_ref, dim=1, keepdim=True)
        p = torch.cat((cv_min, ref), dim=1)  # (B, 1 + 16, H4, W4)
        p = self.conv_hyp(p)  # (B, 13, H4, W4)
        # pack disp as first 3 channels in a small style to match old interface: [d, 0, 0] + p
        d_pad = torch.cat([disp, torch.zeros_like(disp), torch.zeros_like(disp)], dim=1)  # (B,3,H4,W4)
        p_out = torch.cat((d_pad, p), dim=1)  # (B, 3+13, H4, W4)
        # return (p_out, cv_ref, disp, conf)
        return p_out, cv_ref, disp, conf


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

    def forward(self, hyps, l, r, conf=None):
        # hyps: list of hypothesis feature maps (each B, C, H4, W4)
        # l/r: original features (B, cin, H, W)
        # conf: (B,1,H4,W4) optional confidence to weight warping cost
        cost = [warp_and_aggregate(h, l, r) for h in hyps]  # returns (B, C_cost, H4, W4)
        cost = torch.cat(cost, dim=1)  # (B, Ccat, H4, W4)
        x = self.conv_neighbors(cost)
        hyps_cat = torch.cat(hyps, dim=1)
        x = torch.cat((hyps_cat, x), dim=1)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)
        dh = x[:, :16 * self.h_size]
        w = x[:, 16 * self.h_size :]
        return hyps_cat + dh, w
        
class Level(nn.Module):
    def __init__(self, cin, max_disp, h_size, cref=16):
        super(Level, self).__init__()
        self.h_size = h_size
        self.init = LevelInit(cin, max_disp, cref)
        self.prop = LevalProp(self.h_size)

    def forward(self, l, r, h=None, ref=None):
        # returns (h_out, cv_ref, disp, conf) similar to before but differentiable
        p_out, cv_ref, disp, conf = self.init(l, r, ref)
        if self.h_size == 1:
            h_out, w = self.prop([p_out], l, r, conf=conf)
            return h_out, cv_ref, disp, w, conf  # keep similar interface
        else:
            # if multi-level pipeline desired (not used often here), we can pass h into prop
            h_out, w = self.prop([p_out, h], l, r, conf=conf)
            # combine hypothesis channels with soft selection (weighted sum) rather than strict where
            # keep simple: return h_out, cv_ref, disp, w
            return h_out, cv_ref, disp, w, conf