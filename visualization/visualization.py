#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  10 11:51:22 2025

@author: arash
"""

import numpy as np
import cv2

def bev_box(predictions, resolution):
  """
  Predictions: numpy array with shape N, 8
  resolution: from configuration

  *** The output is an image where height is the depth in 2d grid (top to down reducing depth)
      and width is width of the 2d grid, vehicle is bottom-centered
  """
  H, W = resolution[2], resolution[0]
  
  


def bev_hm(predictions, resolution):
  """
  Predictions: numpy array with shape res_w, res_l, 8
  resolution: from configuration

  *** The output is a heatmep of objectness,
      where height is the depth in 2d grid (top to down reducing depth)
      and width is width of the 2d grid, vehicle is bottom-centered
  """
  H, W = resolution[2], resolution[0]

  objectess = predictions[:, :, 0]
  max_obj = np.max(objectess)
  min_obj = np.min(objectess)

