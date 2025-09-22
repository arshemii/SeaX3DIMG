#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 11:48:47 2025

@author: arash
"""
from shapely.geometry import Polygon
import numpy as np


def get_corners(cx, cz, w, l, yaw):
    """
    Compute the (x,z) coordinates of the 4 corners of the rotated box.
    Returns np.array of shape (4,2).
    
    ChatGPT generated function!
    """
    # half-dimensions
    w2, l2 = w / 2.0, l / 2.0

    # corners in box local frame (before rotation)
    # order: [front-left, front-right, back-right, back-left]
    corners = np.array([
        [ w2,  l2],
        [-w2,  l2],
        [-w2, -l2],
        [ w2, -l2]
    ])

    # rotation matrix around yaw (z-axis rotation in BEV plane)
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # rotate + translate
    rotated = corners @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cz

    return rotated

def bev_iou(box1, box2):
    """
    Compute the IoU of two oriented 2D bounding boxes in BEV (x-z plane).

    Args:
        box1: list [cx, cz, w, l, yaw]
              cx, cz = box center coordinates
              w, l   = width (x-axis extent), length (z-axis extent)
              yaw    = rotation angle in radians (counter-clockwise, from x-axis)
        box2: same format as box1

    Returns:
        iou: float, intersection over union (0..1)
    
    ChatGPT generated function!
    """

    # convert both boxes to polygons
    poly1 = Polygon(get_corners(*box1))
    poly2 = Polygon(get_corners(*box2))

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    # intersection & union
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter

    if union <= 0:
        return 0.0

    return inter / union


# TODO: add to configuration all new params from here

def drop_lq_preds(prediction, objectness_score):
