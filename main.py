#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script handle everything for training process

Notes:
    1. change the dataset module to have previous step images

"""

import argparse 
from model_cong import config_generator




def main(cfg):
    return 0



if __name__ == "__main__":
    cfg = config_generator()
    
    parser = argparse.ArgumentParser(description="Main script for Detection inference. Use flags:")
    
    parser.add_argument('--mode', 
                        choices=['train_sq', 'train_rnd', 'eval_sq', 'eval_rnd', 'inference'],
                        required=True,
                        help="The mode of execution")
    
    args = parser.parse_args()
    
    cfg.mode = args.mode
    
    main(cfg)