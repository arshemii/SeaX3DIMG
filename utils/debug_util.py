#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 17:50:27 2025

@author: arash
"""

import logging
import os
from datetime import datetime

def setup_logger(name='my_logger', log_dir='logs'):
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Adjust level as needed

    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Create console handler (optional)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
