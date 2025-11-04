#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  10 11:51:22 2025

@author: arash
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt


files = sorted(glob.glob('/home/arash/SeaX3DIMG/logging_dir/epoch_*_log.csv'))


df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


df = df.sort_values(by='epoch').reset_index(drop=True)

selected_columns = ['epoch', '0', '990', '1830']  # 👈 replace with your own column names
df_selected = df[selected_columns].copy()

values_list = df_selected[['0', '990', '1830']].values.flatten().tolist()


x = range(len(values_list))

plt.scatter(x, values_list, s=10)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Flattened Values Scatter')
plt.grid(True)
plt.show()

