#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:47:11 2024

@author: hussain
"""

import matplotlib.pyplot as plt

x_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
y_values1 = [37.21, 39, 41.73, 38.64, 40.07, 41.34, 40.47, 40.38, 41.66, 41.82]
y_values2 = [1, 2, 1, 2, 1, 3, 5, 3, 2, 5]

plt.plot(x_values, y_values1, label='MCTFormer+', color='blue', marker='o')
plt.plot(x_values, y_values2, label='Mask R-CNN', color='red', marker='s')

plt.xlabel('Number of samples')
plt.ylabel('iou (%)')
plt.title('')
plt.legend()

plt.show()