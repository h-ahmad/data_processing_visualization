#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 00:14:11 2023

@author: hussain
"""

import os
import csv
import pandas as pd

model_list = ['convnext', 'deit', 'swin', 'volo']

for i in range(len(model_list)):
    print('processing folder ', i)
    df_csv_append = pd.DataFrame()
    folder = model_list[i]
    csv_list = [csv_name for csv_name in os.listdir(folder)]
    for j in range(len(csv_list)):
        csv_path = os.path.join(folder, csv_list[j])
        # with open(csv_path, 'r') as file:
        #     reader = csv.reader(file)
        #     for k, row in enumerate(reader):
        #         if k > 0:
        #             print(row)
        #             break 
        df = pd.read_csv(csv_path)
        df_csv_append = df_csv_append.append(df, ignore_index=True)
    df_csv_append.to_csv(os.path.join(folder, folder+'_merged_logits.csv'))