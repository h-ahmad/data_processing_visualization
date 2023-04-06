#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:03:08 2023

@author: hussain
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch

feature_path =  'PACS_features.pkl'
with open(feature_path, 'rb') as file:
        data_store = pickle.load(file)
        print(data_store.keys()) 
        features = data_store['features']
        print(features.keys())
        data = features['feats']
        labels = features['labels']
        
        data1 = data['env0']    # list
        labels1 = labels['env0']    # list
        data1 = torch.stack(data1)  # tensor torch.Size([2048])
        labels1 = torch.tensor(labels1)   # tensor torch.Size([1523])
        
        data2 = data['env1']
        data2 = torch.stack(data2)
        
        data3 = data['env2']
        data3 = torch.stack(data3)
        
        data1 = np.array(data1.cpu())
        data2 = np.array(data2.cpu())
        data3 = np.array(data3.cpu())
        
        fig = plt.figure(1)
        ax = plt.axes([0.0, 0.0, 1.0, 1.0])
        s = 10   # size of the bubbles
        plt.scatter(data1[:, 0], data1[:, 1], color="turquoise", s=s, lw=0, label="Domain 1")
        plt.scatter(data2[:, 0], data2[:, 1], color="darkorange", s=s, lw=0, label="Domain 2")
        plt.scatter(data3[:, 0], data3[:, 1], color="navy", s=s, lw=0, label="Domain 3")
        plt.legend(scatterpoints=1, loc="best", shadow=False)
        plt.show()
