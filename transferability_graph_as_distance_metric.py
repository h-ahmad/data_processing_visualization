#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:10:14 2023

@author: hussain
"""

import pickle
import torch
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances

feature_path =  'PACS_features.pkl'
with open(feature_path, 'rb') as file:
    data_store = pickle.load(file)
    print('data_store: ', data_store.keys()) 
    features = data_store['features']
    print('features: ', features.keys())
    data = features['feats']
    labels = features['labels']
    print('labels: ', labels.keys())
    
    global_index = 0
    global_data = []
    for index, env in enumerate(labels):
        dom_cls_data = {}
        dom_cls_label = {}
        cls_wise_dist = []
        env_labels = np.array(torch.tensor(labels[env]).cpu())
        env_data = np.array(torch.stack(data[env]).cpu())
        # get unique class and their counts.
        env_classes, env_counts = np.unique(env_labels, return_counts=True)
        for i in env_classes:
            cls_data = []
            cls_label = []
            for iteration, j in enumerate(env_labels):
                # Class wise sorting.
                if i == j:
                    cls_data.append(env_data[iteration])
                    cls_label.append(j)        
                    
                    dom_cls_label[global_index] = j
                    dom_cls_data[global_index] = env_data[iteration]
                    global_index += 1
                    
            cls_array = np.array(cls_data)
            cls_array -= cls_array.mean()
            cls_dist = euclidean_distances(cls_array)
            cls_wise_dist.append(cls_dist)
        global_data.append(cls_wise_dist)
    print('global_data: ', type(global_data[0][0]))
    
    
    # classes, counts = np.unique(labels, return_counts=True)
    # print('classes: ', classes, ' counts: ', counts)
    