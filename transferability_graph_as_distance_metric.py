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
from sklearn import manifold
from statistics import mean
import random

seed = np.random.RandomState(seed=3)   # seed:  RandomState(MT19937)
mds = manifold.MDS(n_components=7, max_iter=30, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)  # mds:  <class 'sklearn.manifold._mds.MDS'>

feature_path = 'PACS_features.pkl'
with open(feature_path, 'rb') as file:
    data_store = pickle.load(file)
    print('data_store: ', data_store.keys())
    features = data_store['features']
    print('features: ', features.keys())
    data = features['feats']
    labels = features['labels']
    print('labels: ', labels.keys())

    colors = ['red', 'blue', 'darksalmon', 'green', 'black', 'yellow', 'magenta']
    legen_name = []
    global_array = []
    fig, ax = plt.subplots()
    for index, env in enumerate(labels):
        dom_color = colors[index]
        dom_array = []
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

            # domain-class pair (d, c) --> M = D * C
            cls_array = np.array(cls_data)
            cls_array -= cls_array.mean()
            similarity = euclidean_distances(cls_array)
            similarity = np.average(similarity)

            myList = np.array([similarity, cls_array.shape[0]])
            dom_array.append(myList)
        global_array.append(np.array(dom_array))

        # pos = mds.fit(similarity).embedding_
        volume = global_array[index][:, 1]

        scatter = ax.scatter(global_array[index][:, 0], volume, color=dom_color, s=volume, lw=0, alpha=0.6, label='Domain '+str(index+1))
        
        for i in range(global_array[index].shape[0]):
            ax.annotate(str(i), (global_array[index][:, 0][i], volume[i]), ha='center', va='center')
    legend = ax.legend(loc="upper left", markerscale=0.4)
