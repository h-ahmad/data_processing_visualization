# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:45:46 2024

@author: hussain
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import argparse
import ast

def get_args_parser():
    parser = argparse.ArgumentParser('Data indexing', add_help=False)
    parser.add_argument('--output_csv_path', default='datasets/', type=str)
    parser.add_argument('--output_csv_name', default='metadata_kf10_raabin.csv', type=str)
    parser.add_argument('--data_path', default='../../../data/raabindata/Original/', type=str, help=['root image path to extract image and mask names'])
    parser.add_argument('--img_path', default='data/raabindata/Original/', type=str)
    parser.add_argument('--mask_path', default='data/raabindata/Ground Truth/', type=str)
    parser.add_argument('--splits', default=10, type=int, help=["number of splits. i.e., 5, 10 etc."])
    parser.add_argument('--label_names', type=str, default="{'Basophil' : 'basophil', 'Eosinophil' : 'eosinophil', 'Lymphocyte' : 'lymphocyte', 'Monocyte' : 'monocyte', 'Neutrophil' : 'neutrophil'}", help='label mapping')
    return parser

def metadata_generator(data_path, img_path, mask_path, label_names, splits):    
    img_path_list, label, msk_path = [], [], []
    img_count = []
    class_folders = sorted(os.listdir(data_path))
    for cl in class_folders:
        for img in tqdm(os.listdir(os.path.join(data_path, cl))):
            img_full_path = os.path.join(img_path, os.path.join(cl+'/', img))
            mask_full_path = os.path.join(mask_path, os.path.join(cl+'/', img))       
            img_path_list.append(img_full_path)   
            label.append(label_names[cl])
            msk_path.append(mask_full_path)
            img_count.append("train")
    metadata = pd.DataFrame()
    metadata['image'] = img_path_list
    metadata['mask'] = msk_path
    metadata['label'] = label
    for splt in range(splits):
        metadata['set'+str(splt)] = img_count

    return metadata

def metadata_split(metadata, splits):
    seed=314 #for random variable
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    i=0
    x_train={}
    x_test={}
    y_train={}
    y_test={}
    for train_index, test_index in skf.split(metadata.index, metadata.label):
        x_train[i], x_test[i] = metadata.index[train_index], metadata.index[test_index]
        y_train[i], y_test[i] = metadata.label[train_index], metadata.label[test_index]
        i+=1
        
    for splt in range(splits):
        metadata.loc[x_test[splt], 'set'+str(splt)]='test'
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data indexing', parents=[get_args_parser()])
    args = parser.parse_args()
    args.label_names = ast.literal_eval(args.label_names)
    metadata = metadata_generator(args.data_path, args.img_path, args.mask_path, args.label_names, args.splits)   
    metadata = metadata_split(metadata, args.splits) 
    os.makedirs(args.output_csv_path, exist_ok = True)
    metadata.to_csv(os.path.join(args.output_csv_path, args.output_csv_name), index=False)
