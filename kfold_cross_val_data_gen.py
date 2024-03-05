# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:46:18 2024

@author: hussain
"""

import os
import pandas as pd
import argparse
import re
from sklearn.model_selection import StratifiedKFold

def get_args_parser():
    parser = argparse.ArgumentParser('Merge data from multiple domains', add_help=False)
    parser.add_argument('--data_dir', default='../../../data/', type=str)
    parser.add_argument('--dataset_names', nargs='+', default=['AML', 'MLL', 'PBC', 'BMC', 'matek', 'raabin'], type=str)
    parser.add_argument('--output_dir', default='datasets/', type=str)
    parser.add_argument('--output_filename', default='kfold_aml_mll_pbc_matek_raabin.csv', type=str)
    parser.add_argument('--splits', default=5, type=int, help=["number of splits. i.e., 5, 10 etc."])
    return parser

class MergeData():
    def __init__(self, data_dir, dataset_names, splits):
        self.data_dir = data_dir
        self.dataset_names = dataset_names
        self.splits = splits
        self.class_mapping = self.class_mapping_label()
        self.image_paths, self.label_names, self.img_count = [], [], []
    
    def generate_metadata(self):
        for dataset in self.dataset_names:
            dataset_path = os.path.join(self.data_dir, dataset)
            for cls_name in os.listdir(dataset_path):                            
                full_data_path = os.path.join(dataset_path+'/', cls_name)
                if(os.path.isdir(full_data_path)):                    
                    print('Processing dataset: ', dataset, ', class: ', cls_name)
                    for subfolder_img in os.listdir(full_data_path):
                        check_path = os.path.join(full_data_path+'/', subfolder_img)
                        if(os.path.isdir(check_path)):
                            for subfolder in os.listdir(check_path):
                                img_path = os.path.join(check_path+'/', subfolder)
                                self.image_paths.append(self.clean_path(img_path))
                                self.label_names.append(self.class_mapping[cls_name])
                                self.img_count.append('train')
                        else:
                            img_path = check_path
                            self.image_paths.append(self.clean_path(img_path))
                            self.label_names.append(self.class_mapping[cls_name])
                            self.img_count.append('train')
                else:
                    img_path = full_data_path
                    self.image_paths.append(self.clean_path(img_path))
                    cls_name = cls_name.split('_')[0]
                    self.label_names.append(self.class_mapping[cls_name])
                    self.img_count.append('train')
        return self.image_paths, self.label_names, self.img_count
    
    def metadata_split(self):
        seed=314 #for random variable
        image_list, label_list, img_count = self.generate_metadata()
        metadata = pd.DataFrame()
        metadata['image'] = image_list
        metadata['mask'] = ['NULL' for i in range(len(self.label_names))] # if no masks
        metadata['label'] = label_list
        for splt in range(self.splits):
            metadata['set'+str(splt)] = img_count
        
        # Remove unknown classes
        metadata.label = metadata.label.replace(self.class_mapping_label())
        known_index = metadata.label != "unknown"
        metadata = metadata.loc[known_index,:].reset_index(drop = True)
        
        skf = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=seed)
        i=0
        x_train={}
        x_test={}
        y_train={}
        y_test={}
        for train_index, test_index in skf.split(metadata.index, metadata.label):
            x_train[i], x_test[i] = metadata.index[train_index], metadata.index[test_index]
            y_train[i], y_test[i] = metadata.label[train_index], metadata.label[test_index]
            i+=1
            
        for splt in range(self.splits):
            metadata.loc[x_test[splt], 'set'+str(splt)]='test'
        return metadata
    
    def clean_path(self, path):
        cleaned_path = path.replace('\\', '/')
        # cleaned_path = re.sub(r'/\.\./', '/', cleaned_path)        
        # cleaned_path = re.sub(r'^\.\./', '', cleaned_path)
        # cleaned_path = re.sub(r'/\.\.$', '', cleaned_path)
        # cleaned_path = re.sub(r'^\.\.$', '', cleaned_path)
        cleaned_path = re.sub(r'\.\./', '', cleaned_path)
        return cleaned_path

    def class_mapping_label(self):
        class_mapping = {
            #Acevedo_20 dataset
            'basophil': 'basophil',
            'eosinophil': 'eosinophil',
            'erythroblast': 'erythroblast',
            'IG': "unknown", #immature granulocytes,
            'PMY': 'promyelocyte', #immature granulocytes,
            'MY': 'myelocyte', #immature granulocytes,
            'MMY': 'metamyelocyte', #immature granulocytes,
            'lymphocyte': 'lymphocyte_typical',
            'monocyte': 'monocyte',
            'NEUTROPHIL': "unknown",
            'BNE': 'neutrophil_banded',
            'SNE': 'neutrophil_segmented',
            'platelet': "unknown",
            #Matek_19 dataset
            'BAS': 'basophil',
            'EBO': 'erythroblast',
            'EOS': 'eosinophil',
            'KSC': 'unknown',
            'LYA': 'unknown',
            'LYT': 'lymphocyte_typical',
            'MMZ': 'metamyelocyte',
            'MOB': 'monocyte', #monoblast
            'MON': 'monocyte',
            'MYB': 'myelocyte',
            'MYO': 'unknown',
            'NGB': 'neutrophil_banded',
            'NGS': 'neutrophil_segmented',
            'PMB': "unknown",
            'PMO': 'promyelocyte',
            #MLL_20 dataset
            '01-NORMO': 'erythroblast',
            '04-LGL': "unknown", #atypical
            '05-MONO': 'monocyte',
            '08-LYMPH-neo': 'unknown',
            '09-BASO': 'basophil',
            '10-EOS': 'eosinophil',
            '11-STAB': 'neutrophil_banded',
            '12-LYMPH-reaktiv': 'unknown',
            '13-MYBL': 'unknown',
            '14-LYMPH-typ': 'lymphocyte_typical',
            '15-SEG': 'neutrophil_segmented',
            '16-PLZ': "unknown",
            '17-Kernschatten': 'unknown',
            '18-PMYEL': 'promyelocyte',
            '19-MYEL': 'myelocyte',
            '20-Meta': 'metamyelocyte',
            '21-Haarzelle': "unknown",
            '22-Atyp-PMYEL': "unknown",
            #BMC_22
            '01-ABE': 'unknown',
            '02-ART' :	'unknown',
            '03-BAS' :	'basophil',
            '04-BLA' :	'unknown',
            '05-EBO' :	'erythroblast',
            '06-EOS' :	'eosinophil',
            '07-FGC' :	'unknown',
            '08-HAC' :	'unknown',
            '09-KSC' :	'unknown',
            '10-LYI' :	'unknown',
            '11-LYT' :	'lymphocyte_typical',
            '12-MMZ' :	'metamyelocyte',
            '13-MON' :	'monocyte',
            '14-MYB' :	'myelocyte',
            '15-NGB' :	'neutrophil_banded',
            '16-NGS' :	'neutrophil_segmented',
            '17-NIF' :	'unknown',
            '18-OTH' :	'unknown',
            '19-PEB' :	'unknown',
            '20-PLM' :	'unknown',
            '21-PMO' :	'promyelocyte',
            # Raabin
            'Basophil' : 'basophil',
            'Eosinophil' : 'eosinophil',
            'Lymphocyte': 'lymphocyte_typical',
            'Monocyte' : 'monocyte',
            'Neutrophil': 'neutrophil_segmented',
            # matek
            'BAS': 'basophil',
            'EBO': 'erythroblast',
            'EOS': 'eosinophil',
            'KSC': 'unknown',
            'LYA': 'unknown',
            'LYT': 'lymphocyte_typical',
            'MMZ': 'metamyelocyte',
            'MOB': 'monocyte',
            'MON': 'monocyte',
            'MYB': 'myelocyte',
            'MYO': 'unknown',
            'NGB': 'neutrophil_banded',
            'NGS': 'neutrophil_segmented',
            'PMB': "unknown",
            'PMO': 'promyelocyte'
            }
        return class_mapping
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge data from multiple domains', parents=[get_args_parser()])
    args = parser.parse_args()
    merge_data = MergeData(args.data_dir, args.dataset_names, args.splits)
    metadata = merge_data.metadata_split()
    os.makedirs(args.output_dir, exist_ok=True)
    metadata.to_csv(os.path.join(args.output_dir, args.output_filename), index=False)
    # metadata.sample(frac = 1).iloc[:len(images),:].to_csv(os.path.join(args.output_dir, args.output_filename), index=False) # len(images) = no. of samples you want
