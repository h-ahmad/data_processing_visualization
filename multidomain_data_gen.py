# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:46:18 2024

@author: hussain
"""

import os
import pandas as pd
import argparse
import re

def get_args_parser():
    parser = argparse.ArgumentParser('Merge data from multiple domains', add_help=False)
    parser.add_argument('--data_dir', default='../../../data/', type=str)
    parser.add_argument('--dataset_names', nargs='+', default=['AML', 'MLL', 'PBC', 'BMC'], type=str)
    # parser.add_argument('--dataset_names', nargs='+', default=['BMC'], type=str)
    parser.add_argument('--output_dir', default='datasets/', type=str)
    parser.add_argument('--output_filename', default='kfold_aml_mll_pbc.csv', type=str)
    return parser

class MergeData():
    def __init__(self, data_dir, dataset_names):
        self.data_dir = data_dir
        self.dataset_names = dataset_names
        self.class_mapping = self.class_mapping_label()
        self.image_paths, self.label_names = [], []
    
    def generate_metadata(self):
        for dataset in self.dataset_names:
            dataset_path = os.path.join(self.data_dir, dataset)
            for cls_name in os.listdir(dataset_path):                            
                full_data_path = os.path.join(dataset_path+'/', cls_name)
                print('Processing dataset: ', dataset, ', class: ', cls_name)
                for subfolder_img in os.listdir(full_data_path):
                    check_path = os.path.join(full_data_path+'/', subfolder_img)
                    if(os.path.isdir(check_path)):
                        for subfolder in os.listdir(check_path):
                            img_path = os.path.join(check_path+'/', subfolder)
                            self.image_paths.append(self.clean_path(img_path))
                            self.label_names.append(self.class_mapping[cls_name])
                    else:
                        img_path = check_path
                        self.image_paths.append(self.clean_path(img_path))
                        self.label_names.append(self.class_mapping[cls_name])
        return self.image_paths, self.label_names

    def class_mapping_label(self):
        class_mapping = {
            # matek19/ AML
            'BAS': 'basophil',
            'EBO': 'erythroblast',
            'EOS': 'eosinophil',
            'KSC': 'smudge_cell',
            'LYA': 'lymphocyte_atypical',
            'LYT': 'lymphocyte_typical',
            'MMZ': 'metamyelocyte',
            'MOB': 'monoblast',
            'MON': 'monocyte',
            'MYB': 'myelocyte',
            'MYO': 'myeloblast',
            'NGB': 'neutrophil_band',
            'NGS': 'neutrophil_segmented',
            'PMB': "promyelocyte_bilobled",
            'PMO': 'promyelocyte',
            # mll_marr
            '01-NORMO': 'normo', # ?
            '04-LGL': "atypical",
            '05-MONO': 'monocyte',
            '08-LYMPH-neo': 'lymphocyte_atypical',
            '09-BASO': 'basophil',
            '10-EOS': 'eosinophil',
            '11-STAB': 'neutrophil_banded',
            '12-LYMPH-reaktiv': 'lymphocyte_atypical', 
            '13-MYBL': 'myeloblast',
            '14-LYMPH-typ': 'lymphocyte_typical',
            '15-SEG': 'neutrophil_segmented',
            '16-PLZ': "plz", # ?
            '17-Kernschatten': 'smudge_cell',
            '18-PMYEL': 'promyelocyte',
            '19-MYEL': 'myelocyte',
            '20-Meta': 'metamyelocyte',
            '21-Haarzelle': "haarzelle", # ?
            '22-Atyp-PMYEL': "atyp-pmyel", # ?
            # acevedo / PBC
            'basophil': 'basophil',
            'BNE': 'neutrophil_banded',
            'eosinophil': 'eosinophil',
            'erythroblast': 'erythroblast',
            'IG': "immature_granulocytes",
            'lymphocyte': 'lymphocyte',
            'MMY': 'metamyelocytes',
            'monocyte': 'monocyte',    
            'MY': 'myelocyte',
            'NEUTROPHIL': "neutrophil",
            'platelet': "platelet", # thrombocytes
            'PMY': 'promyelocyte', 
            'promyelocyte': 'promyelocyte',
            'SNE': 'neutrophil_segmented',
            # BMC
            '01-ABE': 'abnormal_eosinophil',
            '02-ART': 'artefact',
            '03-BAS': 'basophil',
            '04-BLA': 'blast',
            '05-EBO': 'erythroblast',
            '06-EOS': 'eosinophil',
            '07-FGC': 'faggott_cell',
            '08-HAC': 'hairy_cell',
            '09-KSC': 'smudge_cell',
            '10-LYI': 'immature_lymphocyte',
            '11-LYT': 'lymphocyte',
            '12-MMZ': 'metamyelocyte',
            '13-MON': 'monocyte',
            '14-MYB': 'myelocyte',
            '15-NGB': 'band_neutrophil',
            '16-NGS': 'segmented_neutrophil',
            '17-NIF': 'not_identifiable',
            '18-OTH': 'other_cell',
            '19-PEB': 'proerythroblast',
            '20-PLM': 'plasma_cell',
            '21-PMO': 'promyelocyte'
            }
        return class_mapping
    def clean_path(self, path):
        cleaned_path = path.replace('\\', '/')
        # cleaned_path = re.sub(r'/\.\./', '/', cleaned_path)        
        # cleaned_path = re.sub(r'^\.\./', '', cleaned_path)
        # cleaned_path = re.sub(r'/\.\.$', '', cleaned_path)
        # cleaned_path = re.sub(r'^\.\.$', '', cleaned_path)
        cleaned_path = re.sub(r'\.\./', '', cleaned_path)
        return cleaned_path
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge data from multiple domains', parents=[get_args_parser()])
    args = parser.parse_args()
    
    merge_data = MergeData(args.data_dir, args.dataset_names)
    images, labels = merge_data.generate_metadata()
    metadata = pd.DataFrame()
    metadata['image'] = images
    metadata['mask'] = ['NULL' for i in range(len(images))]
    metadata['label'] = labels
    os.makedirs(args.output_dir, exist_ok=True)
    metadata.sample(frac = 1).iloc[:len(images),:].to_csv(os.path.join(args.output_dir, args.output_filename), index=False) # len(images) = no. of samples you want
