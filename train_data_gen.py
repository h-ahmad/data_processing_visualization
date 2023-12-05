
import os
import pandas as pd

import collections.abc
# from pptx import Presentation

path = '../../datasets/raabindata/Original_imgs/'
data_path = 'datasets/raabindata/Original_imgs/'
label_map = {'Basophil': 'basophil', 'Eosinophil': 'eosinophil', 'Lymphocyte': 'lymphocyte_typical',
             'Monocyte': 'monocyte', 'Neutrophil': 'unknown'}
img_path, label, data_name = [], [], []
for classes in os.listdir(path):
    for img in os.listdir(os.path.join(path, classes)):
        img_full_path = os.path.join(os.path.join(data_path, classes), img)
        img_path.append(img_full_path)
        data_name.append('Raabin')
        label.append(label_map[classes])
metadata = pd.DataFrame()
metadata['image'] = img_path
metadata['label'] = label
metadata['dataset'] = data_name

known_index = metadata.label != "unknown"
metadata = metadata.loc[known_index,:].reset_index(drop = True)
metadata.to_csv('datasets/'+"metadata_train.csv", index=False)


