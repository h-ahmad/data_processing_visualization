import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def metadata_generator(data_path):
    label_names = {'Basophil' : 'basophil', 'Eosinophil' : 'eosinophil', 'Lymphocyte' : 'lymphocyte', 'Monocyte' : 'monocyte', 'Neutrophil' : 'neutrophil'}
    img_path, label, msk_path, set0, set1, set2, set3, set4 = [], [], [], [], [], [], [], []
    imgs_path = os.path.join(data_path, 'Original_imgs')
    masks_path = os.path.join(data_path, 'GT_masks')
    class_folders = sorted(os.listdir(imgs_path))
    for cl in class_folders:
        for img in tqdm(os.listdir(os.path.join(imgs_path, cl))):
            img_full_path = os.path.join(imgs_path, os.path.join(cl, img))
            mask_full_path = os.path.join(masks_path, os.path.join(cl, img))            
            img_path.append(img_full_path)
            label.append(label_names[cl])
            msk_path.append(mask_full_path)
            set0.append("train")
            set1.append("train")
            set2.append("train")
            set3.append("train")
            set4.append("train")
    metadata = pd.DataFrame()
    metadata['image'] = img_path
    metadata['mask'] = msk_path
    metadata['label'] = label
    metadata['set0'] = set0
    metadata['set1'] = set1
    metadata['set2'] = set2
    metadata['set3'] = set3
    metadata['set4'] = set4 

    return metadata

def metadata_5split(metadata):
    seed=314 #for random variable
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    i=0
    x_train={}
    x_test={}
    y_train={}
    y_test={}
    for train_index, test_index in skf.split(metadata.index, metadata.label):
        x_train[i], x_test[i] = metadata.index[train_index], metadata.index[test_index]
        y_train[i], y_test[i] = metadata.label[train_index], metadata.label[test_index]
        i+=1
        
    metadata.loc[x_test[0], 'set0']='test'
    metadata.loc[x_test[1], 'set1']='test'
    metadata.loc[x_test[2], 'set2']='test'
    metadata.loc[x_test[3], 'set3']='test'
    metadata.loc[x_test[4], 'set4']='test'
    return metadata

if __name__ == "__main__":
    data_path = '../../../datasets/raabindata'
    metadata = metadata_generator(data_path)    
    #splitting in 5 test/train-sets
    metadata = metadata_5split(metadata)   
    metadata.to_csv("dataset/metadata_kf5.csv", index=False)
    
