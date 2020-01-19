import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils.data_utils import *
from sklearn import preprocessing

class SarcopeniaDataset(Dataset):
    def __init__(self, X, asm, asm_h2, sarcopenia, height_squared, patient_id, gender, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.asm = asm
        self.asm_h2 = asm_h2
        self.sarcopenia = sarcopenia
        self.height_squared = height_squared
        self.patient_id = patient_id
        self.gender = gender
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_i = self.X[idx, :]
        asm_i = self.asm[idx]
        asm_h2_i = self.asm_h2[idx]
        sarcopenia_i = self.sarcopenia[idx]
        height_squared_i = self.height_squared[idx]
        patient_id_i = self.patient_id[idx]
        gender_i = self.gender[idx]

        sample = {'X': X_i, 'asm': asm_i, 'asm_h2': asm_h2_i, 'sarcopenia': sarcopenia_i, 
                  'height_squared': height_squared_i, 'patient_id': patient_id_i, 'gender': gender_i
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample

        

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, asm, asm_h2, sarcopenia = sample['X'], sample['asm'], sample['asm_h2'], sample['sarcopenia']
        height_squared, patient_id, gender = sample['height_squared'], sample['patient_id'], sample['gender']

        return {'X': torch.from_numpy(X),
                'asm': torch.from_numpy(np.array(asm)),
                'asm_h2': torch.from_numpy(np.array(asm_h2)),
                'sarcopenia': torch.from_numpy(np.array(sarcopenia)),
                'height_squared': torch.from_numpy(np.array(height_squared)),
                'patient_id': torch.from_numpy(np.array(patient_id)),
                'gender': torch.from_numpy(np.array(gender)),
               }



def normalize_data(X, feature_dict, using_features, dont_show=True):
    """
    Inputs:
     -  X: numpy array of shape(num of patients x num of features)
            eg: 132 x 16.
     -  feature_index: list of ints, corresponding to the txt files.
            eg: [0, 3, 41].
    Outputs:
     -  A normalized X, of same shape.
    Function:
     -  Firstly, Normalize X by numpy.preprocessing.
     -  Secondly, If -9999 is in a feature, we normalize this feature again, but without the -9999 ones.
        Thirdly, we paste the normalized feature back to X.
            e.g: - X = [1, 2, -9999]
                 - Firstly, normalized to [0.01, 0.02, -1.5]. 
                 - Secondly, [1, 2] is normalized to [0.5, 0.6].
                 - Thirdly, paste back, we get [0.5, 0.6, -1.5].
        This could keep the shape of X, makes indexing easier and doesn't harm the performance of kNN.
     """
    X = X
    feature_dict = feature_dict
    features = using_features
    num_patients, num_features = X.shape
    full_data_patients = np.arange(num_patients)
    miss_data_patients = []
    miss_data_features = []
    
    for i in range(num_patients):
        for j in range(num_features):
            if X[i][j] < -9998:
                miss_data_patients.append(i)
                if j not in miss_data_features:
                    miss_data_features.append(j)
                break
    full_data_patients = np.delete(full_data_patients, miss_data_patients)
    if not dont_show:
        print("\nMissing data features:")
        for j in miss_data_features:
            print(feature_dict[features[j]])
        print("\nFull data patients:")
        print(full_data_patients)
    #X_normalized = preprocessing.normalize(X, norm='l2', axis=0)
    scaler = preprocessing.QuantileTransformer(output_distribution='uniform')
    X_normalized = scaler.fit_transform(X)
    if miss_data_features:
        X_miss_data_feature = X[full_data_patients, miss_data_features].reshape(-1, 1) # Miss data feature of patients with full data.
        X_miss_data_feature_normalized = preprocessing.normalize(X_miss_data_feature, norm='l2', axis=0).reshape(-1)  
        X_normalized[full_data_patients, miss_data_features] = X_miss_data_feature_normalized
    return X_normalized

def split_train_predict_set(X_normalized, y_to_makeup, dont_show=True):
    """
    We remove the predicting patients out of training patients set.
    - Inputs:
        X_normalized: Numpy array of shape (Num_patients x Using_features) 
        y_to_makeup: 1-dim numpy array. We want to makeup the missing data of this feature.
    - Outputs:
        X_train, X_predict: numpy array of (Num_train/predict_patients x Using_features)
        y_predict: 1-dim numpy array of length (Num_predict_patient).
        train_patients, predict_patients: A list of ints. We index train/predict patients by this list.
    """
    X_normalized = X_normalized
    y_to_makeup = y_to_makeup
    dont_show = dont_show
    
    num_patients = X_normalized.shape[0]
    train_patients = np.arange(num_patients)
    predict_patients = []
    
    for i in range(num_patients):
        if y_to_makeup[i] < -9998:
            predict_patients.append(i)
    train_patients = np.delete(train_patients, predict_patients)

    X_train = X_normalized[train_patients]
    y_train = y_to_makeup[train_patients]
    X_predict = X_normalized[predict_patients]
    
    if not dont_show:
        print("X_train: {0}, X_predict: {1}, y_train: {2}\n".format(X_train.shape, X_predict.shape, y_train.shape))
    return X_train, y_train, X_predict, train_patients, predict_patients


