import matplotlib.pyplot as plt
import numpy as np
import os
import random

from sklearn import preprocessing

def load_features(path, dont_show=True):
    """
    Load features to a dictionary of (key  value) = (N  Feature_Name)
    e.g.:   key  value
            1    1_24h_urinary_microalbumin
            2    2_24h_urine_protein
    """
    path = path
    dont_show = dont_show
    feature_dict = {}
    
    files = os.listdir(path)
    to_delete = []
    for i in files:
        if not i.endswith('.txt'):
            to_delete.append(i)
    files = [x for x in files if x not in to_delete]
            
    files = sorted(files, key=lambda x: int(x.split('_')[0])) # Sort by the number before the first _(underscore mark).
    
    num_features = len(files)
    for i in range(num_features):
        #print('%d:"%s",' %(i+1, files[i][:-4]))
        feature_dict[i+1] = "%s" %(files[i][:-4])
        if not dont_show:
            print(i+1, feature_dict[i+1])
    print("Feature dict loaded.\n")
    return feature_dict

def load_using_features(feature_dict, using_feature_index, dont_show=True):
    """
    We select some features to predict one missing data feature by kNN.
    - Inputs:
        feature_dict: a dictionary of (key  value) = (N  Feature_Name)
        using_feature_index: A list of ints. Use these features to predict.
    - Output:
        X: A numpy array of shape (Num_of_patients x Num_of_using_features)
    """
    path = "dataset_new"
    using_features = []
    feature_dict = feature_dict
    using_feature_index = using_feature_index # use these features to predict.
    dont_show = dont_show
    for i in using_feature_index:
        using_features.append(feature_dict[i])
    if not dont_show:
        print("Using feature:")
        for feature in using_features:
            print(feature)
    # Load the feature data for predicting missing data of one feature.
    num_patients = 132#len(np.loadtxt("dataset_new/%s.txt" %(using_features[i])))
    num_using_features = len(using_features)
    X = np.zeros((num_using_features, num_patients))
    for i in range(num_using_features):
        feature_path = os.path.join(path, using_features[i])
        y = np.loadtxt("%s.txt" %(feature_path))
        X[i, :] = y
    X = X.T
    print("Loading (%d) features, done." %num_using_features)
    return X



def show_feature_details(feature_dict, show_missing=True, show_full=True):
    path = "dataset_new"
    feature_dict = feature_dict
    missing_data_features = {}
    full_data_features = []
    for (key, value) in feature_dict.items():
        missing_count = 0
        feature_path = os.path.join(path, value)
        y = np.loadtxt("%s.txt" %(feature_path))
        if(np.min(y) < -9998):
            for y_i in y:
                if(y_i < -9998):
                    missing_count = missing_count + 1
            missing_data_features[value] = missing_count
            continue
        else:
            full_data_features.append(value)
    
    if show_missing:
        print("Missing data features:")
        for feature in missing_data_features:
            print(feature, missing_data_features[feature])
    if show_full:
        print("\n\nFull data features:")
        for feature in full_data_features:
            print(feature)
    return


def load_asm(path="dataset_new"):
    path = os.path.join(path, '101_asm')
    asm = np.loadtxt("%s.dat" %(path))
    return asm
def load_gender(path="dataset_new"):
    path = os.path.join(path, '102_gender')
    gender = np.loadtxt("%s.dat" %(path))
    return gender
def load_asm_over_h2(path="dataset_new"):
    path = os.path.join(path, '103_asm_over_h2')
    asm_over_h2 = np.loadtxt("%s.dat" %(path))
    return asm_over_h2
def load_sarcopenia(path="dataset_new"):
    path = os.path.join(path, '104_sarcopenia')
    sarcopenia = np.loadtxt("%s.dat" %(path))
    return sarcopenia
def load_height_squared(path="dataset_new"):
    path = os.path.join(path, '105_height_squared')
    h2 = np.loadtxt("%s.dat" %(path))
    return h2
def load_index(path="dataset_new"):
    path = os.path.join(path, '106_index')
    index = np.loadtxt("%s.dat" %(path))
    return index


def observe_data(feature_dict):
    """
    Observe data in a plot.
    - Inputs: a numpy dictionary of (Num_of_patients x Num_of_features)
    """
    path = "dataset_new"
    len_path = os.path.join(path, feature_dict[1])
    size_X = len(np.loadtxt("%s.txt" %(len_path)))
    X = np.linspace(1, size_X, size_X)
    for (_, value) in feature_dict.items():
        #print(value)
        feature_path = os.path.join(path, value)
        y = np.loadtxt("%s.txt" %(feature_path))
        #print(y)
        plt.subplot(1, 1, 1)
        plt.scatter(X, y, c='k', label='data')
        plt.plot(X, y, c='g', label='data')
        plt.axis('tight')
        plt.legend()
        for i, txt in enumerate(y):
            if(y[i] > -999):
            	if random.randint(0, 9) > 8:
                	plt.annotate("%d %s" %(i+1, txt), (X[i], y[i]))
        ymin = np.min(y)
        plt.ylim(max(-2, ymin), np.max(y))
        if ymin < -999:
            plt.title(value+" !Missing data")
        else:
            plt.title(value)
        plt.tight_layout()
        plt.show()

def shuffle_feature(feature_data, shuffle_index, num_train, num_val, num_test):
    """
    - Shuffle dataset and split them to three sets: Training, Validation and Test set.
    - Parameters:
        feature_data: 1-dim numpy array.
        shuffle_index: 1-dim numpy array generated by np.random.permutation(X.shape[0]).
        num_train, num_val, num_test: Set sizes.
    - Returns:
        Shuffled splited datasets.
    """
    data = feature_data[shuffle_index]
    data_train = data[:num_train]
    data_val = data[num_train:num_train+num_val]
    data_test = data[num_train+num_val:]

    return data_train, data_val, data_test


def set_scaler():
    """
    - Set scaler for data. Uncomment the scaler to use.
    """
    #scaler = preprocessing.RobustScaler()
    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    #scaler = preprocessing.MaxAbsScaler()
    #scaler = preprocessing.RobustScaler(quantile_range=(10, 90)) 
    #scaler = preprocessing.PowerTransformer(method='yeo-johnson')
    #scaler = preprocessing.PowerTransformer(method='box-cox')# positive
    #scaler = preprocessing.QuantileTransformer(output_distribution='normal')
    #scaler = preprocessing.QuantileTransformer(output_distribution='uniform')
    return scaler
def rename_files():
    '''
    Sort .txt files in lexicographical order, give them "index_" prefix.
    Those file should not have been renamed. 
    For example when you want to add new file, you should mannually index it.
    '''
    path = 'dataset_test'
    k = os.listdir(path)
    print(k)
    for i in k:
        if not i.endswith('.txt'):
            k.remove(i)
            pass
    k = sorted(k)
    for i in range(len(k)):
        os.rename('dataset_test/'+k[i] ,("dataset_test/%d_%s" %(i+1, k[i])))