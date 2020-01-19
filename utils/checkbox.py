import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout

from utils.data_utils import *
#from utils.data_processing import *

def gen_checkbox(precheck_boxes, feature_dict):
    feature_checkbox = [widgets.Checkbox(description=d, value=v) 
                        for d,v in zip(feature_dict.values(), precheck_boxes.values())]
    col_length = 15
    num_features = len(feature_dict)
    num_of_cols = int(num_features / col_length) + 1
    hbox = []
    for j in range(num_of_cols):
        offset = col_length * j
        col_checkbox = []
        for i in range(col_length):
            if(offset + i >= num_features):
                break
            col_checkbox.append(feature_checkbox[offset + i])
        hbox.append(VBox(col_checkbox))
    return hbox

def generate_precheck_boxes(feature_pre_selected, feature_dict, dont_show=True):
    precheck_boxes = {}
    num_features = len(feature_dict)
    feature_pre_selected = feature_pre_selected
    
    for i in np.arange(1, num_features+1, 1):
        if i in feature_pre_selected:
            precheck_boxes[str(i)]= True
        else:
            precheck_boxes[str(i)]= False
    if not dont_show:
        print("Checkboxes lists: {0}".format(precheck_boxes))
    return precheck_boxes

def load_feature_groups():
    feature_groups = { 
    1: '24h_urine', 
    2: '25_oh_d',
    3: 'patient_report',
    4: 'liver_function',
    5: 'renal_function',
    6: 'measure',
    7: 'thyroid_function',
    8: 'blood_routine',
    9: 'blood_fat',
    }
    group_examinations = {
    '24h_urine': [1, 2], 
    '25_oh_d': [3, 4],
    'patient_report': [7, 23, 33, 44, 45, 46, 47, 48, 50, 51, 52],
    'liver_function': [8, 9, 10],
    'renal_function': [11, 12, 17, 18],
    'measure': [15, 25, 28, 29, 38, 39, 40, 41],
    'thyroid_function': [21, 22, 36],
    'blood_routine': [26, 30, 32, 37],
    'blood_fat': [27, 34, 35, 49],
    }
    return feature_groups, group_examinations
    

def pre_select_feature(include_feature_groups, include_feature_index, exclude_feature_index, dont_show=True):
    """
    Pre-select some feature to be "Checked" in checkbox.
    Excluded features are excluded even if they were included before.
    Variables:
        - include_feature_groups: A list of int. Representing some physical examinations like blood_routine.
        - include_feature_index: A list of int. Some features alone. 
        - exclude_feature_index: A list of int. Exclude some features.
    Returns:
        - A list of int. Selected feature indexes.
    """
    feature_groups, group_examinations = load_feature_groups()
    feature_selected = []
    for i in include_feature_groups:
        feature_selected = feature_selected + group_examinations[feature_groups[i]]
    
    feature_selected = feature_selected + [x for x in include_feature_index if x not in feature_selected]
    feature_selected = [x for x in feature_selected if x not in exclude_feature_index]
    feature_selected = sorted(feature_selected)
    if not dont_show:
        print("Pre-selected features are: {0}".format(feature_selected))
    return feature_selected

def review_checkbox(hbox, dont_show=True, log=True, to_file=''):
    checked_features = []
    for vbox in hbox:
        for checkbox in vbox.children:
            if checkbox.value is True:
                checkbox_description = checkbox.description
                feature_index_str = checkbox_description.split('_')[0]
                feature_index = int(feature_index_str)
                checked_features.append(feature_index)
    if not dont_show:
        print("Checked features:\n ", checked_features)
    if log:
        if to_file == 'nn_log':
            filename = 'NN_Log.txt'
        else:
            filename = 'SVM_Log.txt'
        with open(filename, 'a') as log_file:
            print('\n\n\n-----------------------------------------------------------------------------------------', file=log_file)
            print('Checked features:', checked_features, file=log_file)
    return checked_features