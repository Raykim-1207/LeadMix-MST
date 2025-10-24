import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

#################################################################

## Function List

## smooth_labels: label smoothing
## onehot: label onehot-encoding
## _collect_files: collect file paths
## _load_data: read signal and label data
## load_single_dataset: load entire dataset

#################################################################

def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

def onehot(y_data, _shape):
    enc = OneHotEncoder()
    y_data_onehot = []
    
    enc.fit(np.array([0, 1]).reshape(-1, 1))

    for xx in range(y_data.shape[0]):
        annotat_onehot = enc.transform(np.reshape(y_data[xx], (-1, 1))).toarray()
        annotat_onehot = smooth_labels(annotat_onehot, factor=0.2)
        y_data_onehot.append(annotat_onehot)

    return np.array(y_data_onehot)

def _collect_files(_read_sig_path, desc_prefix):
    files = []
    
    file_list = []
    for sub in tqdm(sorted(os.listdir(_read_sig_path)), desc=f"{desc_prefix} - Reading patient folders"):
        path = os.path.join(_read_sig_path, sub)
        if os.path.exists(path) and os.path.isdir(path):
            file_list.extend([(sub, f) for f in sorted(os.listdir(path)) if f.endswith('.csv')])
    files = np.array(file_list)
        
    return files

def _load_data(_read_path, _read_sig_path, files, desc):
    """
    Loads X (signal) and y (label) data based on the file list.
    """
    X_data = None
    y_data = None
    ref_len = -1
    ref_label_len = -1

    for item in tqdm(files, desc=desc):
        try:
            sub, file_name = item
            signal_path = os.path.join(_read_sig_path, sub, file_name)
            label_folder_path = os.path.join(_read_path, sub)

            temp = pd.read_csv(signal_path)
            x_data_part = temp['0'].to_numpy()
            
            base_name = file_name.split('.csv')[0]
            
            if not os.path.exists(label_folder_path):
                 print(f"Warning: Label folder not found, skipping {file_name}: {label_folder_path}")
                 continue
                 
            matching_files = [f for f in os.listdir(label_folder_path) if f.split('[')[0] == base_name and not f.endswith('.csv')]
            if not matching_files:
                print(f"Warning: No matching label file found for {base_name} in {label_folder_path}")
                continue
                
            matching_file = matching_files[0]
            y_values = matching_file.split('[')[1].split(']')[0].split()
            y_data_part = np.array(y_values, dtype=int)

            if X_data is None:
                X_data = [x_data_part]
                y_data = [y_data_part]
                ref_len = x_data_part.shape[0]
                ref_label_len = y_data_part.shape[0]
            else:
                if x_data_part.shape[0] != ref_len or y_data_part.shape[0] != ref_label_len:
                    print(f"Warning: Skipping {file_name} due to shape mismatch.")
                    continue
                X_data.append(x_data_part)
                y_data.append(y_data_part)
                
        except Exception as e:
            print(f"Error loading data for item {item}: {e}")
            continue

    if X_data is None or y_data is None:
        return np.array([]), np.array([])
        
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    return X_data, y_data

def load_single_dataset(_read_path, _read_sig_path, desc_prefix):
    
    all_files = _collect_files(_read_sig_path, desc_prefix)
    
    if all_files.size == 0:
        print(f"Warning: {desc_prefix} - No .csv files found in {_read_sig_path}.")
        return np.array([]), np.array([])

    X_data, y_data = _load_data(_read_path, _read_sig_path, all_files, f"{desc_prefix} - Loading data")
    
    if X_data.size == 0 or y_data.size == 0:
        print(f"Error: {desc_prefix} - Failed to load data.")
        return np.array([]), np.array([])

    y_data[y_data == 2] = 1 
  
    y_data = onehot(y_data, y_data.shape[1])
    
    return X_data, y_data
