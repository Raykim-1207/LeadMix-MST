import os
import numpy as np
import re
import pywt
import wfdb
from scipy.signal import resample, butter, sosfiltfilt
import pandas as pd
import ast
from tqdm import tqdm
from collections import defaultdict
import argparse
import h5py

#################################################################

## Function List

## bandpass_filter: bandpass filter signal
## baseline_correction: correct signal baseline
## apply_wavelet_transform: apply wavelet denoising
## downsample_signal: resample signal
## select_leads: select desired ECG leads
## segment_ecg: segment signal into fixed-length blocks
## is_clean_segment: check segment quality
## load_ptbxl_data: load PTB-XL records/metadata
## get_label_example_ptbxl: MI/Normal label extraction from SCP codes
## _collect_and_save_data: main preprocessing and saving pipeline

#################################################################

PTBXL_META = "ptbxl_database.csv"
PTBXL_RECORDS_DIR = "records500"

df_ptbxl = None 

lead_combos = [
    ["v1","v2","v3","v4","v5","v6"], ["i","ii","iii","avl","avr","avf"],
    ["v1","v2","v3","v4","v5","v6","i","ii","iii","avl","avr","avf"],
    ["ii", "iii", "avf"], ["i", "avl", "v5"], ["i", "ii", "v3"], ["v1", "v3", "v5"]
]
orig_fs = 500
target_fs = 250
seg_len_sec = 10

def load_ptbxl_data(ptbxl_root_path, fs_choice="500Hz"):
    global df_ptbxl
    
    meta_path = os.path.join(ptbxl_root_path, PTBXL_META)
    if df_ptbxl is None:
        try:
            df_ptbxl = pd.read_csv(meta_path)
            df_ptbxl['filename_wfdb'] = df_ptbxl['filename_hr'].fillna(df_ptbxl['filename_lr'])
        except FileNotFoundError:
            print(f"Error: PTBXL metadata not found at {meta_path}")
            return

    for _, row in df_ptbxl.iterrows():
        rec_path_rel = row['filename_hr'] if fs_choice == "500Hz" else row['filename_lr']
        h5_path_rel = os.path.join(PTBXL_RECORDS_DIR, rec_path_rel)
        h5_path_full = os.path.join(ptbxl_root_path, h5_path_rel)

        fields = {
            'file_name': os.path.basename(rec_path_rel),
            'patient_id': str(row['patient_id']),
            'record_name': rec_path_rel,
            'h5_path': h5_path_full
        }
        yield row, fields

def get_label_example_ptbxl(row):
    try:
        scp_codes = ast.literal_eval(row['scp_codes'])
        mi_keywords = {"IMI", "ASMI", "ILMI", "AMI", "ALMI"}
        
        # Check for MI
        if any(code in mi_keywords for code in scp_codes.keys()):
            return 1
        # Check for Normal ECG
        elif "NORM" in scp_codes.keys():
            return 0
        else:
            return None
    except Exception:
        return None

def bandpass_filter(sig, lowcut=0.5, highcut=25.0, fs=500, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, sig, axis=0)

def baseline_correction(sig):
    return sig - np.mean(sig, axis=0, keepdims=True)

def apply_wavelet_transform(sig, wavelet='db4', level=1):
    coeffs = pywt.wavedec(sig, wavelet, axis=0, level=level)
    coeffs[-1] = np.zeros_like(coeffs[-1])
    return pywt.waverec(coeffs, wavelet, axis=0)

def downsample_signal(sig, orig_fs=500, target_fs=250):
    num = int(sig.shape[0] * target_fs / orig_fs)
    return resample(sig, num, axis=0)

def select_leads(ecg, desired_leads):
    all_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    idx = []
    try:
        idx = [all_leads.index(ld.lower()) for ld in desired_leads]
    except ValueError:
        return None
    return ecg[:, idx] 

def segment_ecg(sig, fs=250, length_sec=10):
    seg_len = int(fs * length_sec)
    if sig.shape[0] < seg_len: return []
    return [sig[i:i+seg_len] for i in range(0, sig.shape[0]-seg_len+1, seg_len)]

def is_clean_segment(seg):
    if np.any(np.ptp(seg, axis=0) == 0):
        return False
    return True

def _collect_and_save_data(ptbxl_root_path, outpath):
    
    seg_len = target_fs * seg_len_sec
    loader_gen = load_ptbxl_data(ptbxl_root_path, fs_choice="500Hz")
    group_fn  = lambda fields: fields['patient_id']

    patient_segment_clean = defaultdict(lambda: defaultdict(dict))
    segment_count_by_patient = defaultdict(int)

    print("\n--- [1/2] Checking all segment noise ---")
    
    for row, fields in tqdm(loader_gen, desc="Pass 1: Checking Noise"):
        label = get_label_example_ptbxl(row)
        if label is None: continue
        
        g = group_fn(fields)
        h5_path = fields['h5_path']
        
        try:
            with h5py.File(h5_path, 'r') as f:
                ecg = np.array(f['ecg']).T
        except Exception:
            continue
            
        for combo_idx, leads in enumerate(lead_combos):
            rec = select_leads(ecg, leads)
            if rec is None: continue

            rec = bandpass_filter(rec, fs=orig_fs)
            rec = baseline_correction(rec)
            rec = apply_wavelet_transform(rec)
            rec = downsample_signal(rec, orig_fs=orig_fs, target_fs=target_fs)
            segs = segment_ecg(rec, fs=target_fs, length_sec=seg_len_sec)
            
            for idx_seg, seg in enumerate(segs):
                patient_segment_clean[g][idx_seg][combo_idx] = is_clean_segment(seg)
            segment_count_by_patient[g] = max(segment_count_by_patient[g], len(segs))
            
    valid_indices_by_group = defaultdict(list)
    for g in patient_segment_clean:
        for idx in range(segment_count_by_patient[g]):
            if idx in patient_segment_clean[g] and all(patient_segment_clean[g][idx]):
                valid_indices_by_group[g].append(idx)

    os.makedirs(outpath, exist_ok=True)
    print("\n--- [2/2] Building and saving datasets ---")
    
    loader_gen = load_ptbxl_data(ptbxl_root_path, fs_choice="500Hz")
    
    for leads in lead_combos:
        lead_str = "_".join(leads)
        print(f"\n==== Processing PTBXL leads: {lead_str} ====")
        all_X, all_y = [], []
        
        patient_segment_count = defaultdict(int)
        label_patient_set = defaultdict(set)
        label_segment_count = defaultdict(int)

        for row, fields in tqdm(loader_gen, desc=f"Leads={lead_str} (Save)"):
            label = get_label_example_ptbxl(row)
            if label is None: continue

            g = group_fn(fields)
            h5_path = fields['h5_path']
            
            try:
                with h5py.File(h5_path, 'r') as f:
                    ecg = np.array(f['ecg']).T
            except Exception:
                continue
                
            rec = select_leads(ecg, leads)
            if rec is None: continue

            rec = bandpass_filter(rec, fs=orig_fs)
            rec = baseline_correction(rec)
            rec = apply_wavelet_transform(rec)
            rec = downsample_signal(rec, orig_fs=orig_fs, target_fs=target_fs)
            segs = segment_ecg(rec, fs=target_fs, length_sec=seg_len_sec)
            
            valid_indices = set(valid_indices_by_group[g])
            keep_list = [seg for idx_seg, seg in enumerate(segs) if idx_seg in valid_indices]

            if not keep_list: continue

            segs = np.stack(keep_list, axis=0)
            y = np.full((segs.shape[0],), label, dtype=np.int8)

            all_X.append(segs)
            all_y.append(y)
            
            patient_segment_count[g] += len(keep_list)
            label_patient_set[label].add(g)
            label_segment_count[label] += len(keep_list)

        if all_X:
            all_X = np.concatenate(all_X, axis=0)
            all_y = np.concatenate(all_y, axis=0)
        else:
            all_X = np.empty((0, seg_len, len(leads)))
            all_y = np.empty((0,), dtype=np.int8)

        n_segments = all_X.shape[0]
        n_patients = len([p for p, c in patient_segment_count.items() if c > 0])

        print(f"  Total segments: {all_X.shape}")
        
        print(f"  Patients (with ≥1 segment): {n_patients}")
        for label in sorted(label_patient_set.keys()):
            print(f"    Label={label}: Patients={len(label_patient_set[label])}, Segments={label_segment_count[label]}")

        np.save(os.path.join(outpath, f"ptbxl_ecg_{lead_str}.npy"), all_X)
        np.save(os.path.join(outpath, f"ptbxl_label_{lead_str}.npy"), all_y)
        print(f"  Saved to {outpath}")

    print("\n[ALL LEAD COMBOS] Done.")

def main(path, outpath):
    _collect_and_save_data(path, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL data for MI/Normal classification.")
    parser.add_argument("--path", type=str, required=True, help="Path to the root directory containing PTB-XL metadata (ptbxl_database.csv, records500 folder).")
    parser.add_argument("--outpath", type=str, required=True, help="Path to save the preprocessed PTB-XL data (.npy files).")

    args = parser.parse_args()
    main(args.path, args.outpath)
