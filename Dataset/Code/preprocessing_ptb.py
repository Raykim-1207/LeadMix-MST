import os
import numpy as np
import re
import pywt
import wfdb
from scipy.signal import resample, butter, sosfiltfilt
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict

#################################################################

## Function List

## bandpass_filter: bandpass filter signal
## baseline_correction: correct signal baseline
## apply_wavelet_transform: apply wavelet denoising
## downsample_signal: resample signal
## select_leads: select desired ECG leads
## segment_ecg: segment signal into fixed-length blocks
## is_clean_segment: check segment quality
## get_label_example_ptb: MI/Normal label extraction from comments
## load_ptb_records: load PTB records
## _collect_and_save_data: main preprocessing and saving pipeline

#################################################################

PTB_DB_ROOT = ""
CONTROLS_FILE_NAME = "CONTROLS"
healthy_controls = set()

lead_combos = [
    ["v1","v2","v3","v4","v5","v6"], ["i","ii","iii","avl","avr","avf"],
    ["v1","v2","v3","v4","v5","v6","i","ii","iii","avl","avr","avf"],
    ["ii", "iii", "avf"], ["i", "avl", "v5"], ["i", "ii", "v3"], ["v1", "v3", "v5"]
]
orig_fs = 1000
target_fs = 250
seg_len_sec = 10

def load_ptb_records(ptb_path):
    for patient in os.listdir(ptb_path):
        p_dir = os.path.join(ptb_path, patient)
        if not os.path.isdir(p_dir): continue
        for f in os.listdir(p_dir):
            if not f.endswith(".hea"): continue
            rec = f[:-4]
            path = os.path.join(p_dir, rec)
            try:
                record, fields = wfdb.rdsamp(path)
                fields['file_name']  = rec
                fields['patient_id'] = patient
                yield record, fields
            except Exception as e:
                print(f"Error reading {path}: {e}")

def get_label_example_ptb(fields):
    pid = fields['patient_id']
    rec = fields['file_name']
    key = f"{pid}/{rec}"
    txt = " ".join(fields.get("comments", [])).lower() if isinstance(fields.get("comments", []), list) else str(fields.get("comments", "")).lower()

    if "myocardial infarction" in txt:
        return 1
    if key in healthy_controls:
        return 0
    return None

def bandpass_filter(sig, lowcut=0.5, highcut=25.0, fs=1000, order=4):
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

def downsample_signal(sig, orig_fs=1000, target_fs=250):
    num = int(sig.shape[0] * target_fs / orig_fs)
    return resample(sig, num, axis=0)

def select_leads(record, fields, desired_leads):
    sigs = [s.lower() for s in fields["sig_name"]]
    try:
        idx = [sigs.index(ld) for ld in desired_leads]
    except ValueError:
        return None
    return record[:, idx]

def segment_ecg(sig, fs=250, length_sec=10):
    seg_len = int(fs * length_sec)
    if sig.shape[0] < seg_len:
        return []
    return [sig[i:i+seg_len] for i in range(0, sig.shape[0]-seg_len+1, seg_len)]

def is_clean_segment(seg):
    if np.any(np.ptp(seg, axis=0) == 0):
        return False
    return True

def _collect_and_save_data(ptb_dir, outpath):
    
    seg_len = target_fs * seg_len_sec
    loader_fn = lambda: load_ptb_records(ptb_dir)
    label_fn  = get_label_example_ptb
    group_fn  = lambda fields: fields['patient_id']

    patient_segment_clean = defaultdict(lambda: defaultdict(dict))
    segment_count_by_patient = defaultdict(int)

    print("\n--- [1/2] Checking all segment noise ---")
    
    for record, fields in tqdm(loader_fn(), desc="Pass 1: Checking Noise"):
        g = group_fn(fields)
        label = label_fn(fields)
        if label is None: continue

        for combo_idx, leads in enumerate(lead_combos):
            rec = select_leads(record, fields, leads)
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
    
    for leads in lead_combos:
        lead_str = "_".join(leads)
        print(f"\n==== Processing PTB leads: {lead_str} ====")
        all_X, all_y = [], []

        for record, fields in tqdm(loader_fn(), desc=f"Leads={lead_str} (Save)"):
            label = label_fn(fields)
            g = group_fn(fields)
            rec = select_leads(record, fields, leads)
            if rec is None or label is None: continue

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

        if all_X:
            all_X = np.concatenate(all_X, axis=0)
            all_y = np.concatenate(all_y, axis=0)
        else:
            all_X = np.empty((0, seg_len, len(leads)))
            all_y = np.empty((0,), dtype=np.int8)

        print(f"  Total segments: {all_X.shape}")
        
        np.save(os.path.join(outpath, f"ptb_ecg_{lead_str}.npy"), all_X)
        np.save(os.path.join(outpath, f"ptb_label_{lead_str}.npy"), all_y)
        print(f"  Saved to {outpath}")

    print("\n[ALL LEAD COMBOS] Done.")

def main(path, outpath):
    global healthy_controls
    
    controls_path = os.path.join(path, CONTROLS_FILE_NAME)
    if os.path.exists(controls_path):
        with open(controls_path, "r") as f:
            healthy_controls = { line.strip() for line in f if line.strip() }
    else:
        print(f"Warning: {CONTROLS_FILE_NAME} not found at {controls_path}. Healthy control labeling may be incomplete.")
        healthy_controls = set()

    _collect_and_save_data(path, outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PTB data for MI/Normal classification.")
    parser.add_argument("--path", type=str, required=True, help="Path to the root directory containing PTB-DB patient folders (e.g., .../ptb-diagnostic-ecg-database-1.0.0)")
    parser.add_argument("--outpath", type=str, required=True, help="Path to save the preprocessed PTB data (.npy files)")

    args = parser.parse_args()
    main(args.path, args.outpath)
