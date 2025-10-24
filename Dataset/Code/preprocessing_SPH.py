import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict
import h5py
import ast
from scipy.signal import butter, sosfiltfilt, resample
import pywt

#################################################################

## Function List

## bandpass_filter: bandpass filter signal
## baseline_correction: correct signal baseline
## apply_wavelet_transform: apply wavelet denoising
## downsample_signal: resample signal
## select_leads: select desired ECG leads
## segment_ecg: segment signal into fixed-length blocks
## is_clean_segment: check segment quality
## get_label: MI/Normal label extraction from metadata
## _collect_and_save_data: main preprocessing and saving pipeline

#################################################################

lead_combos = [
    ["v1","v2","v3","v4","v5","v6"], ["i","ii","iii","avl","avr","avf"],
    ["v1","v2","v3","v4","v5","v6","i","ii","iii","avl","avr","avf"],
    ["ii", "iii", "avf"], ["i", "avl", "v5"], ["i", "ii", "v3"], ["v1", "v3", "v5"]
]
orig_fs = 500
target_fs = 250
seg_len_sec = 10

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

def select_leads(ecg, all_leads, desired_leads):
    idx = []
    lead_names = [ld.lower() for ld in all_leads]
    for ld in desired_leads:
        if ld in lead_names:
            idx.append(lead_names.index(ld))
        else:
            return None
    return ecg[:, idx]

def segment_ecg(sig, fs=250, length_sec=10):
    seg_len = int(fs * length_sec)
    if sig.shape[0] < seg_len:
        return []
    return [sig[i:i+seg_len] for i in range(0, sig.shape[0]-seg_len+1, seg_len)]

def is_clean_segment(seg):
    if np.any(np.ptp(seg, axis=0) == 0):
        return False
    return True

def get_label(aha_code):
    """MI/Normal label extraction logic from SPH metadata."""
    codes = set(str(aha_code).replace(",", ";").replace("+", ";").replace(";", ";").split(";"))
    codes = {c.strip() for c in codes if c.strip()}
    mi_prefixes = ('160','161','165','166')
    is_mi = any(c.startswith(mi_prefixes) for c in codes)
    is_normal = '1' in codes and len(codes) == 1
    if is_mi: return 1
    elif is_normal: return 0
    else: return None

def _collect_and_save_data(meta_df, outpath, records_dir):
    
    seg_len = target_fs * seg_len_sec
    
    patient_segment_clean = defaultdict(lambda: defaultdict(dict))
    segment_count_by_group = defaultdict(int)

    print("\n--- [1/2] Checking all segment noise ---")
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        label = get_label(row.get('AHA_Code'))
        if label is None: continue

        pid, rec_id = str(row['Patient_ID']), row['ECG_ID']
        group_id = f"{pid}|{rec_id}"
        h5_path = os.path.join(records_dir, f"{rec_id}.h5")
        if not os.path.exists(h5_path): continue

        try:
            with h5py.File(h5_path, 'r') as f:
                ecg = np.array(f['ecg']).T
                all_leads = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
                
                for combo_idx, leads in enumerate(lead_combos):
                    rec = select_leads(ecg, all_leads, leads)
                    if rec is None: continue

                    rec = bandpass_filter(rec, fs=orig_fs)
                    rec = baseline_correction(rec)
                    rec = apply_wavelet_transform(rec)
                    rec = downsample_signal(rec, orig_fs=orig_fs, target_fs=target_fs)
                    segs = segment_ecg(rec, fs=target_fs, length_sec=seg_len_sec)
                    
                    for idx_seg, seg in enumerate(segs):
                        patient_segment_clean[group_id][idx_seg][combo_idx] = is_clean_segment(seg)
                    segment_count_by_group[group_id] = max(segment_count_by_group[group_id], len(segs))
        except Exception:
            continue

    os.makedirs(outpath, exist_ok=True)
    
    for combo_idx, leads in enumerate(lead_combos):
        lead_str = "_".join(leads)
        print(f"\n==== Processing SPH leads: {lead_str} ====")
        all_X, all_y = [], []

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"Leads={lead_str} (Save)"):
            label = get_label(row.get('AHA_Code'))
            if label is None: continue

            pid, rec_id = str(row['Patient_ID']), row['ECG_ID']
            group_id = f"{pid}|{rec_id}"
            h5_path = os.path.join(records_dir, f"{rec_id}.h5")
            if not os.path.exists(h5_path): continue

            try:
                with h5py.File(h5_path, 'r') as f:
                    ecg = np.array(f['ecg']).T 
                    all_leads = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
                    rec = select_leads(ecg, all_leads, leads)
                    if rec is None: continue

                    # Preprocessing Pipeline
                    rec = bandpass_filter(rec, fs=orig_fs)
                    rec = baseline_correction(rec)
                    rec = apply_wavelet_transform(rec)
                    rec = downsample_signal(rec, orig_fs=orig_fs, target_fs=target_fs)
                    segs = segment_ecg(rec, fs=target_fs, length_sec=seg_len_sec)
                    if not segs: continue

                    keep_list = []
                    for idx_seg, seg in enumerate(segs):
                        clean_flag = patient_segment_clean[group_id].get(idx_seg, {}).get(combo_idx, False)
                        if clean_flag:
                            keep_list.append(seg)

                    if not keep_list: continue

                    segs = np.stack(keep_list, axis=0) 
                    y = np.full((segs.shape[0],), label, dtype=np.int8)

            except Exception:
                continue

            all_X.append(segs)
            all_y.append(y)

        if all_X:
            all_X = np.concatenate(all_X, axis=0)
            all_y = np.concatenate(all_y, axis=0)
        else:
            all_X = np.empty((0, seg_len, len(leads)))
            all_y = np.empty((0,), dtype=np.int8)

        print(f"  Total segments: {all_X.shape}")
        
        np.save(os.path.join(outpath, f"sph_ecg_{lead_str}.npy"), all_X)
        np.save(os.path.join(outpath, f"sph_label_{lead_str}.npy"), all_y)
        print(f"  Saved to {outpath}")

def main(path, outpath):    
    try:
        meta_df = pd.read_csv(os.path.join(path, "metadata.csv"))
        records_dir = os.path.join(path, "records")
        code_xlsx_path = os.path.join(path, "code.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing metadata file: {e}")
        return

    _collect_and_save_data(meta_df, outpath, records_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SPH data for MI/Normal classification.")
    parser.add_argument("--path", type=str, required=True, help="Path to the directory containing SPH metadata (metadata.csv, records folder)")
    parser.add_argument("--outpath", type=str, required=True, help="Path to save the preprocessed SPH data (.npy files)")

    args = parser.parse_args()
    main(args.path, args.outpath)
