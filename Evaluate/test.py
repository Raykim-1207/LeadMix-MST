import os
import sys
import argparse
import numpy as np
import glob
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from LeadMix_MST import *
from evaluate import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_save_dir', type=str, default='./')        # Root path of label data
parser.add_argument('--model_name', type=str, default='LeadMixMST')    # Model name saved
parser.add_argument('--batch_size', type=int, default=32)              # Batch size
parser.add_argument('--seg_len', type=int, default=2500)               # Sampling rate * 10s
parser.add_argument('--output_nums', type=int, default=1)              # Number of output classes
parser.add_argument('--input_length', type=int, default=2500)          # Input segment length (Sampling rate * 10 s)
parser.add_argument('--lead_part', type=str, required=True)            # The specific leads to train on (e.g., v1_v3_v5)

args, unknown = parser.parse_known_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def _get_pytorch_data_loaders(read_sig_path, name, lead_part, seg_len, batch_size) -> Tuple[DataLoader, np.ndarray]:
    from dataset import ECGDataset, get_lead_mean_std
    
    te_x = os.path.join(read_sig_path, f"{name.lower()}_ecg_{lead_part}.npy")
    te_y = os.path.join(read_sig_path, f"{name.lower()}_label_{lead_part}.npy")
    
    if not os.path.exists(te_x):
        raise FileNotFoundError(f"Test ECG file not found for {name}: {te_x}")
    
    SPH_SIG_PATH_ROOT = os.path.join(os.path.dirname(read_sig_path), 'sph')
    SPH_X = os.path.join(SPH_SIG_PATH_ROOT, f"sph_ecg_{lead_part}.npy")
    
    if not os.path.exists(SPH_X):
        SPH_X = os.path.join(read_sig_path, f"sph_ecg_{lead_part}.npy")
        if not os.path.exists(SPH_X):
            raise FileNotFoundError("Could not locate SPH training data for normalization.")
    
    means, stds = get_lead_mean_std(SPH_X)
    
    test_dataset  = ECGDataset(te_x, te_y, means, stds, augment=False, use_mmap=True, seg_len=seg_len)
    loader_kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=(device.type == "cuda"), shuffle=False)
    test_loader   = DataLoader(test_dataset, **loader_kwargs)
    
    return test_loader, np.load(te_y, mmap_mode='r') 


def _run_pytorch_prediction(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.permute(0, 2, 1).to(device) 
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x).squeeze(1)
                probs = torch.sigmoid(out)
            all_probs.extend(probs.cpu().numpy().tolist())
            
    return np.array(all_probs)

if __name__ == '__main__':
    print(args, file=sys.stdout, flush=True)

    test_scenarios = [
        {'name': 'PTB', 'read_sig_path': args.read_sig_path, 'index': 0},
        {'name': 'PTBXL', 'read_sig_path': args.read_sig_path, 'index': 1}
    ]
    
    num_leads = len(args.lead_part.split("_"))
    model = LeadMixMST(args.seg_len, num_leads).to(device)
    
    model_name_full = f"{args.model_name}_{args.lead_part}_FINAL_STATE.pth"
    model_path = os.path.join(args.model_save_dir, model_name_full)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nLoaded trained model state from: {model_path}")
    except Exception as e:
        print(f"\nError loading model state from {model_path}: {e}")
        sys.exit(1)

    for ds in test_scenarios:        
        try:
            test_loader, y_test_raw_labels = _get_pytorch_data_loaders(
                ds['read_sig_path'], ds['name'], args.lead_part, args.seg_len, args.batch_size
            )
        except FileNotFoundError as e:
            print(f"Skipping {ds['name']}: {e}")
            continue

        test_probs = _run_pytorch_prediction(model, test_loader, device)
    
        y_test_true = y_test_raw_labels[:, 0].flatten() 
      
        from evaluate import evaluate_metrics_single_step
        
        evaluate_metrics_single_step(
            y_test_true, 
            test_probs, 
            args.model_save_dir, 
            ds['name'], 
            ds['index']
        )
        
