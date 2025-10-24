import os
import sys
import argparse
import numpy as np

from LeadMix_MST import *

parser = argparse.ArgumentParser()
parser.add_argument('--read_path', type=str, required=True)            # Root path of label data
parser.add_argument('--read_sig_path', type=str, required=True)        # Root path where preprocessed SPH .npy files are located
parser.add_argument('--batch_size', type=int, default=32)              # Batch size
parser.add_argument('--epochs', type=int, default=25)                  # Number of training epochs
parser.add_argument('--model_save_dir', type=str, default='./')        # Directory to save trained model
parser.add_argument('--model_name', type=str, default='LeadMixMST')    # Base name for the saved model file
parser.add_argument('--lr', type=float, default=1e-3)                  # Learning rate
parser.add_argument('--weight_decay', type=float, default=1e-3)        # Weight decay for AdamW optimizer
parser.add_argument('--seg_len', type=int, default=2500)               # Input segment length (Sampling rate * 10 s)
parser.add_argument('--lead_part', type=str, required=True)            # The specific leads to train on (e.g., v1_v3_v5)


args, unknown = parser.parse_known_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    try:
        from LeadMix_MST_model import LeadMixMST
        train_loader, criterion, num_leads = _get_pytorch_data_loaders(args, device)
    except Exception as e:
        print(f"FATAL ERROR: Data/Model setup failed. {e}")
        sys.exit(1)

    model = LeadMixMST(args.seg_len, num_leads).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.model_save_dir, exist_ok=True)
    final_model_path = os.path.join(args.model_save_dir, f"{args.model_name}_{args.lead_part}_FINAL_STATE.pth")
        
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_pytorch_epoch(model, train_loader, criterion, device, optimizer)
      
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete. Final model state saved to: {final_model_path}")
