import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_metrics_single_step(y_true, y_pred_probs, outpath, name, index=0):

    y_pred = (y_pred_probs > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_probs)
        auprc = average_precision_score(y_true, y_pred_probs)
    else:
        auc = 0.0
        auprc = 0.0
        
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    s = 0
    
    _step_cm = confusion_matrix(y_true, y_pred)
    cm_dir = os.path.join(outpath, 'confusion')
    os.makedirs(cm_dir, exist_ok=True)
    
    cm_path = os.path.join(cm_dir, f"{name}{index}_step_{s}.csv")
    pd.DataFrame(_step_cm).to_csv(cm_path, index=False)
    
    cr_dir = os.path.join(outpath, 'classification_report')
    os.makedirs(cr_dir, exist_ok=True)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cr_path = os.path.join(cr_dir, f"{name}{index}_step_{s}.csv")
    pd.DataFrame(report).transpose().to_csv(cr_path)

    print(f"Evaluation results for {name} (Index {index}) saved to {outpath}")
    
    return {
        "accuracy": acc, 
        "precision": prec, 
        "recall": rec, 
        "f1_macro": f1_macro, 
        "auc": auc, 
        "auprc": auprc
    }
