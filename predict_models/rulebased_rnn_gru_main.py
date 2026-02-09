"""
RNN and GRU Training with Rule-Based Labels

This module trains RNN and GRU classifiers using rule-based fraud labels
as ground truth. Both models are trained sequentially in one execution.

Architecture:
    - CustomRNN: Basic RNN with multiple layers
    - CustomGRU: GRU variant with gating mechanism
    - Rule-based labels from threshold logic

Configuration:
    - Optimized for H100 GPU with large batch size (4096)
    - High worker count (60) for I/O parallelization

Usage:
    python rulebased_rnn_gru_main.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import csv
from datetime import datetime
from tqdm import tqdm

# Models
from models.rnn_model import CustomRNN
from models.gru_model import CustomGRU

# Trainer & Utils
from dl_trainer import train_one_epoch, get_model_predictions
from eval_utils import print_evaluation_report
from data_utils import LazyDataset, pad_collate_fn, extract_user_id_from_filename

# ==========================================
# 1. Configuration (H100 Optimized)
# ==========================================
CONFIG = {
    'models_to_run': ['rnn', 'gru'], 
    
    'input_size': 33,    # Feature Size (Updated to Match Verification)
    'output_size': 2,    # Binary Classification (Normal/Fraud)
    'hidden_size': 128,  # Increased for efficiency on H100
    'num_layers': 3,     # Deeper network
    'batch_size': 4096,  # Large batch for H100 utilizing memory
    'learning_rate': 0.001,
    'epochs': 200,       # Full training
    'device': 'cuda:0',  # User Request
    'num_workers': 60,   # High CPU core utilization for IO
    
    # Paths (RuleBased)
    'train_label_dir': '/home/srt/Dataset/label/rulebased_label_train_per_user',
    'test_label_dir': '/home/srt/Dataset/label/rulebased_label_test_per_user',
    'train_seq_dir': '/home/srt/Dataset/feature/sequence_data_28d_train',
    'test_seq_dir': '/home/srt/Dataset/feature/sequence_data_28d_test',
    
    'save_base_dir': '/home/srt/workspaces/predict_models/ml_results'
}

# ==========================================
# 2. Robust Data Loading (No Pandas)
# ==========================================
def collect_files(directory, prefix):
    """Collects files recursively from group folders."""
    file_map = {}
    if not os.path.exists(directory):
        return file_map
    
    group_dirs = [d for d in os.listdir(directory) if d.startswith('group_')]
    for gd in group_dirs:
        g_path = os.path.join(directory, gd)
        try:
            files = os.listdir(g_path)
            for f in files:
                if f.startswith(prefix) and f.endswith('.csv'):
                    # filename: prefix_USERID_DATE.csv
                    # Extract UserIDRobustly
                    # Remove prefix
                    rest = f[len(prefix):]
                    parts = rest.split('_')
                    user_id = parts[0]
                    file_map[user_id] = os.path.join(g_path, f)
        except:
            continue
    return file_map

def load_rulebased_data(label_dir, seq_dir, split_name="Train"):
    print(f"\n[{split_name}] Mapping Data Files...")
    
    # 1. Map Files
    label_files = collect_files(label_dir, 'label_')
    seq_files = collect_files(seq_dir, 'seq_')
    
    # 2. Find Intersection
    common_users = sorted(list(set(label_files.keys()) & set(seq_files.keys())))
    print(f"[{split_name}] Matched Users: {len(common_users):,}")
    
    if len(common_users) == 0:
        return [], []

    input_paths = []
    labels = []
    
    # 3. Read Labels (Streaming)
    print(f"[{split_name}] Reading Label CSVs...")
    for user_id in tqdm(common_users, ncols=100, mininterval=1.0):
        lbl_path = label_files[user_id]
        
        try:
            with open(lbl_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check for 'is_risk' or 'is_fraud'
                    val = 0
                    if 'is_risk' in row:
                        val = int(float(row['is_risk']))
                    elif 'is_fraud' in row:
                         val = int(float(row['is_fraud']))
                    
                    labels.append(val)
                    input_paths.append(seq_files[user_id])
                    break # Read one row per user
        except:
            continue
            
    return input_paths, labels

# ==========================================
# 3. Main Loop
# ==========================================
def run_model_training(model_name, train_loader, val_loader, timestamp):
    device = torch.device(CONFIG['device'])
    print(f"\n{'='*50}")
    print(f"🚀 Training Model: {model_name.upper()}")
    print(f"{'='*50}")
    
    # Init Model
    if model_name == 'rnn':
        model = CustomRNN(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['output_size'], CONFIG['num_layers']).to(device)
    elif model_name == 'gru':
        model = CustomGRU(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['output_size'], CONFIG['num_layers']).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Custom Save Logic
    from sklearn.metrics import f1_score
    
    # Dynamic Save Directory based on User Request
    if model_name == 'rnn':
        SAVE_DIR = '/home/srt/ml_results/rulebased_rnn'
    elif model_name == 'gru':
        SAVE_DIR = '/home/srt/ml_results/rulebased_gru'
    else:
        SAVE_DIR = f'/home/srt/ml_results/rulebased_{model_name}'
        
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    best_score = -1.0
    best_epoch = 0
    best_model_state = None
    
    # Training Loop
    for epoch in range(1, CONFIG['epochs'] + 1):
        # A. Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, model_type=model_name)
        print(f"   [Epoch {epoch}] Loss: {avg_loss:.6f}")
        
        # B. Eval
        y_true, y_scores = get_model_predictions(model, val_loader, device, model_type=model_name)
        
        # C. Metric Calculation
        threshold = 0.5
        y_pred = (np.array(y_scores) >= threshold).astype(int)
        current_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Report (no save)
        print_evaluation_report(
            f"{model_name}", y_true, y_scores, threshold=threshold, 
            category='rulebased_dl', save=False, timestamp=timestamp
        )
        
        print(f"   [Result] Epoch {epoch} | Macro F1: {current_macro_f1:.4f} (Best: {best_score:.4f})")
        
        # D. Best Save
        if current_macro_f1 > best_score:
            best_score = current_macro_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
            save_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")
            torch.save(best_model_state, save_path)
            print(f"   🔥 New Best Model Found! Saved to {save_path}")
            
    print(f"\n🏆 Final Best {model_name.upper()}: Epoch {best_epoch}, Macro F1 {best_score:.4f}")
    
    # Final Save
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        final_y_true, final_y_scores = get_model_predictions(model, val_loader, device, model_type=model_name)
        final_threshold = 0.5
        final_y_pred = (np.array(final_y_scores) >= final_threshold).astype(int)
        final_macro_f1 = f1_score(final_y_true, final_y_pred, average='macro', zero_division=0)
        
        # Metrics
        final_metrics = {
            'model': model_name,
            'best_epoch': best_epoch,
            'macro_f1': final_macro_f1,
            'threshold': final_threshold
        }
        
        final_metrics_path = os.path.join(SAVE_DIR, f"{model_name}_metrics.json")
        import json
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
            
        # Scores/Labels
        np.save(os.path.join(SAVE_DIR, f"{model_name}_scores.npy"), final_y_scores)
        np.save(os.path.join(SAVE_DIR, f"{model_name}_labels.npy"), final_y_true)
            
        print(f"💾 Final metrics and results saved to: {SAVE_DIR}")


def main():
    # Setup
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # 1. Load Data
    print("📥 Loading Datasets (RuleBased)...")
    train_paths, train_labels = load_rulebased_data(CONFIG['train_label_dir'], CONFIG['train_seq_dir'], "Train")
    test_paths, test_labels = load_rulebased_data(CONFIG['test_label_dir'], CONFIG['test_seq_dir'], "Test")
    
    if not train_paths or not test_paths:
        print("❌ Data Load Failed.")
        return
        
    # 2. Create Loaders
    print("🔨 Creating DataLoaders...")
    train_ds = LazyDataset(train_paths, train_labels)
    test_ds = LazyDataset(test_paths, test_labels)
    
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
        collate_fn=pad_collate_fn, num_workers=CONFIG['num_workers'], 
        pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        test_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
        collate_fn=pad_collate_fn, num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    
    # 3. Monitor Imbalance
    print(f"Train Imbalance: {sum(train_labels)} / {len(train_labels)} ({sum(train_labels)/len(train_labels):.2%})")
    
    # 4. Run Models
    for model_type in CONFIG['models_to_run']:
        try:
            run_model_training(model_type, train_loader, val_loader, timestamp)
        except Exception as e:
            print(f"Error running {model_type}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
