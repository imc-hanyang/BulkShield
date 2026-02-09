"""
Simple Models Training Script for Fraud Detection

This module trains simple machine learning models (Logistic Regression, SVR, MLP)
on aggregated sequence features. Input sequences are converted to fixed-size
feature vectors using statistical aggregations.

Models:
    - LogisticRegression: Linear binary classifier
    - SVR: Support Vector Regressor for anomaly scoring
    - MLP: Multi-layer Perceptron with hidden layers

Feature Aggregation:
    - Per numeric column: Mean, Max, Standard Deviation
    - Global: Event count, Duration (max - min timestamp)

Usage:
    python simple_models_main.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
import csv
import logging
from tqdm import tqdm
import datetime
import copy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Force line buffering for real-time logging redirection
sys.stdout.reconfigure(line_buffering=True)

# Import Models
from models.simple_models import LogisticRegression, SVR, MLP
# Import Data Utils
from data_utils import load_train_data, load_test_data
from eval_utils import print_evaluation_report

# Use the same NUMERIC_COLS as transformer for consistency
# Copied from models/transformer_model.py
NUMERIC_COLS = [
    "seat_cnt", "buy_amt", "refund_amt", "cancel_fee", "route_dist_km",
    "travel_time", "lead_time_buy", "lead_time_ref", "hold_time",
    "dep_hour",
    "route_buy_cnt", "fwd_dep_hour_median",
    "rev_buy_cnt", "rev_ratio",
    "completed_fwd_cnt", "completed_fwd_dep_interval_median",
    "completed_fwd_dep_hour_median", "completed_rev_cnt",
    "completed_rev_dep_interval_median", "completed_rev_dep_hour_median",
    "unique_route_cnt",
    "rev_dep_hour_median", "rev_return_gap",
    "overlap_cnt", "same_route_cnt", "rev_route_cnt",
    "repeat_interval", "adj_seat_refund_flag",
    "recent_ref_cnt", "recent_ref_amt", "recent_ref_rate",
]

# Logging Setup
# User will handle logging via stdout redirection
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Base Model Save Directory
BASE_SAVE_DIR = "/home/srt/ml_results/simple_models"

class SimpleDataset(Dataset):
    def __init__(self, input_paths, labels, numeric_cols):
        self.input_paths = input_paths
        self.labels = labels
        self.numeric_cols = numeric_cols
        self.num_feats = len(numeric_cols)
        # Output dim = num_feats * 3 (Mean, Max, Std) + 2 (Count, Duration)
        self.output_dim = self.num_feats * 3 + 2

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        path = self.input_paths[idx]
        label = self.labels[idx]
        
        # Read CSV and Aggregate
        # Aggregations: Mean, Max, Std for each numeric column
        # Global stats: Count, Duration (max timestamp - min timestamp)
        
        data_values = {col: [] for col in self.numeric_cols}
        timestamps = []
        
        try:
            with open(path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse timestamp logic if needed, but for 'duration' lets try to use 'timestamp' col
                    if 'timestamp' in row:
                         # Simple string sort or parse? Let's treat as string for sort if format is ISO
                         timestamps.append(row['timestamp'])
                         
                    for col in self.numeric_cols:
                        val_str = row.get(col, "0")
                        try:
                            raw_val = float(val_str)
                            val = math.log1p(abs(raw_val))
                            if raw_val < 0: val = -val
                        except:
                            val = 0.0
                        data_values[col].append(val)
        except Exception as e:
            # logger.warning(f"Error reading {path}: {e}")
            pass
            
        # Compute Features
        feats = []
        
        # 1. Count
        count = len(timestamps) if timestamps else 0
        feats.append(float(count))
        
        # 2. Duration (in minutes)
        if count > 1:
            try:
                # Fast parse start/end
                # Assuming YYYY-MM-DD HH:MM:SS format mostly
                # To be safe and fast, strictly 2 parsed
                t_start = datetime.datetime.strptime(min(timestamps), "%Y-%m-%d %H:%M:%S")
                t_end = datetime.datetime.strptime(max(timestamps), "%Y-%m-%d %H:%M:%S")
                duration_mins = (t_end - t_start).total_seconds() / 60.0
            except:
                duration_mins = 0.0
        else:
            duration_mins = 0.0
        feats.append(duration_mins)
            
        # 3. Aggregates per column
        for col in self.numeric_cols:
            vals = data_values[col]
            if not vals:
                feats.extend([0.0, 0.0, 0.0]) # Mean, Max, Std
                continue
                
            arr = np.array(vals, dtype=np.float32)
            mean_v = np.mean(arr)
            max_v = np.max(arr)
            std_v = np.std(arr)
            
            feats.extend([mean_v, max_v, std_v])
            
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_model(model_name, model, train_loader, val_loader, criterion, device, session_timestamp, epochs=20):
    logger.info(f"[{model_name}] Training Starting...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Logging constraint: short interval logs
        log_interval = 100 
        
                
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"[{model_name}] Ep {epoch+1}")):
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1) # [B, 1]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % log_interval == 0 and batch_idx > 0:
                logger.info(f"[{model_name}] Ep {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        logger.info(f"[{model_name}] Ep {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
        # Validation
        y_true, y_scores = evaluate(model, val_loader, device, model_name)
        
        # Use eval_utils for comprehensive reporting
        # Note: timestamp is not passed to train_model, let's generate one or pass it? 
        # For now, we can just use a placeholder or None if save=False
        print_evaluation_report(model_name, y_true, y_scores, threshold=0.5, category='dl', save=False)

        # Calculate Macro F1 for Model Saving
        # SVR outputs raw values, others output logits. y_scores from evaluate need to be consistent.
        # Logic in evaluate ensures y_scores are probabilities or comparable scores.
        
        y_pred = (np.array(y_scores) >= 0.5).astype(int)
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        logger.info(f"[{model_name}] Ep {epoch+1} | Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"[{model_name}] New Best Macro F1: {best_f1:.4f}")
            
            # Save best model to separate directory per model AND per execution
            # Structure: ml_results/simple_models/{TIMESTAMP}/{ModelName}/best_{ModelName}.pth
            run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Or reuse global start time? 
            # User wants: "n번 실행할때 저장과 n+1 번째 실행의 저장은 다른 디렉터리에 해야해"
            # So we need a SESSION TIMESTAMP generated ONCE per execution (at start of script).
            
            # We will use the 'timestamp' variable passed/generated in main/train_model.
            # Let's assume train_model receives 'session_timestamp'.
            
            save_dir = os.path.join(BASE_SAVE_DIR, session_timestamp, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"best_{model_name}.pth")
            torch.save(best_model_wts, save_path)
            logger.info(f"[{model_name}] Saved best model to {save_path}")
            
    return best_f1

def evaluate(model, loader, device, model_name="Model"):
    model.eval()
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Helper to get scores (probabilities or regression values)
            if model_name in ['LogisticRegression', 'MLP']:
                scores = torch.sigmoid(outputs)
            else: # SVR
                scores = outputs
                
            all_targets.extend(targets.cpu().tolist())
            all_scores.extend(scores.cpu().view(-1).tolist())
            
    return all_targets, all_scores

def main():
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    
    # 1. Load Data
    logger.info("Loading Data Paths...")
    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()
    
    if not train_paths:
        logger.error("No training data found.")
        return

    # 2. Dataset
    # num_workers=32 for 100 cores
    train_ds = SimpleDataset(train_paths, train_labels, NUMERIC_COLS)
    test_ds = SimpleDataset(test_paths, test_labels, NUMERIC_COLS)
    
    # Input Dim check
    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[0]
    logger.info(f"Input Feature Dimension: {input_dim}")
    
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=32)
    val_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=32)
    
    # 3. Models
    # Define models to train
    models_to_train = [
        ("LogisticRegression", LogisticRegression(input_dim), nn.BCEWithLogitsLoss()),
        ("SVR", SVR(input_dim), nn.MSELoss()), # Regression Loss for SVR
        ("MLP", MLP(input_dim), nn.BCEWithLogitsLoss())
    ]
    
    # Generate Session Timestamp relative to Start Time of this Script Execution
    session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Session Timestamp: {session_timestamp}")

    for name, model, criterion in models_to_train:
        logger.info(f"=== Starting {name} ===")
        model = model.to(device)
        train_model(name, model, train_loader, val_loader, criterion, device, session_timestamp, epochs=20)
        logger.info(f"=== Finished {name} ===\n")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Exception occurred", exc_info=True)
