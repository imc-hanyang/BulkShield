"""
GRU with Time Delta Context Training Script

This module implements the training pipeline for time-aware GRU fraud detection.
The model incorporates discretized time deltas between transactions as additional
context, enabling better detection of temporal fraud patterns.

Architecture:
    - TimeContextDataset: Lazy-loads sequences with time delta computation
    - GRUWithTime: GRU model with time delta embedding layer
    - Evaluation: Macro F1 score with 0.5 threshold

Key Features:
    - Time delta encoding for inter-event temporal patterns
    - Best model checkpointing based on validation Macro F1
    - High batch size (2048) and worker count (32) for H100 optimization

Usage:
    python gru_with_time_main.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import logging
from tqdm import tqdm
import datetime
import copy
from sklearn.metrics import f1_score

# Import model and data utilities
from models.gru_with_time_model import GRUWithTime
from data_utils import load_train_data, load_test_data, TimeContextDataset, collate_fn_with_time
from eval_utils import print_evaluation_report

# Force line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)

# Configuration constants
MODEL_NAME = "GRU"
SAVE_DIR_NAME = "gru_with_time_delta"
LOG_DIR = "/home/srt/workspaces/predict_models/logs/gru_with_time"

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
    """
    Evaluate model on validation/test data.
    
    Args:
        model: GRUWithTime model instance.
        loader: Validation DataLoader.
        device: CUDA device.
        
    Returns:
        Tuple of (true_labels, predicted_probabilities) for fraud class.
    """
    model.eval()
    all_y = []
    all_probs = []
    
    with torch.no_grad():
        for x, d, lens, y in loader:
            x, d, lens = x.to(device), d.to(device), lens.to(device)
            logits = model(x, d, lens)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_y.extend(y.tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_y, all_probs


def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    
    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()
    if not train_paths: return
    
    train_ds = TimeContextDataset(train_paths, train_labels)
    test_ds = TimeContextDataset(test_paths, test_labels)
    
    input_dim = len(train_ds.feature_cols)
    logger.info(f"Feature Dimension: {input_dim}")
    
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=32, collate_fn=collate_fn_with_time)
    val_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=32, collate_fn=collate_fn_with_time)
    
    model = GRUWithTime(input_dim, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    logger.info(f"=== Starting {MODEL_NAME} Training ===")
    
    BASE_SAVE_DIR = "/home/srt/workspaces/ml_results"
    
    epochs = 10
    log_interval = 20
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (x, d, lens, y) in enumerate(tqdm(train_loader, desc=f"Ep {epoch}")):
            x, d, lens, y = x.to(device), d.to(device), lens.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, d, lens)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % log_interval == 0:
                logger.info(f"Ep {epoch} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Ep {epoch} | Avg Loss: {avg_loss:.4f}")
        
        y_true, y_scores = evaluate(model, val_loader, device)
        print_evaluation_report(MODEL_NAME, y_true, y_scores, threshold=0.5, category=SAVE_DIR_NAME, save=False, timestamp=timestamp)
        
        y_pred = (np.array(y_scores) >= 0.5).astype(int)
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        logger.info(f"Ep {epoch} | Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"New Best Macro F1: {best_f1:.4f}")
            
            final_save_dir = os.path.join(BASE_SAVE_DIR, SAVE_DIR_NAME, timestamp)
            os.makedirs(final_save_dir, exist_ok=True)
            save_path = os.path.join(final_save_dir, f"best_{MODEL_NAME}.pth")
            torch.save(best_model_wts, save_path)
            logger.info(f"Saved best model to {save_path}")

    # Final Save
    print_evaluation_report(MODEL_NAME, y_true, y_scores, threshold=0.5, category=SAVE_DIR_NAME, save=True, timestamp=timestamp)

if __name__ == "__main__":
    main()
