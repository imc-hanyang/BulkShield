"""
Combined Transformer Fraud Detection Training Script

This module implements the training pipeline for a simplified Transformer-based
fraud detection model. Uses mixed precision training and H100 GPU optimization.

Architecture:
    - CombinedSequenceDataset: Lazy-loads combined user sequences
    - TransformerCombined: Simplified Transformer with CLS token pooling
    - Mixed precision training with GradScaler for H100 efficiency

Key Features:
    - Feature projection layer before Transformer encoder
    - Learnable CLS token for sequence-level classification
    - AdamW optimizer with weight decay
    - Best model checkpointing based on validation Macro F1

Usage:
    python transformer_combined_main.py
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
from models.transformer_combined_model import TransformerCombined
from data_utils import load_train_data, load_test_data, CombinedSequenceDataset, collate_fn_combined_seq
from eval_utils import print_evaluation_report

# Force line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)

# Configuration constants
MODEL_NAME = "TransformerCombined"
SAVE_DIR_NAME = "transformer_combined"
LOG_DIR = "/home/srt/workspaces/predict_models/logs/transformer_combined"

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
    """
    Evaluate model on validation/test data with mixed precision.
    
    Args:
        model: TransformerCombined model instance.
        loader: Validation DataLoader.
        device: CUDA device.
        
    Returns:
        Tuple of (true_labels, predicted_probabilities) for fraud class.
    """
    model.eval()
    all_y = []
    all_probs = []
    
    with torch.no_grad():
        for x, lens, y in loader:
            x, lens = x.to(device), lens.to(device)
            # Mixed Precision inference
            with torch.cuda.amp.autocast():
                logits = model(x, lens)
                probs = torch.softmax(logits, dim=1)[:, 1]
                
            all_y.extend(y.tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_y, all_probs


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    
    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()
    if not train_paths: return
    
    # Dataset
    train_ds = CombinedSequenceDataset(train_paths, train_labels)
    test_ds = CombinedSequenceDataset(test_paths, test_labels)
    
    input_dim = len(train_ds.feature_cols)
    logger.info(f"Feature Dimension: {input_dim}")
    
    # Dataloader (H100 Optimized)
    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, 
        num_workers=32, collate_fn=collate_fn_combined_seq,
        pin_memory=True
    )
    val_loader = DataLoader(
        test_ds, batch_size=2048, shuffle=False, 
        num_workers=32, collate_fn=collate_fn_combined_seq,
        pin_memory=True
    )
    
    model = TransformerCombined(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    best_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    logger.info(f"=== Starting {MODEL_NAME} Training ===")
    
    BASE_SAVE_DIR = "/home/srt/ml_results"
    
    epochs = 20
    log_interval = 20
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        for batch_idx, (x, lens, y) in enumerate(pbar):
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(x, lens)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            total_loss += loss_val
            
            if (batch_idx + 1) % log_interval == 0:
                logger.info(f"Ep {epoch} | Batch {batch_idx+1} | Loss: {loss_val:.4f}")
                
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Ep {epoch} | Avg Loss: {avg_loss:.4f}")
        
        # Validation
        y_true, y_scores = evaluate(model, val_loader, device)
        
        # Save Eval Report EVERY EPOCH (save=True to persist metrics, but usually save=False to avoid massive file clutter? 
        # User said: "매 epoch 마다 eval_utils 를 실행해서 모든 eval methods 를 진행해서 결과를 내도록 해."
        # Usually 'results' implies printing report. Saving CSVs every epoch might be too much, but I'll set save=False to just print report, 
        # AND check best model to save model weights. 
        # WAIT, user request: "해당 이포크에서 best f1 macro avg 스코어가 나왔다면 ~/ml_results/transformer_combined 디렉터리에 저장되게 해."
        # This confirms saving MODEL WEIGHTS on best. 
        # What about eval RESULTS? "결과를 내도록 해" -> usually means print.
        
        category = SAVE_DIR_NAME
        print_evaluation_report(MODEL_NAME, y_true, y_scores, threshold=0.5, category=category, save=False, timestamp=timestamp)
        
        # Macro F1 Check
        y_pred = (np.array(y_scores) >= 0.5).astype(int)
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        logger.info(f"Ep {epoch} | Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"New Best Macro F1: {best_f1:.4f}")
            
            final_save_dir = os.path.join(BASE_SAVE_DIR, SAVE_DIR_NAME)
            os.makedirs(final_save_dir, exist_ok=True)
            save_path = os.path.join(final_save_dir, f"best_{MODEL_NAME}.pth")
            torch.save(best_model_wts, save_path)
            logger.info(f"Saved best model to {save_path}")

    # Final Save of Best Model Eval
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        y_true, y_scores = evaluate(model, val_loader, device)
        print_evaluation_report(MODEL_NAME, y_true, y_scores, threshold=0.5, category=SAVE_DIR_NAME, save=True, timestamp=timestamp)

if __name__ == "__main__":
    main()
