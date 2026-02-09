"""
LSTM Deep Learning Training Script for Fraud Detection

This module implements the training pipeline for LSTM-based fraud detection
models. Supports both classification (CustomLSTM) and autoencoder (LSTM-AE)
approaches.

Models Supported:
    - lstm: Standard LSTM classifier with sequence classification
    - lstm-ae: LSTM AutoEncoder for unsupervised anomaly detection

Pipeline:
    1. Load user transaction sequence file paths and labels
    2. Create lazy-loading datasets for memory efficiency
    3. Train model with per-epoch validation
    4. Save best model based on Macro F1 score

Usage:
    python dl_main.py

Output:
    - Model checkpoint: /home/srt/ml_results/lstm/<timestamp>/<model>_best.pth
    - Metrics JSON: <model>_metrics.json
    - Prediction scores and labels as .npy files
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import traceback
import os
import copy  # For deep copying model parameters

# Model imports
from models.lstm_model import CustomLSTM
from models.lstm_ae_model import LSTMAutoEncoder
from dl_trainer import train_one_epoch, get_model_predictions
from eval_utils import print_evaluation_report

# Data utilities
from data_utils import LazyDataset, pad_collate_fn, load_train_data, load_test_data

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    # Models to train: 'lstm' (classifier) or 'lstm-ae' (autoencoder)
    'models_to_run': ['lstm'], 
    
    'input_size': 33,          # Number of input features per time step
    'output_size': 2,          # Classification output (normal/fraud)
    'hidden_size': 64,         # LSTM hidden dimension
    'num_layers': 2,           # Number of stacked LSTM layers
    'batch_size': 64,          # Training batch size
    'learning_rate': 0.001,    # Adam learning rate
    'epochs': 20,              # Maximum training epochs
}



def run_single_model(model_name, device, train_loader, val_loader, timestamp):
    """
    Train and evaluate a single model with best checkpoint saving.
    
    Args:
        model_name: Model identifier ('lstm' or 'lstm-ae').
        device: CUDA device for training.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        timestamp: Run timestamp for output directory naming.
    """
    print(f"\n{'='*50}")
    print(f"🚀 [{model_name.upper()}] Training Started")
    print(f"{'='*50}")

    # 1. Model initialization
    if model_name == 'lstm':
        model = CustomLSTM(
            CONFIG['input_size'], 
            CONFIG['hidden_size'], 
            CONFIG['output_size'], 
            CONFIG['num_layers']
        ).to(device)
        criterion = nn.CrossEntropyLoss()  # Classification loss
    elif model_name == 'lstm-ae':
        model = LSTMAutoEncoder(
            CONFIG['input_size'], 
            CONFIG['hidden_size'], 
            CONFIG['num_layers']
        ).to(device)
        criterion = nn.MSELoss()  # Reconstruction loss
    else:
        print(f"❌ Unknown model: {model_name}")
        return

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_score = -1.0
    best_model_state = None
    best_epoch = 0

    # 2. Training Loop
    from sklearn.metrics import f1_score
    SAVE_DIR = f'/home/srt/ml_results/lstm/{timestamp}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, CONFIG['epochs'] + 1):
        # A. Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, model_type=model_name
        )
        print(f"   [Epoch {epoch}] Train Loss: {train_loss:.6f}")

        # B. Validation evaluation
        y_true, y_scores = get_model_predictions(model, val_loader, device, model_type=model_name)
        
        # Threshold: use 90th percentile for AE, 0.5 for classifier
        if model_name == 'lstm-ae':
            threshold = np.percentile(y_scores, 90)
        else:
            threshold = 0.5

        # Calculate Macro F1 for best model selection
        y_pred = (np.array(y_scores) >= threshold).astype(int)
        # Macro Average F1 Score calculation
        current_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Print simple report (keeping detailed report call if needed, or just printing locally)
        # Using eval_utils just for printing to console without saving
        print(f"   [Validation] Computing metrics with threshold: {threshold:.4f}...")
        metrics = print_evaluation_report(
            model_name, y_true, y_scores, threshold=threshold, 
            category='dl', save=False, timestamp=timestamp
        )
        
        print(f"   [Result] Epoch {epoch} | Macro F1: {current_macro_f1:.4f} (Best: {best_score:.4f})")

        # D. Save best model based on Macro F1
        if current_macro_f1 > best_score:
            best_score = current_macro_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
            # Immediate checkpoint save
            save_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")
            torch.save(best_model_state, save_path)
            print(f"   🔥 New Best Model Found! Saved to {save_path}")
            
    print(f"\n🏆 Final Best Model: Epoch {best_epoch}, Macro F1: {best_score:.4f}")

    # 3. Best Model 로드 및 최종 평가 결과 저장 (metrics.json 등)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        final_y_true, final_y_scores = get_model_predictions(model, val_loader, device, model_type=model_name)
        
        if model_name == 'lstm-ea':
            final_threshold = np.percentile(final_y_scores, 90)
        else:
            final_threshold = 0.5
            
        # Manually saving metrics to the specific directory to match requirements
        final_y_pred = (np.array(final_y_scores) >= final_threshold).astype(int)
        final_macro_f1 = f1_score(final_y_true, final_y_pred, average='macro', zero_division=0)
        
        # Save explicit metrics file
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
            
        # Also save scores and labels as requested ("keep other saved things")
        np.save(os.path.join(SAVE_DIR, f"{model_name}_scores.npy"), final_y_scores)
        np.save(os.path.join(SAVE_DIR, f"{model_name}_labels.npy"), final_y_true)
            
        print(f"💾 최종 메트릭 및 결과 저장 완료: {SAVE_DIR}")


def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"📅 실행 타임스탬프: {timestamp}")

    # 데이터 로드
    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()

    if not train_paths:
        print("❌ Train 데이터 로드 실패.")
        return
    if not test_paths:
        print("❌ Test 데이터 로드 실패.")
        return

    # Dataset & DataLoader
    train_dataset = LazyDataset(train_paths, train_labels)
    val_dataset = LazyDataset(test_paths, test_labels)

    # DataLoader 생성
    # num_workers=0으로 설정 (안전성) -> 필요 시 증가
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=pad_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=pad_collate_fn, num_workers=4)

    # 정의된 모델 순차 실행
    for model_name in CONFIG['models_to_run']:
        try:
            run_single_model(model_name, device, train_loader, val_loader, timestamp)
        except Exception as e:
            print(f"❌ [{model_name}] 실행 중 오류 발생:")
            traceback.print_exc()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    try:
        main()
    except Exception as e:
        print("!!! [Main] Critical Error Occurred !!!")
        traceback.print_exc()

