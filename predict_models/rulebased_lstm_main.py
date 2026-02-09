"""
LSTM Training with Rule-Based Labels

This module trains an LSTM classifier using rule-based fraud labels as
ground truth instead of human-annotated labels. This approach enables
semi-supervised learning where the rule-based heuristics provide training signals.

Architecture:
    - CustomLSTM model with configurable layers and hidden size
    - Rule-based labels from threshold logic (refund amount + rate)
    - Best model saved based on Macro F1 score

Usage:
    python rulebased_lstm_main.py

GPU: cuda:3 (configurable)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import traceback
import os
import copy
import csv
from datetime import datetime
from tqdm import tqdm

# 기존 모듈 재사용
from models.lstm_model import CustomLSTM
from dl_trainer import train_one_epoch, get_model_predictions
from eval_utils import print_evaluation_report
from data_utils import LazyDataset, pad_collate_fn, collect_files_from_groups

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'model_name': 'lstm',
    'input_size': 37,
    'output_size': 2,
    'hidden_size': 128,  # 모델 용량 증가
    'num_layers': 2,
    'batch_size': 2048,  # H100 메모리 활용을 위해 대폭 증가
    'learning_rate': 0.001,
    'epochs': 200,       # 충분한 에폭
    'device': 'cuda:3',  # 사용자 요청
    'num_workers': 60,   # 코어가 100개이므로 I/O 병목 해소 위해 높게 설정
    
    # 데이터 경로
    'train_label_dir': '/home/srt/Dataset/label/rulebased_label_train_per_user',
    'test_label_dir': '/home/srt/Dataset/label/rulebased_label_test_per_user',
    'train_seq_dir': '/home/srt/Dataset/feature/sequence_data_28d_train',
    'test_seq_dir': '/home/srt/Dataset/feature/sequence_data_28d_test',
    
    'save_dir': '/home/srt/ml_results/rulebased_lstm'
}

# ==========================================
# 2. Data Loading (Custom with CSV module)
# ==========================================
def load_rulebased_data(label_dir, seq_dir, split_name="Train"):
    """
    Pandas read_csv 대신 csv 모듈을 사용하여 안전하게 파일 매핑 로드
    """
    print(f"\n[{split_name}] 데이터 매핑 시작...")
    
    # 1. 파일 목록 수집 (os.listdir 사용)
    # collect_files_from_groups 함수 재사용 (data_utils에 있음)
    # 이 함수는 이미 os.listdir를 사용하므로 안전함
    label_files = collect_files_from_groups(label_dir, 'label_')
    seq_files = collect_files_from_groups(seq_dir, 'seq_')
    
    # 2. 교집합 유저 찾기
    common_users = sorted(list(set(label_files.keys()) & set(seq_files.keys())))
    print(f"[{split_name}] 매칭된 유저 수: {len(common_users):,}")
    
    if len(common_users) == 0:
        return [], []

    input_paths = []
    labels = []
    
    # 3. 라벨 파일 읽기 (CSV 모듈 사용 - SegFault 방지)
    # Multi-processing으로 읽으면 더 빠르겠지만, 라벨 파일은 작아서 메인 스레드에서 해도 됨
    # 하지만 100만개면 꽤 걸림 -> TQDM으로 진행상황 표시
    
    for user_id in tqdm(common_users, desc=f"Reading {split_name} Labels"):
        lbl_path = label_files[user_id]
        
        try:
            with open(lbl_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # is_risk 컬럼 확인 (RuleBased 라벨 파일 컬럼명 확인 필요)
                    # 앞서 만든 코드에서 'is_risk'로 저장함
                    if 'is_risk' in row:
                        val = int(float(row['is_risk'])) # float string 안전 변환
                        labels.append(val)
                        input_paths.append(seq_files[user_id])
                    break # 첫 줄만 사용 (User 단위 라벨링 가정)
        except Exception as e:
            # 읽기 실패 시 건너뜀
            continue
            
    return input_paths, labels

# ==========================================
# 3. Main Training Loop
# ==========================================
def run_training():
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🚀 실행 장치: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # 1. 데이터 로드
    train_paths, train_labels = load_rulebased_data(CONFIG['train_label_dir'], CONFIG['train_seq_dir'], "Train")
    test_paths, test_labels = load_rulebased_data(CONFIG['test_label_dir'], CONFIG['test_seq_dir'], "Test")
    
    if not train_paths or not test_paths:
        print("❌ 데이터 로드 실패 (파일 없음)")
        return

    # 2. Dataset & DataLoader
    # data_utils.LazyDataset은 이미 내부적으로 csv 모듈을 사용하도록 구현되어 있음 (Step 787 확인됨)
    train_dataset = LazyDataset(train_paths, train_labels)
    test_dataset = LazyDataset(test_paths, test_labels)
    
    print(f"📦 DataLoader 설정 (Batch: {CONFIG['batch_size']}, Workers: {CONFIG['num_workers']})")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        collate_fn=pad_collate_fn, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True, # CUDA 전송 가속
        persistent_workers=True # 워커 프로세스 유지 (재생성 오버헤드 방지)
    )
    
    val_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        collate_fn=pad_collate_fn, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. 모델 초기화
    model = CustomLSTM(
        input_size=CONFIG['input_size'], 
        hidden_size=CONFIG['hidden_size'], 
        output_size=CONFIG['output_size'], 
        num_layers=CONFIG['num_layers']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print("\n✅ 학습 시작...")
    
    # Custom Save Logic
    from sklearn.metrics import f1_score
    SAVE_DIR = CONFIG['save_dir']
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    best_score = -1.0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        # A. Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, 
            model_type=CONFIG['model_name']
        )
        print(f"Epoch [{epoch}/{CONFIG['epochs']}] Train Loss: {train_loss:.6f}")
        
        # B. Evaluation (Validation)
        y_true, y_scores = get_model_predictions(model, val_loader, device, model_type=CONFIG['model_name'])
        
        # C. Metric Calculation (Macro F1)
        threshold = 0.5 
        y_pred = (np.array(y_scores) >= threshold).astype(int)
        current_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"   [Val] metrics calculating...")
        # Print report (no save)
        print_evaluation_report(
            CONFIG['model_name'], y_true, y_scores, threshold=threshold, 
            category='dl', save=False, timestamp=timestamp
        )
        
        print(f"   [Result] Epoch {epoch} | Macro F1: {current_macro_f1:.4f} (Best: {best_score:.4f})")
        
        # C. Save Best
        if current_macro_f1 > best_score:
            best_score = current_macro_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
            save_path = os.path.join(SAVE_DIR, f"{CONFIG['model_name']}_best.pth")
            torch.save(best_model_state, save_path)
            print(f"   🔥 New Best Model Found! Saved to {save_path}")
            
    # 4. Final Save & Report
    print(f"\n🏆 최종 Best Model (Epoch {best_epoch}, Macro F1: {best_score:.4f}) 저장")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        final_y_true, final_y_scores = get_model_predictions(model, val_loader, device, model_type=CONFIG['model_name'])
        final_threshold = 0.5
        final_y_pred = (np.array(final_y_scores) >= final_threshold).astype(int)
        final_macro_f1 = f1_score(final_y_true, final_y_pred, average='macro', zero_division=0)
        
        # Metrics
        final_metrics = {
            'model': CONFIG['model_name'],
            'best_epoch': best_epoch,
            'macro_f1': final_macro_f1,
            'threshold': final_threshold
        }
        
        final_metrics_path = os.path.join(SAVE_DIR, f"{CONFIG['model_name']}_metrics.json")
        import json
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
            
        # Scores/Labels
        np.save(os.path.join(SAVE_DIR, f"{CONFIG['model_name']}_scores.npy"), final_y_scores)
        np.save(os.path.join(SAVE_DIR, f"{CONFIG['model_name']}_labels.npy"), final_y_true)
        
        print(f"💾 Final metrics and results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    # 출력 버퍼링 해제 (로그 즉시 출력)
    sys.stdout.reconfigure(line_buffering=True)
    try:
        run_training()
    except Exception as e:
        print("!!! Critical Error !!!")
        traceback.print_exc()
