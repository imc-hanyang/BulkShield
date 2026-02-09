"""
Traditional Machine Learning Models Training Script for Fraud Detection

This module implements the training pipeline for traditional ML-based anomaly
detection models including HMM, ARIMA, and SARIMA. These models are trained
on windowed time-series data using one-class classification approach.

Models Supported:
    - HMM: Hidden Markov Model for sequence probability scoring
    - ARIMA: Autoregressive Integrated Moving Average for forecasting deviation
    - SARIMA: Seasonal ARIMA for capturing periodic patterns

Pipeline:
    1. Load user transaction sequence file paths and labels
    2. Convert sequences to sliding window arrays (memory-optimized)
    3. Train models only on normal class data (one-class approach)
    4. Evaluate using anomaly scores with percentile-based threshold

Usage:
    python ml_main.py

Note:
    Uses ProcessPoolExecutor for parallel model training.
    Memory-optimized chunked loading to handle large datasets.
"""

import numpy as np
import time
import json
import os
import concurrent.futures
import sys
import traceback

# Data loader and model imports
from data_utils import load_train_data, load_test_data
from eval_utils import print_evaluation_report
from models.hmm_model import HMMModel
from models.arima_model import ARIMAModel
from models.sarima_model import SARIMAModel

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    'seq_len': 10,                # Sliding window size
    'random_seed': 42,            # Random seed for reproducibility
    'param_file': 'best_params.json',  # Hyperparameter file path

    # Default parameters (fallback if file not found)
    'default_params': {
        "arima_order": [1, 0, 0],
        "sarima_order": [1, 0, 0],
        "sarima_seasonal": [1, 0, 0, 4]
    },
    'hmm_components': 5           # Number of HMM hidden states
}


# ==========================================
# 2. Utility Functions
# ==========================================
def load_best_params():
    """
    Load optimal hyperparameters from JSON file.
    
    Returns:
        Dictionary with ARIMA/SARIMA orders, or defaults if file not found.
    """
    if os.path.exists(CONFIG['param_file']):
        print(f"[Config] Loading parameters from '{CONFIG['param_file']}'...")
        with open(CONFIG['param_file'], 'r') as f:
            params = json.load(f)
            params['arima_order'] = tuple(params['arima_order'])
            params['sarima_order'] = tuple(params['sarima_order'])
            params['sarima_seasonal'] = tuple(params['sarima_seasonal'])
            return params
    else:
        print("[Warning] Parameter file not found. Using default values.")
        return CONFIG['default_params']

import pandas as pd
import used_raw_columns
from tqdm import tqdm
import csv

def load_and_window_data(paths, labels, seq_len):
    """
    Load and window sequence data for ML model training.
    
    Converts file paths to a single large numpy array using sliding window.
    Uses csv module instead of pandas to prevent segfaults on large datasets.
    Implements chunked processing for memory optimization.
    
    Args:
        paths: List of CSV file paths.
        labels: Corresponding labels for each file.
        seq_len: Sliding window size.
        
    Returns:
        Tuple of (X_windows, y_labels) as numpy arrays.
    """
    print(f"\n[Memory Optimization] Pre-calculating dataset size for {len(paths)} files...")
    
    chunk_size = 100000
    all_X_chunks = []
    all_y_chunks = []
    
    current_X = []
    current_y = []
    
    target_cols = used_raw_columns.columns
    
    for i, (path, label) in enumerate(tqdm(zip(paths, labels), total=len(paths), desc="Loading & Windowing")):
        try:
            # csv module for safe loading (avoids pandas segfaults)
            rows = []
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract columns and convert to float
                    try:
                        vals = [float(row.get(c, 0)) for c in target_cols]
                    except (ValueError, TypeError):
                        vals = [0.0] * len(target_cols)
                    rows.append(vals)
            
            data = np.array(rows, dtype=np.float32)
            
            # Skip files shorter than window size
            if len(data) < seq_len:
                continue
                
            # Apply sliding window
            num_windows = len(data) - seq_len + 1
            windows = np.array([data[j:j+seq_len] for j in range(num_windows)])
            
            # Expand labels for each window
            window_labels = np.full(num_windows, label)
            
            current_X.append(windows)
            current_y.append(window_labels)
            
            # Chunk-based merging for memory management
            if len(current_X) >= 1000:  # Merge every 1000 files
                chunk_X = np.concatenate(current_X, axis=0)
                chunk_y = np.concatenate(current_y, axis=0)
                all_X_chunks.append(chunk_X)
                all_y_chunks.append(chunk_y)
                current_X = []
                current_y = []
                
        except Exception:
            pass

    # Process remaining data
    if current_X:
        chunk_X = np.concatenate(current_X, axis=0)
        chunk_y = np.concatenate(current_y, axis=0)
        all_X_chunks.append(chunk_X)
        all_y_chunks.append(chunk_y)
        
    print("[Memory Optimization] Merging all chunks...")
    if not all_X_chunks:
        return np.array([]), np.array([])
        
    final_X = np.concatenate(all_X_chunks, axis=0)
    final_y = np.concatenate(all_y_chunks, axis=0)
    
    print(f"Dataset Created. Shape: {final_X.shape}, {final_X.nbytes / 1e9:.2f} GB")
    return final_X, final_y


def evaluate(name, y_true, y_scores, timestamp):
    """
    Evaluate model using standardized metrics.
    
    For OCC models, higher scores indicate anomalies.
    Uses 80th percentile as decision threshold.
    
    Args:
        name: Model name for reporting.
        y_true: True labels.
        y_scores: Anomaly scores.
        timestamp: Run timestamp for output naming.
    """
    # OCC models: use 80th percentile as threshold (top 20% as anomalies)
    threshold = np.percentile(y_scores, 80)
    print_evaluation_report(name, y_true, y_scores, threshold=threshold, category='ml', timestamp=timestamp)



def run_single_model(name, model, X_train, X_test):
    """
    Train and evaluate a single model in a subprocess.
    
    Args:
        name: Model identifier.
        model: Model instance with train/predict_score methods.
        X_train: Training data (normal samples only).
        X_test: Test data (all samples).
        
    Returns:
        Tuple of (name, scores, error). Error is None on success.
    """
    pid = os.getpid()
    print(f">>> [{name}] Training started (Process ID: {pid})")
    start_time = time.time()

    try:
        # Train model
        model.train(X_train)
        # Get anomaly scores
        scores = model.predict_score(X_test)

        elapsed = time.time() - start_time
        print(f">>> [{name}] Completed! ({elapsed:.2f}s)")
        return name, scores, None  # Success: error is None
    except Exception as e:
        print(f"!!! [{name}] Error occurred: {e}")
        return name, None, str(e)


def main():
    # 1. 파라미터 로드
    params = load_best_params()
    
    # [설정] 실행 시간 고정 (폴더명 통일)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"[Main] Start Check... Timestamp: {timestamp}")

    # 2. 데이터 로드 및 전처리 (Lazy Path -> Numpy Window Array)

    print("[Main] 데이터 로딩 및 전처리 (Memory Optimized)...")
    
    # 경로만 로드
    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()
    
    if not train_paths or not test_paths: return

    # Numpy Array로 변환 (Windowing 포함)
    X_train_windows, y_train_windows = load_and_window_data(train_paths, train_labels, CONFIG['seq_len'])
    X_test_windows, y_test_windows = load_and_window_data(test_paths, test_labels, CONFIG['seq_len'])
    
    if len(X_train_windows) == 0 or len(X_test_windows) == 0:
        print("데이터 생성 실패 (윈도우 부족 등).")
        return

    print(f"[Info] Train: {len(X_train_windows)}개, Test: {len(X_test_windows)}개 샘플 생성 완료.")

    # 3. 학습용 정상 데이터 추출 (Numpy Masking)
    X_train_normal = X_train_windows[y_train_windows == 0]
    y_test = y_test_windows
    print(f"[Info] 학습용 정상 데이터: {len(X_train_normal)}개")

    # 4. 모델 정의
    models = {
        'HMM': HMMModel(n_components=CONFIG['hmm_components']),
        'ARIMA': ARIMAModel(order=params['arima_order']),
        'SARIMA': SARIMAModel(order=params['sarima_order'], seasonal_order=params['sarima_seasonal'])
    }

    # 5. 멀티 프로세싱 실행
    print(f"\n[Parallel] {len(models)}개 모델 동시 학습 시작 (CPU 병렬 처리)...")
    results = {}

    # max_workers=3 : 모델 개수만큼 프로세스 생성
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(models)) as executor:
        # 작업 제출
        future_to_name = {
            executor.submit(run_single_model, name, model, X_train_normal, X_test_windows): name
            for name, model in models.items()
        }

        # 결과 수집 (완료되는 순서대로)
        for future in concurrent.futures.as_completed(future_to_name):
            name, scores, error = future.result()
            if error is None:
                results[name] = scores
            else:
                print(f"[Main] {name} 모델 실패로 건너뜀.")

    # 6. 최종 리포트 출력
    print("\n" + "=" * 50)
    print("      최종 평가 리포트 (Evaluation Report)")
    print("=" * 50)

    for name, scores in results.items():
        evaluate(name, y_test, scores, timestamp)



if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    try:
        main()
    except Exception as e:
        print("!!! [Main] Critical Error Occurred !!!")
        traceback.print_exc()