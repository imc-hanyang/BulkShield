
import os
import glob
import csv
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import argparse
import time
import sys
import gc
from scipy.stats import ttest_ind

# 1. Configuration
SEQ_DIR = "/home/srt/Dataset/feature/sequence_data_28d_train"
LOG_FILE = "/home/srt/workspaces/predict_models/verification_stats.txt"

# Datasets to compare
DATASETS = {
    'GMM': {
        'label_dir': "/home/srt/Dataset/label/label_train_per_user",
        'model_path_hint': "/home/srt/ml_results/dl/[TIMESTAMP]/lstm_best_epoch[N].pth"
    },
    'RuleBased': {
        'label_dir': "/home/srt/Dataset/label/rulebased_label_train_per_user",
        'model_path_hint': "/home/srt/workspaces/predict_models/ml_results/rulebased_lstm/best_model.pth"
    }
}

NUM_FEATURES = 37

def log_print(msg):
    """Print to stdout and append to file."""
    print(msg)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"[WARN] Log file write failed: {e}")

def get_files_map(directory, prefix):
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
                    base = f[len(prefix):]
                    parts = base.split('_')
                    user_id = parts[0]
                    file_map[user_id] = os.path.join(g_path, f)
        except OSError:
            continue
    return file_map

def worker_process_user(args):
    """
    Computes User-level aggregated stats: Mean, Std, Min, Max for each feature.
    """
    user_id, label_path, seq_path = args
    try:
        # Read Label
        is_fraud = 0
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            if 'is_risk' in row:
                is_fraud = int(float(row['is_risk']))
            elif 'label' in row:
                is_fraud = int(float(row['label']))
            elif 'is_fraud' in row:
                is_fraud = int(float(row['is_fraud']))
        
        # Read Features
        features = []
        with open(seq_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vals = []
                for k, v in row.items():
                    try:
                        vals.append(float(v))
                    except ValueError:
                        continue
                if vals:
                    features.append(vals)
                    
        if not features:
            return None

        features_np = np.array(features, dtype=np.float32)
        
        # --- Aggregation per User (Collapse Time Axis) ---
        # Shape: (Timesteps, Features) -> (Features,)
        
        u_mean = np.mean(features_np, axis=0) # User Average
        u_std  = np.std(features_np, axis=0)  # User Variability
        u_min  = np.min(features_np, axis=0)  # User Minimum
        u_max  = np.max(features_np, axis=0)  # User Peak
        
        return (is_fraud, u_mean, u_std, u_min, u_max)

    except Exception:
        return None

def compute_dataset_stats(dataset_name, config):
    log_print(f"\n{'='*70}")
    log_print(f"🔍 Analyzing Dataset (User Level: Mean, Std, Min, Max): {dataset_name}")
    log_print(f"Label Dir: {config['label_dir']}")
    log_print(f"{'='*70}")
    
    log_print("Mapping files...")
    label_map = get_files_map(config['label_dir'], 'label_')
    seq_map = get_files_map(SEQ_DIR, 'seq_')
    
    common_users = sorted(list(set(label_map.keys()) & set(seq_map.keys())))
    log_print(f"Found {len(common_users):,} common users.")
    
    if len(common_users) == 0:
        log_print("No users found!")
        return

    tasks = [(u, label_map[u], seq_map[u]) for u in common_users]
    
    # Store aggregated features for all users
    # Dict[Class] -> List of User Vectors
    coll_means = {0: [], 1: []}
    coll_stds  = {0: [], 1: []}
    coll_mins  = {0: [], 1: []}
    coll_maxs  = {0: [], 1: []}
    
    log_print(f"Processing with {mp.cpu_count()} cores...")
    
    start_time = time.time()
    
    with mp.Pool(processes=95) as pool:
        it = pool.imap_unordered(worker_process_user, tasks, chunksize=500)
        
        for result in tqdm(it, total=len(tasks), ncols=100, mininterval=1.0):
            if result is None:
                continue
            
            is_fraud, u_mean, u_std, u_min, u_max = result
            
            # Collect
            coll_means[is_fraud].append(u_mean)
            coll_stds[is_fraud].append(u_std)
            coll_mins[is_fraud].append(u_min)
            coll_maxs[is_fraud].append(u_max)

    elapsed = time.time() - start_time
    log_print(f"Processing finished in {elapsed:.2f} seconds.")
    
    # Validation
    n0 = len(coll_means[0])
    n1 = len(coll_means[1])
    log_print(f"\n📊 Users Summary: Normal={n0:,}, Fraud={n1:,}")
    
    if n0 < 2 or n1 < 2:
        log_print("[ERROR] Not enough data for statistical testing.")
        return

    # Convert to Numpy & Detect Num Features
    for d in [coll_means, coll_stds, coll_mins, coll_maxs]:
        d[0] = np.array(d[0], dtype=np.float32)
        d[1] = np.array(d[1], dtype=np.float32)

    # Auto-detect number of features from data
    if len(coll_means[0]) > 0:
        actual_num_features = coll_means[0].shape[1]
    elif len(coll_means[1]) > 0:
        actual_num_features = coll_means[1].shape[1]
    else:
        actual_num_features = 0
        
    log_print(f"Detected Features: {actual_num_features}")

    # Perform Verification for each aggregation type
    # Type | Feature | ...
    
    agg_types = [
        ("User-MEAN", coll_means),
        ("User-STD",  coll_stds),
        ("User-MIN",  coll_mins),
        ("User-MAX",  coll_maxs)
    ]
    
    for agg_name, data_dict in agg_types:
        log_print(f"\n🧪 Verification: [{agg_name}] (Comparing Normal vs Fraud)")
        log_print(f"{'Feat':<5} {'T-Stat':<10} {'P-Value':<10} {'Significant?':<12} {'NormAvg':<10} {'FraudAvg':<10}")
        log_print("-" * 75)
        
        data0 = data_dict[0]
        data1 = data_dict[1]
        
        for i in range(actual_num_features):
            col0 = data0[:, i]
            col1 = data1[:, i]
            
            # T-Test
            with np.errstate(all='ignore'): # suppress division by zero warnings
                t_stat, p_val = ttest_ind(col0, col1, equal_var=False, nan_policy='omit')
            
            m0 = np.mean(col0)
            m1 = np.mean(col1)
            
            # Handle NaN results (e.g., zero variance)
            if np.isnan(t_stat) or np.isnan(p_val):
                if np.isclose(m0, m1):
                    # Means are equal, variance likely 0 -> No difference
                    t_stat = 0.0
                    p_val = 1.0
                else:
                    # Means different but variance 0 -> Infinite difference
                    # This happens if all Normal=0.0 and all Fraud=0.0001 (constant)
                    # We can mark as significant or just leave as inf
                    t_stat = 9999.0 if m1 > m0 else -9999.0
                    p_val = 0.0 
            
            # Check Significance
            sig = "YES" if p_val < 0.05 else "NO"
            p_str = f"{p_val:.2e}"
                
            log_print(f"{i:<5} {t_stat:<10.2f} {p_str:<10} {sig:<12} {m0:<10.2f} {m1:<10.2f}")


def main():
    # Clear log file
    with open(LOG_FILE, "w") as f:
        f.write(f"User-Level Verification (Mean/Std/Min/Max) started at {time.ctime()}\n")
        
    print("🚀 Starting Statistical Verification (User-Level Aggregates)")
    for name, conf in DATASETS.items():
        compute_dataset_stats(name, conf)

if __name__ == "__main__":
    main()
