"""
Rule-Based Fraud Detection Model

This module implements a simple threshold-based fraud detection algorithm
using refund statistics. It serves as a baseline comparison for ML models.

Detection Rules:
    - Mark as fraud if BOTH conditions are met in 1-month window:
        1. Total refund amount >= 1,000,000 KRW
        2. Refund rate (refund_count / total_count) >= 90%

Pipeline:
    1. Load test set user IDs from sequence data
    2. Scan raw passenger CSV files in parallel
    3. Apply rule-based logic per user
    4. Evaluate against ground truth labels

Usage:
    python rulebased_model_main.py --num_cores 32
"""

import os
import glob
import csv
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import torch

# Custom Imports
import data_utils
import eval_utils

# Configuration
DEFAULT_RAW_DIR = "/home/srt/Dataset/passenger_split_final"
REFUND_THRESHOLD = 1_000_000
REFUND_RATE_THRESHOLD = 0.90
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_dot_date(date_str):
    """
    Parse date string like '2024.11.29' to datetime object.
    Returns None on failure.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip(), "%Y.%m.%d")
    except:
        return None

def process_single_user(args):
    """
    Worker function to process a single user's raw csv file.
    val_args: (file_path, user_id)
    """
    file_path, user_id = args
    
    try:
        # 1. Read CSV using csv.DictReader for speed & low memory
        rows = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            
            # Normalize header (strip spaces)
            if reader.fieldnames:
                reader.fieldnames = [x.strip() for x in reader.fieldnames]
            
            # Check required columns
            if '출발일자' not in reader.fieldnames or '반환금액' not in reader.fieldnames:
                return user_id, 0.0, "missing_col"

            # Parse relevant data
            for row in reader:
                d_str = row.get('출발일자')
                r_str = row.get('반환금액', '0')
                
                dt = parse_dot_date(d_str)
                if dt:
                    try:
                        amt = float(r_str)
                    except:
                        amt = 0.0
                    
                    rows.append((dt, amt))
        
        if not rows:
            return user_id, 0.0, "empty"

        # 2. Logic Check
        # Sort by date (though usually sorted)
        rows.sort(key=lambda x: x[0])
        
        max_date = rows[-1][0]
        # Previous 1 Month Window
        # Rule: last departure date criteria, 1 month before window
        # start_date = max_date - 1 month + 1 day
        start_date = max_date - relativedelta(months=1) + relativedelta(days=1)
        
        # Filter window
        window_rows = [r for r in rows if start_date <= r[0] <= max_date]
        
        if not window_rows:
            return user_id, 0.0, "no_window"

        # Calculate Stats
        total_refund = sum(r[1] for r in window_rows)
        pass_count = len(window_rows)
        refund_count = sum(1 for r in window_rows if r[1] > 0)
        
        refund_rate = refund_count / pass_count if pass_count > 0 else 0.0
        
        # Predict Fraud
        is_fraud = 0.0
        if (total_refund >= REFUND_THRESHOLD) and (refund_rate >= REFUND_RATE_THRESHOLD):
            is_fraud = 1.0
            
        return user_id, is_fraud, "success"

    except Exception as e:
        return user_id, 0.0, f"error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Rule-Based Fraud Detection Model")
    parser.add_argument("--num_cores", type=int, default=multiprocessing.cpu_count(), help="Number of cores to use")
    args = parser.parse_args()

    print(f"[{datetime.now()}] Starting Rule-Based Model Evaluation")
    print(f"Dataset Path: {DEFAULT_RAW_DIR}")
    print(f"Using Cores: {args.num_cores}")
    print(f"Using Device: {DEVICE}") # Just to valid setup, though calculation is CPU based

    # 1. Load Ground Truth (Test Set)
    print("\n[Step 1] Loading Test Data (Ground Truth)...")
    try:
        # data_utils.load_test_data() returns (input_paths, labels)
        seq_paths, labels = data_utils.load_test_data()
        
        # Extract User IDs
        gt_map = {} # user_id -> label
        test_user_ids = set()
        
        print("Extracting User IDs from sequence paths...")
        for path, label in zip(seq_paths, labels):
            # Extract ID from filename like 'seq_USERID_DATE.csv'
            # Using data_utils.extract_user_id_from_filename which handles regex
            fname = os.path.basename(path)
            uid = data_utils.extract_user_id_from_filename(fname)
            if uid:
                gt_map[uid] = float(label)
                test_user_ids.add(uid)
        
        print(f"Total Test Users Loaded: {len(test_user_ids)}")
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # 2. Find Raw Files for Test Users
    # Since we don't have a direct map, we scan the raw directory.
    # To optimize, we scan and filter immediately.
    print("\n[Step 2] Scanning Raw Files...")
    raw_files_map = {} # user_id -> file_path
    
    # Using glob to list all files. This might be heavy but is standard.
    # Optimizing: List only directories first? No, structure is group_*/hash.csv
    # We will use glob.iglob for iterator to save memory if list is huge
    
    pattern = os.path.join(DEFAULT_RAW_DIR, "group_*", "*.csv")
    
    # We iterate and check if the file corresponds to a test user
    found_count = 0
    
    # To speed up, we can assume filename == user_id.csv (mostly true based on other scripts)
    # The raw_rulebased_labeling_main.py says: user_id = os.path.basename(file_rel_path).replace(".csv", "")
    
    iterator = glob.iglob(pattern, recursive=True)
    
    # We can parallelize the filtering if needed, but iglob is serial. 
    # Let's just iterate fast.
    for fpath in tqdm(iterator, desc="Scanning files"):
        bname = os.path.basename(fpath)
        # Fast check: is slicing faster?
        # Assuming filename is just the hash + .csv
        u_id = bname[:-4] # remove .csv
        
        if u_id in test_user_ids:
            raw_files_map[u_id] = fpath
            found_count += 1
            
            # Early exit if we found everyone? 
            # Ideally yes, but there might be duplicates or we want to be sure.
            # But duplicate user IDs shouldn't exist in different files usually.
            if found_count >= len(test_user_ids):
                break
    
    print(f"Found {len(raw_files_map)} raw files out of {len(test_user_ids)} test users.")
    missing_users = test_user_ids - set(raw_files_map.keys())
    if missing_users:
        print(f"Warning: {len(missing_users)} users missing from raw data directory.")

    # 3. Process Logic
    print("\n[Step 3] Processing Rules...")
    
    # Prepare tasks
    tasks = []
    for uid, fpath in raw_files_map.items():
        tasks.append((fpath, uid))
        
    results = {}
    
    with multiprocessing.Pool(processes=args.num_cores) as pool:
        # imap_unordered for progress bar
        for uid, score, status in tqdm(pool.imap_unordered(process_single_user, tasks), total=len(tasks), desc="Evaluating"):
            if status == "success":
                results[uid] = score
            else:
                # If error or empty, we default to 0.0 (Not Fraud) but log it?
                # Usually better to predict 0 than fail
                results[uid] = 0.0
                
    # 4. Evaluation
    print("\n[Step 4] Evaluating...")
    
    y_true = []
    y_pred = []
    
    # Align by user_id
    # Only evaluate users we found/processed. 
    # Or should we evaluate all test users and treat missing as 0?
    # Treating missing as 0 is safer for consistent test set size.
    
    for uid in test_user_ids:
        gt = gt_map[uid]
        pred = results.get(uid, 0.0) # Default 0 if missing file or error
        
        y_true.append(gt)
        y_pred.append(pred)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print(f"Eval Set Size: {len(y_true)}")
    print(f"Pred Frauds: {sum(y_pred)}")
    print(f"True Frauds: {sum(y_true)}")
    
    # Use eval_utils
    model_name = "RuleBased_90Refund_1M"
    category = "rulebased"
    
    eval_utils.print_evaluation_report(
        model_name=model_name,
        y_true=y_true,
        y_scores=y_pred,
        threshold=0.5, # Binary logic returns 0 or 1, so 0.5 works
        category=category,
        save=True
    )
    
    print("\nDone.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
