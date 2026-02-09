"""
Raw Rule-Based Labeling Script

This module generates per-user fraud labels based on rule-based criteria
by processing raw passenger CSV files. Labels are computed using:
    - 1-month sliding window from each user's last departure date
    - Threshold logic: refund_amount >= 1M KRW AND refund_rate >= 90%

Output:
    Per-user label CSV files in train/test subdirectories, containing:
    user_id, end_date, is_risk, refund_amt, refund_rate

Usage:
    python raw_rulebased_labeling_main.py --num_cores 95 --chunksize 500
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool
from dateutil.relativedelta import relativedelta
from datetime import datetime
from tqdm import tqdm
import sys
import gc
import csv

# Config
DEFAULT_RAW_DIR = "/home/srt/Dataset/passenger_split_final"
TRAIN_MANIFEST = "/home/srt/Dataset/manifest_train.csv"
TEST_MANIFEST = "/home/srt/Dataset/manifest_test.csv"

OUT_TRAIN_DIR = "/home/srt/Dataset/label/rulebased_label_train_per_user"
OUT_TEST_DIR = "/home/srt/Dataset/label/rulebased_label_test_per_user"

# Thresholds
REFUND_THRESHOLD = 1_000_000
REFUND_RATE_THRESHOLD = 0.90

# Global Vars
G_RAW_DIR = None
G_TRAIN_ID_SET = None
G_TEST_ID_SET = None
G_OUT_TRAIN = None
G_OUT_TEST = None

def parse_dot_date(date_str):
    try:
        if pd.isna(date_str): return pd.NaT
        return pd.to_datetime(str(date_str).strip(), format="%Y.%m.%d")
    except:
        return pd.NaT

def worker_init(raw_dir, train_ids, test_ids, out_train, out_test):
    global G_RAW_DIR, G_TRAIN_ID_SET, G_TEST_ID_SET, G_OUT_TRAIN, G_OUT_TEST
    G_RAW_DIR = raw_dir
    G_TRAIN_ID_SET = set(train_ids)
    G_TEST_ID_SET = set(test_ids)
    G_OUT_TRAIN = out_train
    G_OUT_TEST = out_test

def process_user_file(file_rel_path):
    try:
        user_id = os.path.basename(file_rel_path).replace(".csv", "")
        
        # Split Check
        # 우선순위: Train > Test (중복 시 Train으로 간주하거나, 둘 다? 보통 분리됨)
        target_dir = None
        split_name = ""
        
        if user_id in G_TRAIN_ID_SET:
            target_dir = G_OUT_TRAIN
            split_name = "train"
        elif user_id in G_TEST_ID_SET:
            target_dir = G_OUT_TEST
            split_name = "test"
        else:
            return ("skipped_unknown_split", None)
            
        # File Read (Use csv module to avoid Pandas SegFault/Hang)
        file_path = os.path.join(G_RAW_DIR, file_rel_path)
        
        rows = []
        try:
            # csv.DictReader 사용 (encoding 주의: utf-8-sig or cp949? Raw Data는 보통 utf-8로 보였음)
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                
                # 헤더 공백 제거
                if reader.fieldnames:
                    reader.fieldnames = [x.strip() for x in reader.fieldnames]

                # 필수 컬럼 확인
                if '출발일자' not in reader.fieldnames or '반환금액' not in reader.fieldnames:
                     return ("error_missing_col", None)
                
                for row in reader:
                    # 필요한 컬럼만 추출
                    rows.append({
                        '출발일자': row.get('출발일자'),
                        '반환금액': row.get('반환금액'),
                        '고객관리번호': row.get('고객관리번호', user_id) 
                    })
        except Exception as e:
            return ("error_read", {"msg": str(e)})

        df = pd.DataFrame(rows)
        
        # Date Parse
        df['출발일자_dt'] = df['출발일자'].apply(parse_dot_date)
        df = df.dropna(subset=['출발일자_dt'])
        
        if df.empty:
            return ("skipped_empty_date", None)
            
        # Window (Max Date)
        end_date = df['출발일자_dt'].max()
        start_date = end_date - relativedelta(months=1) + pd.Timedelta(days=1)
        
        # Filter
        mask = (df['출발일자_dt'] >= start_date) & (df['출발일자_dt'] <= end_date)
        window_df = df[mask].copy()
        
        if window_df.empty:
            return ("skipped_no_window", None)
            
        # Stats
        # 반환금액 숫자 변환
        window_df['반환금액'] = pd.to_numeric(window_df['반환금액'], errors='coerce').fillna(0)
        
        total_issued = len(window_df)
        total_refund_amt = window_df['반환금액'].sum()
        refund_cnt = (window_df['반환금액'] > 0).sum()
        refund_rate = refund_cnt / total_issued if total_issued > 0 else 0.0
        
        # Labeling
        is_risk = (total_refund_amt >= REFUND_THRESHOLD) and (refund_rate >= REFUND_RATE_THRESHOLD)
        
        # Save
        end_date_str = end_date.strftime("%Y%m%d")
        group_folder = os.path.dirname(file_rel_path)
        
        out_group_dir = os.path.join(target_dir, group_folder)
        os.makedirs(out_group_dir, exist_ok=True)
        
        out_path = os.path.join(out_group_dir, f"label_{user_id}_{end_date_str}.csv")
        
        result = {
            '고객관리번호': user_id,
            'end_date': end_date,
            'is_risk': int(is_risk),
            'refund_amt': total_refund_amt,
            'refund_rate': refund_rate
        }
        
        pd.DataFrame([result]).to_csv(out_path, index=False, encoding="utf-8-sig")
        
        return ("saved", {"split": split_name, "is_risk": int(is_risk)})
        
    except Exception as e:
        return ("error", {"msg": str(e)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_cores", type=int, default=95) # H100 Server Optimized
    ap.add_argument("--chunksize", type=int, default=500)
    args = ap.parse_args()
    
    # 1. Load User Manifests
    print("Loading Manifests...")
    try:
        train_users = set(pd.read_csv(TRAIN_MANIFEST)['user_id'].astype(str))
        test_users = set(pd.read_csv(TEST_MANIFEST)['user_id'].astype(str))
        print(f"Train Users: {len(train_users):,}")
        print(f"Test Users: {len(test_users):,}")
    except Exception as e:
        print(f"Error loading manifests: {e}")
        return

    # 2. List Raw Files
    print(f"Listing Raw Files in {DEFAULT_RAW_DIR}...")
    files = glob.glob(os.path.join(DEFAULT_RAW_DIR, "group_*", "*.csv"))
    # 상대경로 변환
    rel_files = [os.path.relpath(f, DEFAULT_RAW_DIR) for f in files]
    print(f"Total Raw Files: {len(rel_files):,}")
    
    # 3. Processing
    print(f"Starting Processing with {args.num_cores} cores...")
    
    stats = {"train_saved": 0, "test_saved": 0, "skipped": 0, "error": 0, "train_risk": 0, "test_risk": 0}
    
    # Init Args: lists are pickled, so convert sets to list if needed, but 'set' is picklable.
    # Note: sharing large sets via initargs works because of fork (subprocess memory copy).
    
    with Pool(processes=args.num_cores, initializer=worker_init, 
              initargs=(DEFAULT_RAW_DIR, train_users, test_users, OUT_TRAIN_DIR, OUT_TEST_DIR)) as pool:
        
        it = pool.imap_unordered(process_user_file, rel_files, chunksize=args.chunksize)
        
        for status, payload in tqdm(it, total=len(rel_files), ncols=120):
            if status == "saved":
                split = payload['split']
                stats[f"{split}_saved"] += 1
                if payload['is_risk'] == 1:
                    stats[f"{split}_risk"] += 1
            elif status.startswith("skipped"):
                stats["skipped"] += 1
            else:
                stats["error"] += 1
                
    print("\n✅ Labeling Complete!")
    print(f"Train Saved: {stats['train_saved']:,} (Risk: {stats['train_risk']:,})")
    print(f"Test Saved: {stats['test_saved']:,} (Risk: {stats['test_risk']:,})")
    print(f"Skipped (Unknown Split / Empty / No Window): {stats['skipped']:,}")
    print(f"Errors: {stats['error']:,}")

if __name__ == "__main__":
    # Force minimal memory alloc for workers if possible (not much control in Py)
    main()
