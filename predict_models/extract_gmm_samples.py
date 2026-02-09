"""
GMM-Based Sample Extraction Script

This module extracts representative fraud samples for manual analysis
based on GMM clustering results. Samples are stratified by:
    - Pool A: High-risk users (refund > 2M KRW AND rate >= 90%)
    - Pool B: General risk users
    - Pool C: Non-risk users in Cluster 2 (potential anomalies)

Output:
    CSV file with user_id, group, statistics, and cluster assignments.

Usage:
    python extract_gmm_samples.py --num_cores 32
"""

import os
import glob
import csv
import random
import argparse
import multiprocessing
import shutil
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import torch

# Custom Imports
import data_utils

# Configuration
DEFAULT_RAW_DIR = "/home/srt/Dataset/passenger_split_final"
OUTPUT_DIR = "/home/srt/ml_results/diff_extraction_samples"
SAMPLE_SIZE = 100
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

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

def process_get_stats_from_label(args):
    """
    Worker to read LABEL file and get stats directly.
    args: (label_file_path, user_id)
    Returns: dict with keys: user_id, group, is_risk, cluster, total_ticket_count, total_refund_count, total_refund_amount, refund_rate
    """
    label_path, user_id = args
    group_name = os.path.basename(os.path.dirname(label_path))
    
    try:
        with open(label_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [x.strip() for x in reader.fieldnames]
            
            for row in reader:
                try:
                    is_risk = int(row.get('is_risk', 0))
                    cluster = int(row.get('Cluster', -1))
                    
                    # Columns: 발매승차권수, 반환승차권수, 반환금액, 반환율
                    total_ticket_count = float(row.get('발매승차권수', 0.0))
                    total_refund_count = float(row.get('반환승차권수', 0.0))
                    total_refund_amount = float(row.get('반환금액', 0.0))
                    refund_rate = float(row.get('반환율', 0.0))

                    return {
                        'user_id': user_id,
                        'group': group_name,
                        'is_risk': is_risk,
                        'cluster': cluster,
                        'total_ticket_count': total_ticket_count,
                        'total_refund_count': total_refund_count,
                        'total_refund_amount': total_refund_amount,
                        'refund_rate': refund_rate
                    }
                except ValueError:
                    continue
                
    except Exception as e:
        return None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cores", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    print(f"Using Device: {DEVICE}") 
    print(f"Output Directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Train Users & Collect Data
    print("Loading Train Data...")
    try:
        label_folder = data_utils.TRAIN_LABEL_PATH
        print(f"Collecting file paths from: {label_folder}")
        label_files = data_utils.collect_files_from_groups(label_folder, 'label_')
        
        # Prepare tasks for multiprocessing
        tasks = []
        for uid, path in label_files.items():
            tasks.append((path, uid))
            
        print(f"Total Users Found: {len(tasks)}")
        
        all_data = []
        with multiprocessing.Pool(args.num_cores) as pool:
            for res in tqdm(pool.imap_unordered(process_get_stats_from_label, tasks), total=len(tasks)):
                if res:
                    all_data.append(res)
                    
        print(f"Successfully loaded data for {len(all_data)} users.")
        
    except Exception as e:
        print(f"Error loading train data: {e}")
        return

    # 2. Filtering Pools
    # is_risk 가 true 인것들 중 총 반환 금액이 200만원 이 넘으면서 반환율이 90% 이상인 것들
    pool_A = [x for x in all_data if x['is_risk'] == 1 and x['total_refund_amount'] > 2000000 and x['refund_rate'] >= 0.9]
    
    # is_risk 가 true 인것들 (Pool A 제외 여부는 나중에 처리)
    pool_B = [x for x in all_data if x['is_risk'] == 1]
    
    # is_risk 가 false이면서 Cluster 가 2 인 데이터
    pool_C = [x for x in all_data if x['is_risk'] == 0 and x['cluster'] == 2]

    print(f"Pool A (Risk=T, Amt>2m, Rate>=90%): {len(pool_A)}")
    print(f"Pool B (Risk=T): {len(pool_B)}")
    print(f"Pool C (Risk=F, Cluster=2): {len(pool_C)}")

    # 3. Sampling
    final_samples = []
    
    # 3-1. Select 5 from Pool A
    count_A = 5
    selected_A = []
    if len(pool_A) < count_A:
        print(f"Warning: Pool A has only {len(pool_A)} users. taking all.")
        selected_A = pool_A[:]
    else:
        selected_A = random.sample(pool_A, count_A)
    
    final_samples.extend(selected_A)
    selected_A_ids = set([x['user_id'] for x in selected_A])

    # 3-2. Select 15 from Pool B (excluding already selected in A)
    count_B = 15
    pool_B_candidates = [x for x in pool_B if x['user_id'] not in selected_A_ids]
    selected_B = []
    
    if len(pool_B_candidates) < count_B:
        print(f"Warning: Pool B candidates has only {len(pool_B_candidates)} users. taking all.")
        selected_B = pool_B_candidates[:]
    else:
        selected_B = random.sample(pool_B_candidates, count_B)
        
    final_samples.extend(selected_B)

    # 3-3. Select 10 from Pool C
    count_C = 10
    selected_C = []
    if len(pool_C) < count_C:
        print(f"Warning: Pool C has only {len(pool_C)} users. taking all.")
        selected_C = pool_C[:]
    else:
        selected_C = random.sample(pool_C, count_C)
        
    final_samples.extend(selected_C)
    
    print(f"Total Selected: {len(final_samples)}")
    print(f"  - From A: {len(selected_A)}")
    print(f"  - From B: {len(selected_B)}")
    print(f"  - From C: {len(selected_C)}")

    # 4. Shuffle
    random.shuffle(final_samples)

    # 5. Save
    out_csv = os.path.join(OUTPUT_DIR, "extraction_summary.csv")
    print(f"Saving to {out_csv}...")
    
    # 추출해야 하고 저장해야 하는 칼럼은 user_id, group, 총 발매 건수, 총 반환 건수, 총 반환 액수, 반환율, Cluster
    header = ['user_id', 'group', 'total_ticket_count', 'total_refund_count', 'total_refund_amount', 'refund_rate', 'cluster']
    
    with open(out_csv, "w", encoding="utf-8-sig", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        for item in final_samples:
            row = {k: item[k] for k in header}
            writer.writerow(row)
            
    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
