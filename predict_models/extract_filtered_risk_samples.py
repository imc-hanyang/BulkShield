import os
import csv
import random
import multiprocessing
import argparse
from tqdm import tqdm
import data_utils
import sys

# Configuration
OUTPUT_FILE = "extraction_filtered_summary.csv"
SAMPLE_SIZE = 200

# Columns in Label File:
# 고객관리번호, end_date, 반환율, 반환금액, 반환승차권수, 발매승차권수, Cluster, is_risk, ...

def process_group(group_path):
    """
    Process a single group directory.
    Scans all CSV files in the directory.
    Returns a list of tuples: (user_id, group_name, ticket_count, refund_count, refund_amt, refund_rate)
    meeting the criteria.
    """
    candidates = []
    group_name = os.path.basename(group_path)
    
    try:
        # Get all CSV files in the group directory
        # Using os.scandir for better performance than os.listdir if needed, but listdir is fine for now
        with os.scandir(group_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.startswith('label_') and entry.name.endswith('.csv'):
                    file_path = entry.path
                    
                    try:
                        # Use csv.DictReader for safe parsing
                        with open(file_path, "r", encoding="utf-8-sig") as f:
                            reader = csv.DictReader(f)
                            
                            # Normalize field names (strip whitespace)
                            if reader.fieldnames:
                                reader.fieldnames = [x.strip() for x in reader.fieldnames]
                            
                            for row in reader:
                                # We only process the first row as it contains the user summary
                                
                                # 1. Check is_risk
                                try:
                                    is_risk = int(row.get('is_risk', 0))
                                except ValueError:
                                    is_risk = 0
                                    
                                if is_risk != 1:
                                    break # Skip if not risk
                                
                                # 2. Extract values for filtering
                                try:
                                    refund_amt = float(row.get('반환금액', 0.0))
                                    refund_rate = float(row.get('반환율', 0.0))
                                    ticket_count = float(row.get('발매승차권수', 0.0))
                                    refund_count = float(row.get('반환승차권수', 0.0))
                                    user_id = row.get('고객관리번호', '')
                                except (ValueError, TypeError):
                                    break # Skip on data error
                                    
                                # 3. Apply Exclusion Criteria
                                # Exclude if refund_amt >= 1,000,000 AND refund_rate >= 0.9 (90%)
                                # Note: refund_rate is usually 0.0 to 1.0. If it's percentage, it might be 90. 
                                # Based on previous view_file, it looked like 0.0, 1.0 (float).
                                # Let's assume 0.9 for 90%.
                                if refund_amt >= 1000000 and refund_rate >= 0.9:
                                    break
                                
                                # 4. Add to candidates
                                candidates.append({
                                    'id': user_id,
                                    'group': group_name,
                                    'ticket_count': ticket_count,
                                    'refund_count': refund_count,
                                    'refund_amt': refund_amt,
                                    'refund_rate': refund_rate
                                })
                                break # Stop after first row
                    except Exception:
                        continue # Skip bad file
    except Exception:
        pass # Skip bad group
        
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Extract filtered risk samples.")
    parser.add_argument("--num_cores", type=int, default=100, help="Number of cores to use.")
    args = parser.parse_args()

    # Determine CPU count to use
    num_cores = min(args.num_cores, multiprocessing.cpu_count())
    print(f"Using {num_cores} cores.")

    # 1. Get List of Groups
    label_root = data_utils.TRAIN_LABEL_PATH
    if not os.path.exists(label_root):
        print(f"Error: Label path {label_root} does not exist.")
        return

    print(f"Scanning groups in {label_root}...")
    group_dirs = [os.path.join(label_root, d) for d in os.listdir(label_root) if d.startswith('group_')]
    print(f"Found {len(group_dirs)} groups.")

    # 2. Parallel Process Groups to find Candidates
    print("Scanning files for candidates...")
    all_candidates = []
    
    with multiprocessing.Pool(num_cores) as pool:
        # Use imap_unordered for better responsiveness in tqdm
        for group_candidates in tqdm(pool.imap_unordered(process_group, group_dirs), total=len(group_dirs)):
            if group_candidates:
                all_candidates.extend(group_candidates)

    print(f"Total candidates found: {len(all_candidates)}")

    # 3. Sample 200 Users
    if len(all_candidates) < SAMPLE_SIZE:
        print(f"Warning: Only found {len(all_candidates)} candidates! Taking all.")
        sampled_data = all_candidates
    else:
        print(f"Sampling {SAMPLE_SIZE} random users...")
        sampled_data = random.sample(all_candidates, SAMPLE_SIZE)

    # 4. Save to CSV
    # Output columns: id, group, ticket_count, refund_count, refund_amt, refund_rate
    print(f"Saving to {OUTPUT_FILE}...")
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            fieldnames = ['id', 'group', 'ticket_count', 'refund_count', 'refund_amt', 'refund_rate']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_data)
        print("Done.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
