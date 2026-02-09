import os
import sys
import csv
import glob
import time
import math
import torch
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from tqdm import tqdm

# Configuration
TRAIN_DIR = "/home/srt/Dataset/feature/sequence_data_28d_causal_by_day_train"
TEST_DIR = "/home/srt/Dataset/feature/sequence_data_28d_causal_by_day_test"
OUTPUT_FILE = "stats_result.txt"
LOG_INTERVAL = 5  # Log every 5 seconds

# Features to process
FEATURES = [
    "dep_station_id",
    "arr_station_id",
    "route_id",
    "train_id",
    "action_type",
    "seat_cnt",
    "buy_amt",
    "refund_amt",
    "cancel_fee",
    "route_dist_km",
    "travel_time",
    "lead_time_buy",
    "lead_time_ref",
    "hold_time",
    "dep_dow",
    "dep_hour",
    "route_buy_cnt",
    "fwd_dep_hour_median",
    "fwd_dep_dow_median",
    "rev_buy_cnt",
    "rev_ratio",
    "unique_route_cnt",
    "rev_dep_hour_median",
    "rev_dep_dow_median",
    "rev_return_gap",
    "overlap_cnt",
    "same_route_cnt",
    "rev_route_cnt",
    "repeat_interval",
    "adj_seat_refund_flag",
    "recent_ref_cnt",
    "recent_ref_amt",
    "recent_ref_rate"
]

def process_file_batch(file_list):
    """
    Process a list of files and return local statistics (count, sum, sum_sq).
    """
    local_count = 0
    # Initialize sums as float64 numpy arrays to valid overflow/precision issues on CPU
    local_sum = np.zeros(len(FEATURES), dtype=np.float64)
    local_sum_sq = np.zeros(len(FEATURES), dtype=np.float64)
    
    # Pre-compute index mapping for speed
    feature_indices = {name: i for i, name in enumerate(FEATURES)}
    num_features = len(FEATURES)
    
    for string_path in file_list:
        try:
            with open(string_path, 'r', encoding='utf-8') as f:
                # Using DictReader as requested
                reader = csv.DictReader(f)
                
                # Iterate rows
                for row in reader:
                    local_count += 1
                    
                    # Extract values
                    # Optimization: Use list comprehension or direct loop
                    # We assume data is well-formed, but handle missing/empty as 0 or skip?
                    # Usually for mean/std, missing should be handled carefully. 
                    # Assuming 0 for missing or strict float conversion.
                    # float() handles "123.45" well.
                    
                    # Vectorized extraction is harder with DictReader row-by-row, 
                    # so we just loop.
                    vals = []
                    for name in FEATURES:
                        val_str = row.get(name, '0')
                        if not val_str:
                            val_str = '0'
                        try:
                            vals.append(float(val_str))
                        except ValueError:
                            vals.append(0.0)
                            
                    vals_arr = np.array(vals, dtype=np.float64)
                    local_sum += vals_arr
                    local_sum_sq += (vals_arr ** 2)
                    
        except Exception as e:
            print(f"Error reading {string_path}: {e}")
            continue
            
    return local_count, local_sum, local_sum_sq

def main():
    # Force line buffering for stdout so logs appear immediately in nohup.out
    sys.stdout.reconfigure(line_buffering=True)
    start_time = time.time()
    
    # 1. Gather all files
    print("Scanning for files...", flush=True)
    files = []
    
    # Glob allows recursive search with **
    train_pattern = os.path.join(TRAIN_DIR, "**", "*.csv")
    test_pattern = os.path.join(TEST_DIR, "**", "*.csv")
    
    # We use glob.iglob for iterator but list is needed for chunking
    # os.walk might be safer if glob is too slow or misses hidden files? 
    # But recursive glob is standard.
    # Note: python 3.10+ glob has root_dir, but we are generic.
    
    files.extend(glob.glob(train_pattern, recursive=True))
    files.extend(glob.glob(test_pattern, recursive=True))
    
    total_files = len(files)
    print(f"Found {total_files} files.", flush=True)
    
    if total_files == 0:
        print("No files found. Exiting.")
        return

    # 2. Set up Multiprocessing
    # Use substantially more cores if available to saturate I/O and CPU parsing
    num_processes = 90 # Leave a few cores for system/main
    
    # Chunk files - smaller batches for frequent updates
    # 20M files / 4000 chunks ~ 5k files per chunk.
    # This allows 90 workers to pick up tasks frequently.
    chunk_size = 5000
    # Use generator if list is too large, but list slicing is fast enough for 20M items typically (pointers).
    chunks = [files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    print(f"Starting processing with {num_processes} processes on {len(chunks)} chunks (approx {chunk_size} files/chunk)...", flush=True)
    
    # Global accumulators (on GPU eventually, but intermediate on CPU)
    # We aggregate on CPU first then move to GPU for final math to satisfy requirement
    # (Calculating mean/std is trivial, but we strictly follow 'Use cuda:0')
    
    total_count = 0
    total_sum = np.zeros(len(FEATURES), dtype=np.float64)
    total_sum_sq = np.zeros(len(FEATURES), dtype=np.float64)
    
    processed_files = 0
    
    with Pool(processes=num_processes) as pool:
        # We use imap_unordered to process as they finish
        # result_iter = pool.imap_unordered(process_file_batch, chunks)
        
        # Use tqdm for progress bar
        # Since we are using buffering, tqdm output might need file=sys.stdout to be captured properly if stderr is buffering differs,
        # but 2>&1 handles it.
        # We iterate over the imap_unordered result with tqdm wrapper.
        
        for count, s, s_sq in tqdm(pool.imap_unordered(process_file_batch, chunks), total=len(chunks), mininterval=1.0):
            total_count += count
            total_sum += s
            total_sum_sq += s_sq
            
            # processed_files += len(chunks[0]) # Approximate 
    
    print(f"Processing complete. Total rows: {total_count}", flush=True)
    
    if total_count == 0:
        print("No data rows processed.")
        return

    # 3. Final Calculation on GPU
    # "Use cuda:0"
    print("Performing final calculation on CUDA:0...", flush=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Move accumulators to GPU
    t_sum = torch.tensor(total_sum, device=device, dtype=torch.float64)
    t_sum_sq = torch.tensor(total_sum_sq, device=device, dtype=torch.float64)
    t_count = torch.tensor(total_count, device=device, dtype=torch.float64)
    
    # Calculate Mean
    mean = t_sum / t_count
    
    # Calculate Std = sqrt( E[x^2] - (E[x])^2 )
    # Var = sum_sq/N - mean^2
    variance = (t_sum_sq / t_count) - (mean ** 2)
    # Handle precision issues (negative variance close to 0)
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance)
    
    # Move back to CPU for printing
    mean_cpu = mean.cpu().numpy()
    std_cpu = std.cpu().numpy()
    
    # 4. Output Results
    print("-" * 60)
    print(f"{'Feature':<30} | {'Mean':<15} | {'Std':<15}")
    print("-" * 60)
    
    with open(OUTPUT_FILE, 'w') as f:
        header = f"{'Feature':<30} | {'Mean':<15} | {'Std':<15}\n"
        f.write(header)
        print(header, end="")
        
        for i, feature in enumerate(FEATURES):
            line = f"{feature:<30} | {mean_cpu[i]:<15.4f} | {std_cpu[i]:<15.4f}\n"
            f.write(line)
            print(line, end="")
            
    print("-" * 60)
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
