
import os
import csv
import sys
import shutil
import random
import time
import logging
from multiprocessing import Pool, Manager, cpu_count
from datetime import datetime
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
class Config:
    # Source Directories
    GMM_TRAIN_DIR = '/home/srt/Dataset/label/label_train_per_user'
    GMM_TEST_DIR = '/home/srt/Dataset/label/label_test_per_user'
    
    # Destination
    OUTPUT_DIR = '/home/srt/ml_results/diff_extraction_samples'
    
    # Parameters
    TARGET_SAMPLE_SIZE = 100
    NUM_WORKERS = 100  
    RANDOM_SEED = 42
    
    LOG_FILE = f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Extractor")


# ==========================================
# Worker Function (Must be top-level)
# ==========================================
def scan_worker(file_path):
    """
    Worker function to process a single CSV file.
    Args:
        file_path: str
    Returns:
        tuple (user_id, is_fraud, refund_rate, refund_amount, group_name) or None
    """
    try:
        user_id = None
        is_fraud = False
        refund_rate = 0.0
        refund_amount = 0.0
        
        # 1. Extract Group Name
        # file_path: .../group_XX/filename.csv
        group_name = os.path.basename(os.path.dirname(file_path))
        
        # 2. Extract UserID from filename (robust split)
        basename = os.path.basename(file_path)
        
        # If prefix is "label_", remove it.
        if basename.startswith('label_'):
            temp = basename[6:] # remove 'label_'
            parts = temp.split('_')
            user_id = parts[0]
        else:
            parts = basename.split('_')
            if parts[0].isalpha():
                user_id = parts[1]
            else:
                user_id = parts[0]
                
        # 3. Read Content
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            # Normalize header
            if reader.fieldnames:
                reader.fieldnames = [x.strip() for x in reader.fieldnames]

            for row in reader:
                # Check Fraud Label
                val = 0
                if 'label' in row: val = float(row['label'])
                elif 'is_fraud' in row: val = float(row['is_fraud'])
                elif 'is_risk' in row: val = float(row['is_risk'])
                
                if int(val) == 1:
                    is_fraud = True
                    
                # Extract Refund Metrics
                if '반환율' in row:
                    refund_rate = float(row['반환율'])
                if '반환금액' in row:
                    refund_amount = float(row['반환금액'])
                    
                break # One row check sufficient for per-user file
                
        if user_id:
            return (user_id, is_fraud, refund_rate, refund_amount, group_name)
            
    except Exception as e:
        return None
        
    return None

def collect_file_paths(directory):
    """Recursively collect all .csv files from directory"""
    paths = []
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return paths
        
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.csv'):
                paths.append(os.path.join(root, f))
    return paths

# ==========================================
# Main Orchestrator
# ==========================================
def main():
    start_time = time.time()
    random.seed(Config.RANDOM_SEED)
    
    logger.info(f"🚀 Starting Extraction Task (GMM Fraud Only)")
    logger.info(f"   - Cores: {Config.NUM_WORKERS}")
    logger.info(f"   - Target: {Config.TARGET_SAMPLE_SIZE} samples")
    
    # 1. Collect All File Paths
    logger.info("📂 Collecting file paths...")
    
    gmm_files = collect_file_paths(Config.GMM_TRAIN_DIR) + collect_file_paths(Config.GMM_TEST_DIR)
    logger.info(f"   - GMM Files found: {len(gmm_files):,}")
    
    # 2. Scanning Files (Parallel)
    logger.info("🔍 Scanning GMM Labels for Fraud...")
    
    gmm_fraud_users = []
    gmm_user_data = {} # Map[uid] -> {rate, amt, group}
    
    with Pool(processes=Config.NUM_WORKERS) as pool:
        # GMM Scan
        results = pool.imap_unordered(scan_worker, gmm_files, chunksize=100)
        
        for res in tqdm(results, total=len(gmm_files), desc="Scanning"):
            if res:
                uid, is_fraud, rate, amt, group = res
                if is_fraud:
                    gmm_fraud_users.append(uid)
                    gmm_user_data[uid] = {'rate': rate, 'amt': amt, 'group': group}
                    
    # Remove duplicates
    gmm_fraud_users = list(set(gmm_fraud_users))
    logger.info(f"✅ Unique GMM Fraud Users: {len(gmm_fraud_users):,}")
    
    # 3. Random Sampling
    if len(gmm_fraud_users) < Config.TARGET_SAMPLE_SIZE:
        logger.warning(f"⚠️ Warning: Available users ({len(gmm_fraud_users)}) < Target ({Config.TARGET_SAMPLE_SIZE}). Extracting all.")
        sample_users = gmm_fraud_users
    else:
        sample_users = random.sample(gmm_fraud_users, Config.TARGET_SAMPLE_SIZE)
        
    logger.info(f"🎯 Selected {len(sample_users)} users for extraction.")
    
    # 4. Save Summary CSV
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(Config.OUTPUT_DIR, "extraction_summary.csv")
    
    logger.info(f"💾 Saving summary to: {output_path}")
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # Header: id, group, refund_rate_1month, refund_area_1month
            # Mapped to English: id, group, refund_rate_1_month, refund_amount_1_month
            fieldnames = ['id', 'group', 'refund_rate_1_month', 'refund_amount_1_month']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for uid in sample_users:
                data = gmm_user_data.get(uid)
                if data:
                    writer.writerow({
                        'id': uid,
                        'group': data['group'],
                        'refund_rate_1_month': data['rate'],
                        'refund_amount_1_month': data['amt']
                    })
                
        logger.info("✅ CSV Save Successful.")
        
    except Exception as e:
        logger.error(f"❌ Failed to save CSV: {e}")
            
    logger.info(f"{'='*50}")
    logger.info(f"✅ Extraction Complete.")
    logger.info(f"   - Total Extracted: {len(sample_users)}")
    logger.info(f"   - Output File: {output_path}")
    logger.info(f"   - Execution Time: {time.time() - start_time:.2f} seconds")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
