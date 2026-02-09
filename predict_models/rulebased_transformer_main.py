"""
Transformer Training with Rule-Based Labels

This module trains the SRTTransformerClassifier using rule-based fraud labels
as ground truth. The Transformer architecture uses categorical embeddings,
numeric features, and time delta encoding.

Features:
    - Mixed precision training (AMP) for H100 efficiency
    - Dynamic vocabulary building from training subset
    - Best model checkpointing based on Macro F1 score

Data Sources:
    - Labels: /home/srt/Dataset/label/rulebased_label_*
    - Sequences: /home/srt/Dataset/feature/sequence_data_28d_*

Usage:
    python rulebased_transformer_main.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys
import traceback
import os
import csv
import math
import datetime
import json
import logging
from tqdm import tqdm
import copy

# Import Model
from models.transformer_model import SRTTransformerClassifier, CFG, CATEGORICAL_COLS, NUMERIC_COLS

# Import Utils
from eval_utils import print_evaluation_report, get_run_dir
from sklearn.metrics import f1_score

# =========================================================
# 1. Setup & Config
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
]

def fast_parse_ts(ts_str: str):
    if not ts_str: return None
    for fmt in TS_FORMATS:
        try:
            return datetime.datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None

def safe_int(x, default=-1):
    try:
        if not x: return default
        return int(float(x))
    except:
        return default

def safe_float(x):
    try:
        if not x: return 0.0
        return float(x)
    except:
        return 0.0

# =========================================================
# 2. Dataset
# =========================================================
class TransformerDataset(Dataset):
    def __init__(self, input_paths, labels, vocabs, cfg: CFG):
        self.input_paths = input_paths
        self.labels = labels
        self.vocabs = vocabs
        self.cfg = cfg
        self.max_len = cfg.max_len

    def __len__(self):
        return len(self.input_paths)

    def _enc_cat(self, col: str, v: int) -> int:
        # Default to 0 (Unknown/Padding) if not found
        return self.vocabs.get(col, {}).get(v, 0)

    def __getitem__(self, idx):
        seq_path = self.input_paths[idx]
        label = self.labels[idx]

        data_rows = []
        try:
            # csv.DictReader for memory efficiency
            with open(seq_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # Parse timestamp for delta calculation
                    ts_val = None
                    if "timestamp" in r:
                        ts_val = fast_parse_ts(r["timestamp"])

                    item = {"_ts": ts_val}
                    
                    # Store needed columns
                    for c in CATEGORICAL_COLS:
                        item[c] = r.get(c, "")
                    for c in NUMERIC_COLS:
                        item[c] = r.get(c, "")
                    
                    data_rows.append(item)
        except Exception as e:
            # logger.warning(f"Error reading {seq_path}: {e}")
            pass

        if not data_rows:
            # Return empty padded tensor
            T = 1
            cat = torch.zeros(T, len(CATEGORICAL_COLS), dtype=torch.long)
            num = torch.zeros(T, len(NUMERIC_COLS), dtype=torch.float32)
            delta = torch.zeros(T, dtype=torch.long)
            return {
                "cat": cat, "num": num, "delta": delta,
                "y": torch.tensor(label, dtype=torch.long)
            }

        # Sort by timestamp (if available)
        max_ts = datetime.datetime.max
        data_rows.sort(key=lambda x: x["_ts"] if x["_ts"] else max_ts)

        # Truncate if too long (Keep latest)
        if len(data_rows) > self.max_len:
            data_rows = data_rows[-self.max_len:]
            
        T = len(data_rows)
        cat_feats = []
        num_feats = []
        timestamps = []

        for r in data_rows:
            # Categorical Encoding
            c_vec = []
            for c in CATEGORICAL_COLS:
                val = safe_int(r[c], default=-1)
                c_vec.append(self._enc_cat(c, val))
            cat_feats.append(c_vec)
            
            # Numeric Scaling (Log1p)
            n_vec = []
            for c in NUMERIC_COLS:
                raw_val = safe_float(r[c])
                val = math.log1p(abs(raw_val))
                if raw_val < 0: val = -val
                n_vec.append(val)
            num_feats.append(n_vec)
            
            timestamps.append(r["_ts"])

        cat = torch.tensor(cat_feats, dtype=torch.long)
        num = torch.tensor(num_feats, dtype=torch.float32)

        # Delta Calculation
        delta_vals = [0] * T
        if T > 1 and any(timestamps):
            delta_vals[0] = 0
            for i in range(1, T):
                t_curr = timestamps[i]
                t_prev = timestamps[i-1]
                if t_curr and t_prev:
                    diff_min = (t_curr - t_prev).total_seconds() / 60.0
                else:
                    diff_min = 0.0
                
                # Clip and Bucket
                diff_min = max(0.0, min(diff_min, 60.0 * 48))
                b = int(diff_min // self.cfg.delta_bucket_size_min)
                b = max(0, min(b, self.cfg.delta_max_bucket))
                delta_vals[i] = b
        
        delta = torch.tensor(delta_vals, dtype=torch.long)

        return {
            "cat": cat,
            "num": num,
            "delta": delta,
            "y": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    # Determine max sequence length in this batch
    maxT = max(x["cat"].shape[0] for x in batch)
    n_cat = batch[0]["cat"].shape[1]
    n_num = batch[0]["num"].shape[1]

    # Initialize padded tensors
    cat_pad = torch.zeros(len(batch), maxT, n_cat, dtype=torch.long)
    num_pad = torch.zeros(len(batch), maxT, n_num, dtype=torch.float32)
    delta_pad = torch.zeros(len(batch), maxT, dtype=torch.long)
    pad_mask = torch.ones(len(batch), maxT, dtype=torch.bool) # True = padding

    ys = []
    
    for i, x in enumerate(batch):
        T = x["cat"].shape[0]
        cat_pad[i, :T] = x["cat"]
        num_pad[i, :T] = x["num"]
        delta_pad[i, :T] = x["delta"]
        pad_mask[i, :T] = False
        ys.append(x["y"])

    y = torch.stack(ys) # [B]
    
    return {
        "cat": cat_pad, 
        "num": num_pad, 
        "delta": delta_pad, 
        "pad_mask": pad_mask, 
        "y": y
    }

# =========================================================
# 3. Rule-Based Data Loading
# =========================================================
def collect_files(directory, prefix):
    """Collects files recursively from group folders."""
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
                    # filename: prefix_USERID_DATE.csv
                    # Extract UserID Robustly
                    rest = f[len(prefix):]
                    parts = rest.split('_')
                    user_id = parts[0]
                    file_map[user_id] = os.path.join(g_path, f)
        except:
            continue
    return file_map

def load_rulebased_data(label_dir, seq_dir, split_name="Train"):
    logger.info(f"[{split_name}] Mapping Data Files from {label_dir}...")
    
    # 1. Map Files
    label_files = collect_files(label_dir, 'label_')
    seq_files = collect_files(seq_dir, 'seq_')
    
    # 2. Find Intersection
    common_users = sorted(list(set(label_files.keys()) & set(seq_files.keys())))
    logger.info(f"[{split_name}] Matched Users: {len(common_users):,}")
    
    if len(common_users) == 0:
        return [], []

    input_paths = []
    labels = []
    
    # 3. Read Labels (Streaming)
    logger.info(f"[{split_name}] Reading Label CSVs...")
    
    # Use tqdm to show progress (write to stdout to avoid log clutter)
    for user_id in tqdm(common_users, ncols=100, mininterval=1.0, desc=f"Loading {split_name}"):
        lbl_path = label_files[user_id]
        try:
            with open(lbl_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check for 'is_risk' or 'is_fraud'
                    val = 0
                    if 'is_risk' in row:
                        val = int(float(row['is_risk']))
                    elif 'is_fraud' in row:
                         val = int(float(row['is_fraud']))
                    
                    labels.append(val)
                    input_paths.append(seq_files[user_id])
                    break # Read one row per user
        except:
            continue
            
    return input_paths, labels

# =========================================================
# 4. Vocab Building
# =========================================================
def build_vocabs_on_the_fly(paths, limit=5000):
    logger.info("Building vocabulary from subset...")
    vocabs = {c: {} for c in CATEGORICAL_COLS}
    
    # Random sample if possible, or just first N
    sample_paths = paths[:limit] if len(paths) > limit else paths
    
    logger.info(f"Vocab Building: Processing {len(sample_paths)} files...")
    
    for i, p in enumerate(sample_paths):
        if i % 1000 == 0 and i > 0:
             logger.info(f"  Vocab Building: {i}/{len(sample_paths)}")
        try:
            with open(p, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    for c in CATEGORICAL_COLS:
                        v_str = r.get(c, "")
                        v = safe_int(v_str, default=-1)
                        if v not in vocabs[c]:
                            # Assign new ID (1-based)
                            vocabs[c][v] = len(vocabs[c]) + 1
        except:
            pass
            
    logger.info("Vocab building complete.")
    for c in CATEGORICAL_COLS:
        logger.info(f"  {c}: {len(vocabs[c])} tokens")
        
    return vocabs

# =========================================================
# 5. Training / Eval Function
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=True)
    for batch in pbar:
        cat = batch["cat"].to(device, non_blocking=True)
        num = batch["num"].to(device, non_blocking=True)
        delta = batch["delta"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision
        with torch.cuda.amp.autocast(enabled=model.cfg.use_amp):
            logits = model(cat, num, delta, pad_mask) # [B, 2]
            loss = criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_probs = [] # Probabilities of class 1
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate"):
            cat = batch["cat"].to(device, non_blocking=True)
            num = batch["num"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=model.cfg.use_amp):
                logits = model(cat, num, delta, pad_mask)
                probs = torch.softmax(logits, dim=1)[:, 1] # Probability of Fraud
                
            all_y.extend(batch["y"].tolist())
            all_probs.extend(probs.cpu().tolist())
            
    return all_y, all_probs

# =========================================================
# 6. Main
# =========================================================
def main():
    # Hardware
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    
    cfg = CFG() # H100 optimized config
    
    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Rule-Based Paths
    TRAIN_LABEL_DIR = '/home/srt/Dataset/label/rulebased_label_train_per_user'
    TEST_LABEL_DIR = '/home/srt/Dataset/label/rulebased_label_test_per_user'
    TRAIN_SEQ_DIR = '/home/srt/Dataset/feature/sequence_data_28d_train'
    TEST_SEQ_DIR = '/home/srt/Dataset/feature/sequence_data_28d_test'
    
    # 1. Load Data
    logger.info("Loading Rule-Based Data...")
    train_paths, train_labels = load_rulebased_data(TRAIN_LABEL_DIR, TRAIN_SEQ_DIR, "Train")
    test_paths, test_labels = load_rulebased_data(TEST_LABEL_DIR, TEST_SEQ_DIR, "Test")
    
    if not train_paths:
        logger.error("No training data found.")
        return

    # 2. Build Vocabulary (Simplify: build from train subset)
    vocabs = build_vocabs_on_the_fly(train_paths, limit=20000) 

    # 3. Dataset & DataLoader
    num_workers = 32
    
    train_ds = TransformerDataset(train_paths, train_labels, vocabs, cfg)
    val_ds = TransformerDataset(test_paths, test_labels, vocabs, cfg)
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn, 
        pin_memory=cfg.pin_memory, persistent_workers=cfg.persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=cfg.pin_memory, persistent_workers=cfg.persistent_workers
    )
    
    # 4. Model
    model = SRTTransformerClassifier(vocabs, len(NUMERIC_COLS), cfg).to(device)
    logger.info(f"Model initialized. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    best_f1 = -1.0
    best_model_state = None
    
    # 5. Training Loop
    logger.info("Starting Training...")
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"Epoch {epoch} | Train Loss: {loss:.6f}")
        
        # Validation
        y_true, y_scores = evaluate(model, val_loader, device)
        
        # Evaluate using Utils
        # Using category 'rulebased_dl' for saving results
        metrics = print_evaluation_report(
            "rulebased_transformer", y_true, y_scores, threshold=0.5, 
            category='rulebased_transformer', save=False, timestamp=timestamp
        )
        
        # Custom Metric: Macro F1
        y_pred = (np.array(y_scores) >= 0.5).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        logger.info(f"Epoch {epoch} | Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"🔥 New Best Macro F1: {best_f1:.4f}")
            
            # Immediate Save
            # Category 'rulebased_dl'
            save_base_dir = "/home/srt/ml_results/rulebased_transformer"
            save_dir = os.path.join(save_base_dir, timestamp)
            os.makedirs(save_dir, exist_ok=True)
            
            # Symlink latest
            try:
                latest_link = os.path.join(save_base_dir, "latest")
                if os.path.exists(latest_link): os.unlink(latest_link)
                os.symlink(timestamp, latest_link)
            except: pass
            
            torch.save(model.state_dict(), os.path.join(save_dir, "transformer_best.pth"))
            with open(os.path.join(save_dir, "vocabs.json"), "w") as f:
                json.dump(vocabs, f, indent=4)
            logger.info(f"  -> Saved model & vocabs to {save_dir}")
            
    # 6. Final Evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        y_true, y_scores = evaluate(model, val_loader, device)
        print_evaluation_report(
            "rulebased_transformer", y_true, y_scores, threshold=0.5, 
            category='rulebased_transformer', save=True, timestamp=timestamp
        )
        logger.info("Final evaluation on best model completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error(traceback.format_exc())
