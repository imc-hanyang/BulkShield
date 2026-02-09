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
import pdb
import time

# Import Renamed Model
from models.transformer_without_added_labeling_model import SRTTransformerWithoutAddedLabeling, CFG, CATEGORICAL_COLS, NUMERIC_COLS

# Import Utils
from eval_utils import print_evaluation_report, get_run_dir
from data_utils import load_train_data, load_test_data
from sklearn.metrics import f1_score

# =========================================================
# 1. Setup & Config
# =========================================================
# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("transformer_without_added_labeling_main.log"),
        logging.StreamHandler(sys.stdout)
    ]
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
class TransformerDatasetV2(Dataset):
    def __init__(self, input_paths, labels, vocabs, cfg: CFG):
        self.input_paths = input_paths
        self.labels = labels
        self.vocabs = vocabs
        self.cfg = cfg
        self.max_len = cfg.max_len

    def __len__(self):
        return len(self.input_paths)

    def _enc_cat(self, col: str, v: int) -> int:
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
                    # Parse timestamp strictly for sorting, not for feature delta
                    ts_val = None
                    if "timestamp" in r:
                        ts_val = fast_parse_ts(r["timestamp"])

                    item = {"_ts": ts_val}
                    
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
            return {
                "cat": cat, "num": num,
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
            
        cat = torch.tensor(cat_feats, dtype=torch.long)
        num = torch.tensor(num_feats, dtype=torch.float32)

        return {
            "cat": cat,
            "num": num,
            "y": torch.tensor(label, dtype=torch.long)
        }

def collate_fn_v2(batch):
    # Determine max sequence length in this batch
    maxT = max(x["cat"].shape[0] for x in batch)
    n_cat = batch[0]["cat"].shape[1]
    n_num = batch[0]["num"].shape[1]

    # Initialize padded tensors
    cat_pad = torch.zeros(len(batch), maxT, n_cat, dtype=torch.long)
    num_pad = torch.zeros(len(batch), maxT, n_num, dtype=torch.float32)
    pad_mask = torch.ones(len(batch), maxT, dtype=torch.bool) # True = padding

    ys = []
    
    for i, x in enumerate(batch):
        T = x["cat"].shape[0]
        cat_pad[i, :T] = x["cat"]
        num_pad[i, :T] = x["num"]
        pad_mask[i, :T] = False
        ys.append(x["y"])

    y = torch.stack(ys) # [B]
    
    return {
        "cat": cat_pad, 
        "num": num_pad, 
        "pad_mask": pad_mask, 
        "y": y
    }

# =========================================================
# 3. Vocab Building
# =========================================================
def build_vocabs_on_the_fly(paths, limit=5000):
    logger.info("Building vocabulary from subset...")
    vocabs = {c: {} for c in CATEGORICAL_COLS}
    
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
# 4. Training / Eval Function
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    # Frequent logging setup
    log_interval = 20  # Log every N batches
    start_time = time.time()
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=True)
    count = 0
    
    for batch_idx, batch in enumerate(pbar):
        cat = batch["cat"].to(device, non_blocking=True)
        num = batch["num"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision
        with torch.cuda.amp.autocast(enabled=model.cfg.use_amp):
            logits = model(cat, num, pad_mask) # [B, 2]
            loss = criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.grad_clip)
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        count += 1
        
        pbar.set_postfix({'loss': f"{loss_val:.4f}"})
        
        # Periodic logging
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss_val:.4f} | Time: {elapsed:.2f}s")
            start_time = time.time()
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_probs = [] 
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate"):
            cat = batch["cat"].to(device, non_blocking=True)
            num = batch["num"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=model.cfg.use_amp):
                logits = model(cat, num, pad_mask)
                probs = torch.softmax(logits, dim=1)[:, 1] 
                
            all_y.extend(batch["y"].tolist())
            all_probs.extend(probs.cpu().tolist())
            
    return all_y, all_probs

# =========================================================
# 5. Main
# =========================================================
def main():
    # Hardware - Ensure cuda:0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA is NOT available. Training will be slow.")

    cfg = CFG() # H100 optimized config
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # 1. Load Data Paths
    logger.info("Loading Data Paths...")
    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()
    if not train_paths:
        logger.error("No training data found.")
        return

    # 2. Build Vocabulary
    vocabs = build_vocabs_on_the_fly(train_paths, limit=20000) 

    # 3. Dataset & DataLoader (High num_workers for 100 Cores)
    num_workers = 32
    
    train_ds = TransformerDatasetV2(train_paths, train_labels, vocabs, cfg)
    val_ds = TransformerDatasetV2(test_paths, test_labels, vocabs, cfg)
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn_v2, 
        pin_memory=cfg.pin_memory, persistent_workers=cfg.persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_v2,
        pin_memory=cfg.pin_memory, persistent_workers=cfg.persistent_workers
    )
    
    # 4. Model
    model = SRTTransformerWithoutAddedLabeling(vocabs, len(NUMERIC_COLS), cfg).to(device)
    logger.info(f"Model initialized. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    best_f1 = -1.0
    best_model_state = None
    
    # Save Directory (Custom)
    # Using 'transformer_without_added_labeling' as category in get_run_dir effectively creates:
    # /home/srt/ml_results/transformer_without_added_labeling/{timestamp}/
    # which aligns with user request to use that path.
    save_category = "transformer_without_added_labeling"
    
    # 5. Training Loop
    logger.info("Starting Training...")
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"Epoch {epoch} | Train Loss: {loss:.6f}")
        
        # Validation
        y_true, y_scores = evaluate(model, val_loader, device)
        
        # Evaluate using Utils
        print_evaluation_report(
            "transformer_without_added_labeling", y_true, y_scores, threshold=0.5, 
            category=save_category, save=False, timestamp=timestamp
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
            save_dir = get_run_dir(save_category, timestamp)
            torch.save(model.state_dict(), os.path.join(save_dir, "transformer_without_added_labeling_best.pth"))
            with open(os.path.join(save_dir, "vocabs.json"), "w") as f:
                json.dump(vocabs, f, indent=4)
            logger.info(f"  -> Saved model & vocabs to {save_dir}")
            
    # 6. Final Evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        y_true, y_scores = evaluate(model, val_loader, device)
        print_evaluation_report(
            "transformer_without_added_labeling", y_true, y_scores, threshold=0.5, 
            category=save_category, save=True, timestamp=timestamp
        )
        logger.info("Final evaluation on best model completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error(traceback.format_exc())
