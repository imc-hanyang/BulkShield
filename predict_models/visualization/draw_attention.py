
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/srt/modeling")

# Import from transformer_modeling instead of transformer_main/model
from transformer_modeling import (
    SRTTransformerClassifier, CFG, SRTSequenceDataset, collate_fn, 
    load_vocabs, NUMERIC_COLS, VOCABS_JSON, MANIFEST_TEST, BEST_PT
)

def draw_attention_map(model, batch, layer_idx=-1, head_idx=0, save_path="attention_map.png"):
    """
    Draws attention map for a single sample in the batch.
    layer_idx: Which layer to visualize (default: last layer)
    head_idx: Which head to visualize (default: 0)
    """
    model.eval()
    
    # 1. Forward to get Attention Weights
    with torch.no_grad():
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        delta = batch["delta"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        # return_attn=True is required
        logits, attns = model(cat, num, delta, pad_mask, return_attn=True)
    
    # attns is list of [B, NumHeads, SeqLen, SeqLen]
    # Get specific layer
    attn_layer = attns[layer_idx] # [B, Heads, T, T]
    
    # Get specific sample (index 0) and head
    # shape: [T, T]
    attn_map = attn_layer[0, head_idx].cpu().numpy()
    
    # Filter padding (optional, based on pad_mask)
    # pad_mask[0] is [SeqLen], True=Padding
    # We only want to visualize real tokens
    # Note: Model adds CLS token at index 0, so real tokens start at 1
    # pad_mask length is T-1 (original input length)
    
    mask = pad_mask[0].cpu().numpy() # [Original_T]
    
    # Actual length including CLS
    T_total = attn_map.shape[0]
    # Valid tokens: CLS (idx 0) + Non-padded tokens
    valid_indices = [0] + [i+1 for i, m in enumerate(mask) if not m]
    
    # Slice the attention map
    attn_map_valid = attn_map[np.ix_(valid_indices, valid_indices)]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_map_valid, cmap="viridis", square=True)
    plt.title(f"Attention Map (Layer {layer_idx}, Head {head_idx})")
    plt.xlabel("Key (Source)")
    plt.ylabel("Query (Target)")
    
    plt.savefig(save_path)
    print(f"Saved attention map to {save_path}")
    plt.close()

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    
    # 1. Define Model Path (Use latest DL result)
    # Check /home/srt/ml_results/dl/latest/
    LATEST_DIR = "/home/srt/ml_results/dl/latest"
    MODEL_PATH = os.path.join(LATEST_DIR, "transformer_best.pth")
    VOCABS_PATH = os.path.join(LATEST_DIR, "vocabs.json")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run transformer_main.py first to train a model.")
        sys.exit(1)
        
    if not os.path.exists(VOCABS_PATH):
        print(f"Error: Vocabs not found at {VOCABS_PATH}")
        print("Please ensure transformer_main.py saves vocabs.json.")
        sys.exit(1)

    # 1. Load Vocabs
    print(f"Loading Vocabs from {VOCABS_PATH}...")
    with open(VOCABS_PATH, 'r') as f:
        vocabs = json.load(f)
        # Convert keys to int if needed (json keys are strings)
        # Check structure: {col: {val: id}}
        # We need to ensure string keys for inner dict are converted to int if they were ints
        # But wait, in transformer_main, we build vocabs with int or str?
        # transformer_main: vocabs[c][v_int] = id. json dumps keys as strings.
        # We need to be careful. safe_int logic uses int keys typically.
        # Let's rebuild properly.
        new_vocabs = {}
        for c, v_map in vocabs.items():
            new_map = {}
            for k, v in v_map.items():
                try:
                    new_map[int(k)] = v
                except:
                    new_map[k] = v
            new_vocabs[c] = new_map
        vocabs = new_vocabs

    cfg = CFG()
    
    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    model = SRTTransformerClassifier(vocabs, len(NUMERIC_COLS), cfg).to(device)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # 3. Load Data Sample
    print("Loading Data Sample...")
    # Use MANIFEST_TEST if available, or just load from paths
    # transformer_modeling defines MANIFEST_TEST, so we can use it
    if os.path.exists(MANIFEST_TEST):
        test_df = pd.read_csv(MANIFEST_TEST)
        # Filter for a positive sample (Fraud) if possible
        fraud_df = test_df[test_df['y'] == 1]
        if len(fraud_df) > 0:
            sample_df = fraud_df.iloc[:1]
            print("Selected a FRAUD sample.")
        else:
            sample_df = test_df.iloc[:1]
            print("Selected a NORMAL sample (no fraud found in test).")
            
        ds = SRTSequenceDataset(sample_df, vocabs, cfg.max_len)
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        
        batch = next(iter(loader))
        
        # 4. Draw
        os.makedirs("attn_vis", exist_ok=True)
        
        # Draw Layer 0, Head 0
        draw_attention_map(model, batch, layer_idx=0, head_idx=0, save_path="attn_vis/attn_l0_h0.png")
        
        # Draw Last Layer, Head 0
        draw_attention_map(model, batch, layer_idx=-1, head_idx=0, save_path="attn_vis/attn_last_h0.png")
        
        # Draw Last Layer, Head 1
        draw_attention_map(model, batch, layer_idx=-1, head_idx=1, save_path="attn_vis/attn_last_h1.png")
    else:
        print(f"Manifest not found at {MANIFEST_TEST}")

