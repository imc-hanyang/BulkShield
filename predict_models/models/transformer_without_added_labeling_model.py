import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Dict, List
import pdb

# =========================================================
# 1. Config & Constraints
# =========================================================
@dataclass
class CFG:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 128  # Increased for H100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.1

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.1

    max_len: int = 512
    # Removed delta config as requested

    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    use_amp: bool = True
    use_tf32: bool = True

CATEGORICAL_COLS = [
    "dep_station_id", "arr_station_id", "route_id", "train_id",
    "action_type",
    "dep_dow",
    "fwd_dep_dow_median",
    "completed_fwd_dep_dow_median",
    "completed_rev_dep_dow_median",
    "rev_dep_dow_median",
]

NUMERIC_COLS = [
    "seat_cnt", "buy_amt", "refund_amt", "cancel_fee", "route_dist_km",
    "travel_time", "lead_time_buy", "lead_time_ref", "hold_time",
    "dep_hour",
    "route_buy_cnt", "fwd_dep_hour_median",
    "rev_buy_cnt", "rev_ratio",
    "completed_fwd_cnt", "completed_fwd_dep_interval_median",
    "completed_fwd_dep_hour_median", "completed_rev_cnt",
    "completed_rev_dep_interval_median", "completed_rev_dep_hour_median",
    "unique_route_cnt",
    "rev_dep_hour_median", "rev_return_gap",
    "overlap_cnt", "same_route_cnt", "rev_route_cnt",
    "repeat_interval", "adj_seat_refund_flag",
    "recent_ref_cnt", "recent_ref_amt", "recent_ref_rate",
]

# =========================================================
# 2. Model Definition
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]


class SRTTransformerWithoutAddedLabeling(nn.Module):
    def __init__(self, vocabs: Dict[str, Dict[int, int]], num_numeric: int, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.cat_embs = nn.ModuleDict()

        for c in CATEGORICAL_COLS:
            # Vocabulary size is max_id + 2 (0: padding, max_id+1: current max, +1 for safety/unknown)
            max_id = max(vocabs.get(c, {}).values()) if vocabs.get(c) else 0
            vocab_size = max_id + 2
            self.cat_embs[c] = nn.Embedding(vocab_size, cfg.d_model)

        self.num_proj = nn.Linear(num_numeric, cfg.d_model)
        
        # Removed self.delta_emb as requested

        self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.max_len + 1)
        
        # Class token
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2) # Classification Head (2 classes)

    def forward(self, cat, num, pad_mask=None):
        B, T, _ = cat.shape

        cat_sum = 0.0
        for i, c in enumerate(CATEGORICAL_COLS):
            cat_sum = cat_sum + self.cat_embs[c](cat[:, :, i])
        
        # Combine Categorical + Numeric embeddings
        x = cat_sum + self.num_proj(num)
        
        # Prepend CLS token
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) # [B, T+1, D]

        # Extend mask for CLS token
        if pad_mask is not None:
            # pad_mask is [B, T], True means padding (ignore)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
            key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1) # [B, T+1]
        else:
            key_padding_mask = None

        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        
        # Use CLS token output for classification
        logit = self.head(x[:, 0]) # [B, 2]
        return logit
