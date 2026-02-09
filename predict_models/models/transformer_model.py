"""
Transformer Model for Sequence-Based Fraud Detection

This module implements a Transformer-based classifier for detecting fraudulent
ticket transactions in the SRT railway system. The model processes sequences
of user transaction events and produces a binary fraud/normal classification.

Architecture Overview:
    1. Feature Embedding: Separate embeddings for categorical features
    2. Numeric Projection: Linear projection for continuous features
    3. Time Delta Embedding: Discretized time differences between events
    4. Positional Encoding: Sinusoidal encoding for sequence position
    5. CLS Token: Learnable classification token prepended to sequence
    6. Transformer Encoder: Multi-layer self-attention with GELU activation
    7. Classification Head: Linear layer on CLS token output

Key Design Decisions:
    - Pre-norm Transformer architecture for training stability
    - Sum of embeddings rather than concatenation for memory efficiency
    - Time delta bucketization to handle variable inter-event intervals
    - Attention mask handling for variable-length sequences

Configuration:
    CFG dataclass provides all hyperparameters with H100-optimized defaults.

Usage:
    >>> model = SRTTransformerClassifier(vocabs, num_numeric=31, cfg=CFG())
    >>> logits = model(cat_tensor, num_tensor, delta_tensor, pad_mask)
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Dict, List


# =============================================================================
# Configuration and Feature Definitions
# =============================================================================

@dataclass
class CFG:
    """
    Hyperparameter configuration for the Transformer model.
    
    Default values are optimized for H100 GPU training with mixed precision.
    
    Attributes:
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        batch_size: Batch size (increased for H100 GPU memory).
        lr: Learning rate for AdamW optimizer.
        weight_decay: L2 regularization strength.
        grad_clip: Maximum gradient norm for clipping.
        val_ratio: Fraction of data for validation.
        
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_ff: Feedforward network hidden dimension.
        dropout: Dropout probability.
        
        max_len: Maximum sequence length (truncates longer sequences).
        delta_bucket_size_min: Minutes per time delta bucket.
        delta_max_bucket: Maximum bucket index (288 = 48 hours / 10 min).
        
        pin_memory: Use pinned memory for DataLoader.
        persistent_workers: Keep DataLoader workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        
        use_amp: Enable automatic mixed precision training.
        use_tf32: Enable TensorFloat-32 for matrix operations.
    """
    seed: int = 42
    epochs: int = 10
    batch_size: int = 128
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
    delta_bucket_size_min: int = 10
    delta_max_bucket: int = 48 * 6  # 288 buckets (48 hours at 10min intervals)

    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    use_amp: bool = True
    use_tf32: bool = True


# Categorical features requiring embedding lookup
# Each maps to a vocabulary of discrete values
CATEGORICAL_COLS = [
    "dep_station_id",              # Departure station (32 stations)
    "arr_station_id",              # Arrival station (32 stations)
    "route_id",                    # Route hash (~395 routes)
    "train_id",                    # Train number (~180 trains)
    "action_type",                 # 1=purchase, 2=refund
    "dep_dow",                     # Departure day of week (0-6)
    "fwd_dep_dow_median",          # Forward ticket median departure day
    "completed_fwd_dep_dow_median",# Completed forward trip median departure day
    "completed_rev_dep_dow_median",# Completed reverse trip median departure day
    "rev_dep_dow_median",          # Reverse ticket median departure day
]

# Numeric features projected through linear layer
# These are continuous values that undergo log1p scaling
NUMERIC_COLS = [
    # Transaction attributes
    "seat_cnt", "buy_amt", "refund_amt", "cancel_fee", "route_dist_km",
    # Time features
    "travel_time", "lead_time_buy", "lead_time_ref", "hold_time", "dep_hour",
    # Route history
    "route_buy_cnt", "fwd_dep_hour_median", "rev_buy_cnt", "rev_ratio",
    # Completed trip statistics
    "completed_fwd_cnt", "completed_fwd_dep_interval_median",
    "completed_fwd_dep_hour_median", "completed_rev_cnt",
    "completed_rev_dep_interval_median", "completed_rev_dep_hour_median",
    "unique_route_cnt",
    # Active ticket features
    "rev_dep_hour_median", "rev_return_gap",
    # Fraud signals
    "overlap_cnt", "same_route_cnt", "rev_route_cnt",
    "repeat_interval", "adj_seat_refund_flag",
    # Refund history
    "recent_ref_cnt", "recent_ref_amt", "recent_ref_rate",
]


# =============================================================================
# Model Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    
    Uses the original Transformer positional encoding formula from
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    Args:
        d_model: Model embedding dimension.
        max_len: Maximum supported sequence length.
    """
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [B, T, D].
            
        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class SRTTransformerClassifier(nn.Module):
    """
    Transformer-based classifier for SRT fraud detection.
    
    Combines categorical embeddings, numeric projections, and time delta
    embeddings into a unified representation processed by a Transformer
    encoder. Uses a CLS token for sequence-level classification.
    
    Args:
        vocabs: Dictionary mapping categorical column names to their
                vocabulary dictionaries (value -> integer id).
        num_numeric: Number of numeric features (typically 31).
        cfg: Configuration dataclass with hyperparameters.
    
    Example:
        >>> vocabs = {'dep_station_id': {101: 1, 102: 2, ...}, ...}
        >>> model = SRTTransformerClassifier(vocabs, num_numeric=31, cfg=CFG())
        >>> logits = model(cat, num, delta, pad_mask)  # [B, 2]
    """
    
    def __init__(self, vocabs: Dict[str, Dict[int, int]], num_numeric: int, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        # Categorical feature embeddings
        # Each categorical column gets its own embedding table
        self.cat_embs = nn.ModuleDict()
        for c in CATEGORICAL_COLS:
            # Vocab size = max_id + 2 (for padding/unknown handling)
            max_id = max(vocabs.get(c, {}).values()) if vocabs.get(c) else 0
            vocab_size = max_id + 2
            self.cat_embs[c] = nn.Embedding(vocab_size, cfg.d_model)

        # Numeric features projection to d_model dimension
        self.num_proj = nn.Linear(num_numeric, cfg.d_model)
        
        # Time delta embedding (discretized inter-event intervals)
        self.delta_emb = nn.Embedding(cfg.delta_max_bucket + 1, cfg.d_model)
        
        # Positional encoding
        self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.max_len + 1)
        
        # Learnable CLS token for classification
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        # Transformer encoder with pre-norm architecture
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        
        # Final normalization and classification head
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)  # Binary classification

    def forward(self, cat, num, delta, pad_mask):
        """
        Forward pass for fraud classification.
        
        Args:
            cat: Categorical features [B, T, num_cat_cols] as integer indices.
            num: Numeric features [B, T, num_numeric] as float values.
            delta: Time delta bucket indices [B, T] as integers.
            pad_mask: Padding mask [B, T] where True = padding (ignore).
            
        Returns:
            Logits tensor of shape [B, 2] for binary classification.
        """
        B, T, _ = cat.shape

        # Sum all categorical embeddings
        # Each categorical column produces [B, T, d_model], summed together
        cat_sum = 0.0
        for i, c in enumerate(CATEGORICAL_COLS):
            cat_sum = cat_sum + self.cat_embs[c](cat[:, :, i])
        
        # Combine all feature representations
        # cat_sum: [B, T, d_model]
        # num_proj(num): [B, T, d_model]
        # delta_emb(delta): [B, T, d_model]
        x = cat_sum + self.num_proj(num) + self.delta_emb(delta)

        # Prepend learnable CLS token
        cls = self.cls.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)     # [B, T+1, d_model]

        # Extend mask for CLS token (CLS should never be masked)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)  # [B, T+1]
        
        # Apply positional encoding and Transformer encoder
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        
        # Classification from CLS token (first position)
        logit = self.head(x[:, 0])  # [B, 2]
        return logit

