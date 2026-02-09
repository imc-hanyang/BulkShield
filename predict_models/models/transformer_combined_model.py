"""
Combined Transformer Model for Sequence-Based Fraud Detection

This module implements a simplified Transformer classifier that uses
feature projection (without separate categorical/numeric handling) and
variable-length sequence support. Unlike SRTTransformerClassifier, this
model does not use separate embeddings for categorical features.

Architecture:
    1. Feature Projection: Linear layer projects all features to d_model
    2. CLS Token: Learnable classification token
    3. Positional Encoding: Sinusoidal encoding
    4. Transformer Encoder: Pre-norm multi-layer encoder
    5. Classification Head: Linear layer on CLS output

Comparison to SRTTransformerClassifier:
    - Simpler: No separate categorical embeddings or time delta embeddings
    - Faster: Fewer embedding lookups
    - May be less expressive for features with many categories

Usage:
    >>> model = TransformerCombined(input_dim=37, d_model=128)
    >>> logits = model(features, lengths)  # [B, 2]
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    
    Standard Transformer positional encoding from "Attention Is All You Need".
    
    Args:
        d_model: Model embedding dimension.
        max_len: Maximum supported sequence length (default: 5000).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, T, d_model].
            
        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1)]


class TransformerCombined(nn.Module):
    """
    Simplified Transformer classifier with feature projection.
    
    Projects all input features through a single linear layer before
    processing with Transformer encoder. Uses CLS token for classification.
    
    Args:
        input_dim: Number of input features per time step.
        d_model: Transformer hidden dimension (default: 128).
        nhead: Number of attention heads (default: 4).
        num_layers: Number of encoder layers (default: 3).
        dim_feedforward: FFN hidden dimension (default: 256).
        dropout: Dropout probability (default: 0.1).
        max_len: Maximum sequence length (default: 512).
    
    Example:
        >>> model = TransformerCombined(input_dim=37)
        >>> x = torch.randn(32, 100, 37)  # [B, T, F]
        >>> lengths = torch.tensor([100] * 32)
        >>> logits = model(x, lengths)  # [32, 2]
    """
    
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_len=512):
        super(TransformerCombined, self).__init__()
        
        self.d_model = d_model
        
        # Project all input features to d_model dimension
        self.feature_proj = nn.Linear(input_dim, d_model)
        
        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional encoding (+1 for CLS token)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len + 1)
        
        # Transformer encoder with pre-norm architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True, 
            norm_first=True  # Pre-norm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, 2)
        
    def forward(self, x, lengths):
        """
        Forward pass for fraud classification.
        
        Args:
            x: Input features [B, T, input_dim].
            lengths: Actual sequence lengths [B] (excluding CLS token).
            
        Returns:
            Logits tensor of shape [B, 2].
        """
        B, T, _ = x.shape
        
        # 1. Project input features to d_model dimension
        x = self.feature_proj(x)  # [B, T, d_model]
        
        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, T+1, d_model]
        
        # 3. Add positional encoding
        x = self.pos_encoder(x)
        
        # 4. Create padding mask
        # True = padding position (to be ignored)
        # CLS token is always valid; subsequent tokens valid up to lengths[i]
        T_plus_1 = x.size(1)
        mask = torch.arange(T_plus_1, device=x.device).expand(B, T_plus_1)
        # Add 1 to lengths to account for CLS token
        target_lengths = (lengths + 1).unsqueeze(1)
        padding_mask = mask >= target_lengths
        
        # 5. Apply Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 6. Classify using CLS token output
        cls_out = x[:, 0, :]
        logits = self.fc(cls_out)
        
        return logits

