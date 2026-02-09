"""
GRU Model for Sequence-Based Fraud Detection

This module implements a Gated Recurrent Unit (GRU) classifier for detecting
fraudulent ticket transactions. GRU is a variant of LSTM with a simplified
gating mechanism that often achieves similar performance with fewer parameters.

Architecture:
    - Input: Sequence of feature vectors [B, T, F]
    - GRU: Multi-layer GRU with update and reset gates
    - Classification: Linear layer on final hidden state

Comparison to LSTM:
    - GRU uses 2 gates (update, reset) vs LSTM's 3 gates (input, forget, output)
    - Fewer parameters and potentially faster training
    - Similar performance to LSTM on many sequence tasks

Usage:
    >>> model = CustomGRU(input_size=37, hidden_size=128, output_size=2)
    >>> logits = model(x, lengths)  # [B, 2]
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class CustomGRU(nn.Module):
    """
    GRU model for variable-length sequence classification.
    
    Processes sequences of transaction events using Gated Recurrent Units.
    Uses the final hidden state for binary fraud/normal classification.
    
    Args:
        input_size: Number of input features per time step.
        hidden_size: GRU hidden dimension.
        output_size: Number of output classes (typically 2).
        num_layers: Number of stacked GRU layers (default: 2).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super(CustomGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Classification layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence classification.
        
        Args:
            x: Input features [B, T, input_size].
            lengths: Actual sequence lengths [B] for packing.
            
        Returns:
            Logits tensor of shape [B, output_size].
        """
        # Pack sequences for efficient computation (same as LSTM/RNN)
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # GRU forward pass
        packed_out, hidden = self.gru(packed_x)

        # Extract final hidden state from last layer
        # hidden shape: [num_layers, B, hidden_size]
        last_hidden = hidden[-1]

        out = self.fc(last_hidden)
        return out

