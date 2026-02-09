"""
LSTM Model for Sequence-Based Fraud Detection

This module implements a Long Short-Term Memory (LSTM) classifier for detecting
fraudulent ticket transactions. The model processes variable-length sequences
of user transaction events and outputs binary classification logits.

Architecture:
    - Input: Sequence of feature vectors [B, T, F]
    - LSTM: Multi-layer LSTM with hidden state tracking
    - Classification: Linear layer on final hidden state

Key Features:
    - Handles variable-length sequences via pack_padded_sequence
    - Uses final hidden state for sequence-level classification
    - Supports multi-layer stacking for increased model capacity

Usage:
    >>> model = CustomLSTM(input_size=37, hidden_size=128, output_size=2)
    >>> logits = model(x, lengths)  # [B, 2]
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class CustomLSTM(nn.Module):
    """
    LSTM model for variable-length sequence classification.
    
    Processes sequences of transaction events and produces binary
    fraud/normal classification using the final hidden state.
    
    Args:
        input_size: Number of input features per time step.
        hidden_size: LSTM hidden dimension.
        output_size: Number of output classes (typically 2).
        num_layers: Number of stacked LSTM layers (default: 1).
    
    Example:
        >>> model = CustomLSTM(input_size=37, hidden_size=128, output_size=2)
        >>> x = torch.randn(32, 100, 37)  # [B, T, F]
        >>> lengths = torch.tensor([100] * 32)
        >>> logits = model(x, lengths)  # [32, 2]
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(CustomLSTM, self).__init__()

        # Store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer (batch_first=True means input shape is [Batch, Seq, Feature])
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected classification layer
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
        # 1. Pack: Optimize computation by packing variable-length sequences
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # 2. Forward: Run LSTM on packed sequences
        # out contains all hidden states, hidden contains final states
        packed_out, (hidden, cell) = self.lstm(packed_x)

        # 3. Extract final hidden state from last layer
        # hidden shape: [num_layers, B, hidden_size]
        last_hidden = hidden[-1]  # [B, hidden_size]

        # 4. Classification through fully connected layer
        out = self.fc(last_hidden)
        return out