"""
Vanilla RNN Model for Sequence-Based Fraud Detection

This module implements a vanilla Recurrent Neural Network (RNN) classifier
for detecting fraudulent ticket transactions. The model uses the tanh
non-linearity and processes variable-length sequences.

Architecture:
    - Input: Sequence of feature vectors [B, T, F]
    - RNN: Multi-layer vanilla RNN with tanh activation
    - Classification: Linear layer on final hidden state

Note:
    Vanilla RNN is included for baseline comparison. For better performance
    on long sequences, consider using LSTM or GRU models which handle
    vanishing gradients more effectively.

Usage:
    >>> model = CustomRNN(input_size=37, hidden_size=128, output_size=2)
    >>> logits = model(x, lengths)  # [B, 2]
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class CustomRNN(nn.Module):
    """
    Vanilla RNN model for variable-length sequence classification.
    
    Processes sequences of transaction events using standard RNN cells
    with tanh non-linearity. Uses the final hidden state for classification.
    
    Args:
        input_size: Number of input features per time step.
        hidden_size: RNN hidden dimension.
        output_size: Number of output classes (typically 2).
        num_layers: Number of stacked RNN layers (default: 1).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(CustomRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer (uses tanh non-linearity by default)
        self.rnn = nn.RNN(
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
        # Pack sequences for efficient computation (same as LSTM)
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # RNN forward pass (no cell state unlike LSTM)
        packed_out, hidden = self.rnn(packed_x)

        # Extract final hidden state from last layer
        last_hidden = hidden[-1]

        out = self.fc(last_hidden)
        return out