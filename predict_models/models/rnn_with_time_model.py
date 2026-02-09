"""
Time-Context RNN Models for Sequence-Based Fraud Detection

This module implements RNN variants that incorporate discretized time delta
information as an additional input signal. Unlike standard RNN models that
only see feature sequences, these models also encode the time between events.

Architecture:
    - Feature Projection: Linear layer to project features to hidden dimension
    - Time Delta Embedding: Embedding layer for discretized inter-event times
    - Combined Input: Sum of projected features and time embeddings
    - RNN Core: Configurable RNN type (RNN/LSTM/GRU)
    - Classification: Linear layer on final hidden state

Key Concept:
    Time between transactions is an important fraud signal - scalpers often
    show distinctive temporal patterns (rapid-fire purchases followed by
    strategic refunds). By explicitly encoding time deltas, the model can
    better learn these patterns.

Usage:
    >>> model = RNNWithTime(input_size=37, hidden_size=128)
    >>> logits = model(features, deltas, lengths)  # [B, 2]
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class TimeContextRNNBase(nn.Module):
    """
    Base class for RNN models with time delta context.
    
    Combines feature projections with time delta embeddings before
    processing through a configurable RNN architecture (RNN/LSTM/GRU).
    
    Args:
        input_size: Number of input features per time step.
        hidden_size: RNN hidden dimension.
        output_size: Number of output classes (default: 2).
        num_layers: Number of stacked RNN layers (default: 1).
        delta_vocab_size: Size of time delta vocabulary (default: 289).
        device: CUDA device for computation.
        rnn_type: Type of RNN cell ('RNN', 'LSTM', or 'GRU').
    """
    
    def __init__(self, input_size, hidden_size, output_size=2, num_layers=1, 
                 delta_vocab_size=289, device='cuda:0', rnn_type='RNN'):
        super(TimeContextRNNBase, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.rnn_type = rnn_type
        
        # Project input features to hidden dimension
        self.feature_proj = nn.Linear(input_size, hidden_size)
        
        # Embed discretized time deltas (default: 289 buckets = 48 hours)
        self.delta_emb = nn.Embedding(delta_vocab_size, hidden_size)
        
        # Initialize RNN based on specified type
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Classification layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, delta, lengths):
        """
        Forward pass with feature and time delta inputs.
        
        Args:
            x: Feature sequence [B, T, input_size].
            delta: Time delta bucket indices [B, T].
            lengths: Actual sequence lengths [B] for packing.
            
        Returns:
            Logits tensor of shape [B, output_size].
        """
        # Project features and embed time deltas
        x_proj = self.feature_proj(x)       # [B, T, H]
        d_emb = self.delta_emb(delta)       # [B, T, H]
        
        # Combine feature and time representations
        combined_input = x_proj + d_emb     # [B, T, H]
        
        # Pack for efficient variable-length processing
        packed_input = rnn_utils.pack_padded_sequence(
            combined_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # RNN forward pass (handle LSTM separately due to cell state)
        if self.rnn_type == 'LSTM':
            packed_out, (hidden, cell) = self.rnn(packed_input)
        else:  # RNN, GRU
            packed_out, hidden = self.rnn(packed_input)
        
        # Extract final hidden state and classify
        last_hidden = hidden[-1] 
        logits = self.fc(last_hidden)
        return logits


class RNNWithTime(TimeContextRNNBase):
    """
    Vanilla RNN with time delta context.
    
    Convenience subclass that defaults to RNN architecture.
    
    Args:
        input_size: Number of input features per time step.
        hidden_size: RNN hidden dimension (default: 128).
        device: CUDA device for computation.
    """
    
    def __init__(self, input_size, hidden_size=128, device='cuda:0'):
        super().__init__(input_size, hidden_size, device=device, rnn_type='RNN')

