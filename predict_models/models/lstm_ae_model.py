"""
LSTM AutoEncoder for Unsupervised Anomaly Detection

This module implements an LSTM-based AutoEncoder for detecting fraudulent
transactions through reconstruction error. Unlike supervised models, this
approach learns to reconstruct normal transaction sequences and flags
high-reconstruction-error samples as anomalies.

Architecture:
    - Encoder: Multi-layer LSTM that compresses sequence to fixed-size vector
    - Decoder: Multi-layer LSTM that reconstructs the original sequence
    - Anomaly Score: Mean squared reconstruction error per sample

Key Concepts:
    - Training on normal-only data (or full data with fraud as noise)
    - Fraudulent sequences have higher reconstruction error
    - No explicit labels needed during training (unsupervised)

Usage:
    >>> model = LSTMAEModel(input_dim=37, hidden_dim=64)
    >>> model.fit(train_dataset)
    >>> scores = model.predict_score(test_dataset)  # Higher = more anomalous
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data_utils import pad_collate_fn


class LSTMAutoEncoder(nn.Module):
    """
    LSTM-based sequence AutoEncoder for reconstruction.
    
    Compresses input sequences to a fixed-size hidden representation
    and reconstructs the original sequence. Used for anomaly detection
    via reconstruction error.
    
    Args:
        input_dim: Feature dimension per time step.
        hidden_dim: Hidden dimension of LSTM layers.
        num_layers: Number of stacked LSTM layers in encoder/decoder.
    
    Note:
        Decoder uses the final hidden state repeated across time steps
        to reconstruct the full sequence.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMAutoEncoder, self).__init__()
        # Encoder: compress sequence to hidden representation
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Decoder: reconstruct sequence from hidden representation
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        """
        Encode and reconstruct input sequence.
        
        Args:
            x: Input sequence [B, T, input_dim].
            
        Returns:
            Reconstructed sequence [B, T, input_dim].
        """
        # Encode: extract final hidden state
        _, (hidden, _) = self.encoder(x)
        seq_len = x.size(1)
        
        # Repeat final hidden state across all time steps
        repeat_hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode: reconstruct sequence from repeated hidden state
        output, _ = self.decoder(repeat_hidden)
        return output


class LSTMAEModel:
    """
    Wrapper class for training and inference with LSTM AutoEncoder.
    
    Provides scikit-learn style fit/predict interface for easy integration
    with the evaluation pipeline.
    
    Args:
        input_dim: Feature dimension per time step.
        hidden_dim: LSTM hidden dimension (default: 64).
        epochs: Number of training epochs (default: 10).
        batch_size: Batch size for training (default: 256).
        device: CUDA device for computation (default: 'cuda:3').
    
    Example:
        >>> model = LSTMAEModel(input_dim=37)
        >>> model.fit(train_dataset)
        >>> anomaly_scores = model.predict_score(test_dataset)
    """
    
    def __init__(self, input_dim, hidden_dim=64, epochs=10, batch_size=256, device='cuda:3'):
        self.device = device
        self.model = LSTMAutoEncoder(input_dim, hidden_dim).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, train_data):
        """
        Train the autoencoder on transaction sequences.
        
        Args:
            train_data: LazyDataset instance with transaction sequences.
        """
        # Use pad_collate_fn for variable-length sequence handling
        loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=pad_collate_fn, 
            num_workers=4
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i, batch in enumerate(loader):
                # pad_collate_fn returns (inputs, targets, lengths)
                x, _, _ = batch

                x = x.to(self.device)
                self.optimizer.zero_grad()
                
                # Reconstruction loss (MSE)
                recon = self.model(x)
                loss = torch.mean((recon - x) ** 2)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    print(f"[LSTM-AE] Epoch {epoch + 1} | Batch {i + 1}/{len(loader)} | Loss: {loss.item():.4f}")

    def predict_score(self, test_data) -> np.ndarray:
        """
        Compute anomaly scores for test sequences.
        
        Anomaly score is the mean squared reconstruction error.
        Higher scores indicate more anomalous (potentially fraudulent) samples.
        
        Args:
            test_data: LazyDataset instance with test sequences.
            
        Returns:
            np.ndarray: Anomaly scores for each sample.
        """
        loader = DataLoader(
            test_data, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=pad_collate_fn, 
            num_workers=4
        )

        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in loader:
                # Unpack batch (inputs, targets, lengths)
                x, _, _ = batch

                x = x.to(self.device)
                recon = self.model(x)
                
                # Per-sample reconstruction error (mean over time and features)
                loss_per_sample = torch.mean((recon - x) ** 2, dim=[1, 2])
                scores.extend(loss_per_sample.cpu().numpy())
        
        return np.array(scores)