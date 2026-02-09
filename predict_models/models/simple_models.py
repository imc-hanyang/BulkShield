"""
Simple Neural Network Models for Baseline Fraud Detection

This module provides basic neural network architectures for comparison
against more complex sequence models. These models operate on aggregated
per-user features rather than full transaction sequences.

Models Included:
    - LogisticRegression: Single linear layer for binary classification
    - SVR: Linear regression layer (can be used with SVM-like loss)
    - MLP: Multi-layer perceptron with dropout regularization

Usage:
    >>> model = MLP(input_dim=37, hidden_dim=128)
    >>> logits = model(aggregated_features)  # [B, 1]
"""

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Simple logistic regression for binary classification.
    
    Single linear layer that outputs logits for use with BCEWithLogitsLoss.
    
    Args:
        input_dim: Number of input features.
    """
    
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """Return raw logits for stability with BCEWithLogitsLoss."""
        return self.linear(x)


class SVR(nn.Module):
    """
    Simple linear regression layer.
    
    Can be used with epsilon-insensitive loss for SVM regression behavior.
    
    Args:
        input_dim: Number of input features.
    """
    
    def __init__(self, input_dim):
        super(SVR, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """Return raw regression output."""
        return self.linear(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron for binary classification.
    
    Two hidden layers with ReLU activation and dropout regularization.
    
    Args:
        input_dim: Number of input features.
        hidden_dim: First hidden layer dimension (default: 128).
        dropout_prob: Dropout probability (default: 0.3).
    
    Architecture:
        input_dim -> hidden_dim -> hidden_dim/2 -> 1
    """
    
    def __init__(self, input_dim, hidden_dim=128, dropout_prob=0.3):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """Return raw logits for binary classification."""
        return self.net(x)

