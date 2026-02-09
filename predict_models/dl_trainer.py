"""
Deep Learning Trainer - Training Loop and Inference Functions

This module provides the core training and inference functions for deep learning
models in the SRT fraud detection system. It supports both supervised classification
(LSTM, RNN, GRU) and unsupervised reconstruction-based anomaly detection (LSTM-AE).

Key Functions:
    - train_one_epoch(): Single epoch training with gradient clipping
    - get_model_predictions(): Batch inference with score extraction

Model Types:
    - 'lstm': Supervised classification using cross-entropy loss
    - 'lstm-ea': Unsupervised autoencoder using reconstruction error as anomaly score

Usage:
    >>> from dl_trainer import train_one_epoch, get_model_predictions
    >>> train_loss = train_one_epoch(model, train_loader, optimizer, criterion, 
    ...                               device, epoch=1, model_type='lstm')
    >>> labels, scores = get_model_predictions(model, test_loader, device)
"""

import torch
import torch.nn as nn
import numpy as np


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, model_type='lstm'):
    """
    Train the model for one epoch.
    
    Supports both supervised classification and unsupervised reconstruction objectives.
    Includes gradient clipping for training stability.
    
    Args:
        model: PyTorch model to train.
        dataloader: Training data loader yielding (inputs, targets, lengths).
        optimizer: Optimizer instance (e.g., Adam).
        criterion: Loss function (CrossEntropyLoss for classification, MSELoss for AE).
        device: Device to run training on (cuda/cpu).
        epoch: Current epoch number (for logging).
        model_type: 'lstm' for classification, 'lstm-ea' for autoencoder.
    
    Returns:
        float: Average loss over all batches in the epoch.
    
    Note:
        - Classification mode: Uses sequence lengths for pack_padded_sequence
        - Autoencoder mode: Ignores lengths, computes reconstruction loss
    """
    model.train()
    total_loss = 0

    for i, (inputs, targets, lengths) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()

        if model_type == 'lstm-ea':
            # AutoEncoder mode: Input -> Reconstruction
            # LSTMAutoEncoder.forward() takes only x (no lengths needed)
            outputs = model(inputs)
            # Reconstruction loss: MSE between input and output
            loss = criterion(outputs, inputs)
            
        else:
            # Classification mode: Input -> Class Logits
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)

        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        
        # Progress logging every 10 batches
        if (i + 1) % 10 == 0:
            print(f"[{model_type.upper()} Train] Epoch {epoch} | Batch {i + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def get_model_predictions(model, dataloader, device, model_type='lstm'):
    """
    Get model predictions (risk scores) and labels for the entire dataset.
    
    For classification models, returns the probability of the fraud class.
    For autoencoders, returns the reconstruction error as the anomaly score.
    
    Args:
        model: Trained PyTorch model.
        dataloader: Data loader yielding (inputs, targets, lengths).
        device: Device to run inference on.
        model_type: 'lstm' for classification, 'lstm-ea' for autoencoder.
    
    Returns:
        tuple: (labels, scores) as numpy arrays where:
            - labels: Ground truth labels [N]
            - scores: Risk scores [N] (higher = more likely fraud)
    
    Score Interpretation:
        - Classification: P(fraud) from softmax output
        - Autoencoder: Mean reconstruction error (higher = more anomalous)
    """
    model.eval()
    all_labels = []
    all_scores = []
    
    # MSE with no reduction for per-sample error calculation
    criterion_mse_none = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            if model_type == 'lstm-ea':
                # Reconstruction Error = Anomaly Score
                outputs = model(inputs)
                # Compute per-sample MSE: [B, T, F] -> mean over T and F
                # Using mean instead of sum for scale-invariance
                loss_per_sample = criterion_mse_none(outputs, inputs).mean(dim=[1, 2])
                scores = loss_per_sample
            else:
                # Classification: P(Fraud) = softmax[:, 1]
                outputs = model(inputs, lengths)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                scores = probs

            all_labels.extend(targets.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    return np.array(all_labels), np.array(all_scores)