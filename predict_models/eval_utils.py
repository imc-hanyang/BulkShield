"""
Evaluation Utilities - Unified Metrics and Result Persistence

This module provides standardized evaluation utilities for all fraud detection
models in the SRT system. It ensures consistent metric calculation, formatted
reporting, and structured result storage across deep learning, machine learning,
and one-class classification models.

Key Components:
    - compute_metrics(): Calculate ROC-AUC, PR-AUC, Precision, Recall, F1
    - save_results(): Persist predictions, labels, and metrics to disk
    - print_evaluation_report(): Formatted console output with optional saving
    - get_run_dir(): Timestamp-based directory management for experiments

Result Storage Structure:
    {RESULTS_BASE_DIR}/{category}/{timestamp}/{model_name}/
        - scores.npy: Predicted risk scores
        - labels.npy: Ground truth labels
        - metrics.json: Evaluation metrics
        - meta.json: Experiment metadata

Usage:
    >>> from eval_utils import print_evaluation_report
    >>> metrics = print_evaluation_report("LSTM", y_true, y_scores, 
    ...                                    threshold=0.5, category='dl')
"""

import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Base directory for storing experiment results
RESULTS_BASE_DIR = "/home/srt/ml_results"


def get_run_dir(category: str, timestamp: str = None):
    """
    Create and return a timestamped directory for storing experiment results.
    
    Directory structure: RESULTS_BASE_DIR / category / timestamp
    Also maintains a 'latest' symlink pointing to the most recent run.
    
    Args:
        category: Model category ('dl', 'ml', 'occ_dl', 'occ_ml', 'transformer').
        timestamp: Fixed timestamp string. If None, uses current time.
    
    Returns:
        str: Absolute path to the run directory.
    
    Example:
        >>> run_dir = get_run_dir('dl')  # Creates /home/srt/ml_results/dl/2026-02-09_01-30
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
    # Category is the top-level folder
    category_dir = os.path.join(RESULTS_BASE_DIR, category)
    run_dir = os.path.join(category_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Update 'latest' symlink for easy access to most recent results
    # Example: ml_results/dl/latest -> ml_results/dl/2026-01-22_12-00
    latest_link = os.path.join(category_dir, "latest")
    
    try:
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        # Use relative path for portability
        os.symlink(timestamp, latest_link)
    except:
        pass  # Continue even if symlink creation fails
    
    return run_dir


def compute_metrics(y_true, y_scores, threshold=0.5):
    """
    Compute all evaluation metrics and return as a dictionary.
    
    Calculates standard binary classification metrics optimized for
    fraud detection where positive class (fraud) is the minority.
    
    Args:
        y_true: Ground truth labels (0=normal, 1=fraud).
        y_scores: Predicted fraud probabilities or risk scores.
        threshold: Classification threshold for converting scores to labels.
    
    Returns:
        dict: Dictionary containing:
            - ROC-AUC: Area under ROC curve (threshold-independent)
            - PR-AUC: Area under Precision-Recall curve (better for imbalanced data)
            - Precision: True positives / predicted positives
            - Recall: True positives / actual positives (fraud detection rate)
            - F1-Score: Harmonic mean of precision and recall
            - threshold: The threshold used for binary predictions
    
    Note:
        Returns None for metrics that cannot be computed (e.g., if y_true
        contains only one class).
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {}
    
    # ROC-AUC: Threshold-independent ranking metric
    try:
        metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_scores))
    except:
        metrics['ROC-AUC'] = None
    
    # PR-AUC: More informative for imbalanced datasets
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        metrics['PR-AUC'] = float(auc(recall_curve, precision_curve))
    except:
        metrics['PR-AUC'] = None
    
    # Threshold-dependent metrics
    try:
        metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['F1-Score'] = float(f1_score(y_true, y_pred, zero_division=0))
    except:
        metrics['Precision'] = None
        metrics['Recall'] = None
        metrics['F1-Score'] = None
    
    metrics['threshold'] = float(threshold)
    
    return metrics


def save_results(model_name: str, category: str, y_true, y_scores, 
                 metrics: dict = None, timestamp: str = None):
    """
    Save model predictions and evaluation results to disk.
    
    Creates a structured directory with predictions, labels, metrics,
    and metadata for reproducibility and later analysis.
    
    Args:
        model_name: Model identifier (e.g., 'lstm', 'iforest', 'transformer').
        category: Model category ('dl', 'ml', 'occ_dl', 'occ_ml').
        y_true: Ground truth labels.
        y_scores: Predicted risk scores.
        metrics: Pre-computed metrics dict. If None, computed automatically
                 using 80th percentile threshold.
        timestamp: Experiment timestamp. If None, uses current time.
    
    Returns:
        str: Path to the saved model directory.
    
    Saved Files:
        - scores.npy: Numpy array of risk scores
        - labels.npy: Numpy array of ground truth labels
        - metrics.json: Evaluation metrics dictionary
        - meta.json: Experiment metadata (sample counts, score statistics)
    """
    run_dir = get_run_dir(category, timestamp)

    model_dir = os.path.join(run_dir, model_name.lower().replace('-', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 1. Save predictions and labels
    np.save(os.path.join(model_dir, 'scores.npy'), y_scores)
    np.save(os.path.join(model_dir, 'labels.npy'), y_true)
    
    # 2. Save metrics
    if metrics is None:
        threshold = np.percentile(y_scores, 80)
        metrics = compute_metrics(y_true, y_scores, threshold)
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 3. Save metadata for reproducibility
    meta = {
        'model_name': model_name,
        'category': category,
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(np.sum(y_true == 0)),
        'score_min': float(np.min(y_scores)),
        'score_max': float(np.max(y_scores)),
        'score_mean': float(np.mean(y_scores)),
    }
    
    with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    
    print(f"[Save] Results saved to: {model_dir}")
    return model_dir


def print_evaluation_report(model_name, y_true, y_scores, threshold=0.5, 
                            category=None, save=True, timestamp=None):
    """
    Print formatted evaluation report and optionally save results.
    
    Outputs a comprehensive report including key metrics, confusion matrix,
    and sklearn classification report. Designed for easy comparison across
    different models and experiments.
    
    Args:
        model_name: Model identifier for display.
        y_true: Ground truth labels.
        y_scores: Predicted risk scores.
        threshold: Classification threshold for binary predictions.
        category: If provided with save=True, saves results to this category.
        save: Whether to save results to disk.
        timestamp: Experiment timestamp for result organization.
    
    Returns:
        dict: Computed metrics dictionary.
    
    Output Format:
        - Key metrics (Recall, PR-AUC, F1, ROC-AUC, Precision)
        - Confusion matrix with Normal/Fraud labels
        - Full sklearn classification report
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = compute_metrics(y_true, y_scores, threshold)
    
    print("\n" + "=" * 60)
    print(f"  📊 Evaluation Report: {model_name}")
    print("=" * 60)
    
    # Print key metrics
    print(f"\n  [Key Metrics] (Threshold: {threshold:.4f})")
    print(f"  {'─' * 40}")
    print(f"  {'Recall (Fraud)':<20}: {metrics['Recall']:.4f}" if metrics['Recall'] else "  Recall: N/A")
    print(f"  {'PR-AUC':<20}: {metrics['PR-AUC']:.4f}" if metrics['PR-AUC'] else "  PR-AUC: N/A")
    print(f"  {'F1-Score (Fraud)':<20}: {metrics['F1-Score']:.4f}" if metrics['F1-Score'] else "  F1-Score: N/A")
    print(f"  {'ROC-AUC':<20}: {metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] else "  ROC-AUC: N/A")
    print(f"  {'Precision (Fraud)':<20}: {metrics['Precision']:.4f}" if metrics['Precision'] else "  Precision: N/A")
    
    # Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  [Confusion Matrix]")
        print(f"  {'─' * 40}")
        print(f"                  Predicted")
        print(f"                  Normal  Fraud")
        print(f"  Actual Normal   {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"  Actual Fraud    {cm[1][0]:>6}  {cm[1][1]:>6}")
    except:
        pass
    
    # Classification Report
    try:
        print(f"\n  [Classification Report]")
        print(f"  {'─' * 40}")
        report = classification_report(y_true, y_pred, target_names=['Normal', 'Fraud'], digits=4)
        for line in report.split('\n'):
            print(f"  {line}")
    except:
        pass
    
    print("=" * 60 + "\n")
    
    # Save results if category provided
    if save and category:
        save_results(model_name, category, y_true, y_scores, metrics, timestamp=timestamp)

    return metrics

