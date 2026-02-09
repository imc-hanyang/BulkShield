"""
Isolation Forest for One-Class Anomaly Detection

This module implements an Isolation Forest wrapper for detecting fraudulent
transactions as anomalies. Isolation Forest isolates observations by randomly
selecting a feature and then randomly selecting a split value between the
maximum and minimum values of the selected feature.

Key Concepts:
    - Anomalies are isolated faster (fewer splits) than normal points
    - No training labels needed (unsupervised)
    - Returns anomaly score (higher = more anomalous)

Reference:
    Liu, Fei Tony, Ting, Kai Ming, and Zhou, Zhi-Hua. "Isolation forest."
    Data Mining, 2008. ICDM'08.

Usage:
    >>> model = IForestModel(n_estimators=100, contamination=0.1)
    >>> model.fit(train_features)  # [N, F] numpy array
    >>> scores = model.predict_score(test_features)
"""

import numpy as np
import torch
from sklearn.ensemble import IsolationForest


class IForestModel:
    """
    Isolation Forest wrapper for anomaly-based fraud detection.
    
    Args:
        n_estimators: Number of isolation trees (default: 100).
        contamination: Expected proportion of anomalies (default: 0.1).
        n_jobs: Number of parallel jobs (default: -1, use all CPUs).
    """
    
    def __init__(self, n_estimators=100, contamination=0.1, n_jobs=-1):
        self.model = IsolationForest(
            n_estimators=n_estimators, 
            contamination=contamination, 
            n_jobs=n_jobs, 
            random_state=42
        )

    def fit(self, train_data: np.ndarray):
        """
        Fit Isolation Forest on training data.
        
        Args:
            train_data: Feature array of shape [N, F].
        """
        self.model.fit(train_data)

    def predict_score(self, test_data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for test data.
        
        Args:
            test_data: Feature array of shape [N, F].
            
        Returns:
            Anomaly scores (negated decision function, higher = more anomalous).
        """
        # decision_function returns: higher values = more normal
        # We negate to get: higher values = more anomalous
        return -self.model.decision_function(test_data)