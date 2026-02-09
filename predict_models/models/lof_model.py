"""
Local Outlier Factor (LOF) for One-Class Anomaly Detection

This module implements a Local Outlier Factor wrapper for detecting fraudulent
transactions as anomalies. LOF measures the local deviation of density of a
given sample with respect to its neighbors, identifying samples that have
substantially lower density than their neighbors.

Key Concepts:
    - Compares local density of a point to its neighbors' densities
    - Points with much lower density are classified as outliers
    - novelty=True enables prediction on new unseen data
    - No training labels needed (unsupervised)

Reference:
    Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. "LOF: identifying
    density-based local outliers." ACM Sigmod Record, 2000.

Usage:
    >>> model = LOFModel(n_neighbors=20, contamination=0.1)
    >>> model.fit(train_features)  # [N, F] numpy array
    >>> scores = model.predict_score(test_features)
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LOFModel:
    """
    Local Outlier Factor wrapper for anomaly-based fraud detection.
    
    Args:
        n_neighbors: Number of neighbors for LOF calculation (default: 20).
        contamination: Expected proportion of anomalies (default: 0.1).
        n_jobs: Number of parallel jobs (default: -1, use all CPUs).
    
    Note:
        novelty=True is required to use predict/decision_function on new data.
        Without it, LOF can only be used for training data outlier detection.
    """
    
    def __init__(self, n_neighbors=20, contamination=0.1, n_jobs=-1):
        # novelty=True enables prediction on new (test) data
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=n_jobs
        )

    def fit(self, train_data: np.ndarray):
        """
        Fit LOF on training data.
        
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