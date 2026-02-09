"""
One-Class SVM for Large-Scale Anomaly Detection

This module implements a scalable One-Class SVM using SGD optimization
with Nystroem kernel approximation. Standard One-Class SVM (LibSVM) has
O(N^2) complexity which is infeasible for 7M+ samples. This implementation
uses linear-time SGD with kernel approximation for scalability.

Architecture:
    1. StandardScaler: Normalize features to zero mean, unit variance
    2. Nystroem: Approximate RBF kernel using random feature maps
    3. SGDOneClassSVM: Linear-time one-class classification

Scalability Notes:
    - Standard OneClassSVM: O(N^2) memory and time, crashes on large data
    - This implementation: O(N) time complexity, can handle 7M+ samples
    - Nystroem approximation provides RBF-like non-linear boundary

Reference:
    Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & 
    Williamson, R. C. "Estimating the support of a high-dimensional 
    distribution." Neural computation, 2001.

Usage:
    >>> model = OCSVMModel(nu=0.1)
    >>> model.fit(train_features)  # [N, F] numpy array, N can be millions
    >>> scores = model.predict_score(test_features)
"""

import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class OCSVMModel:
    """
    Scalable One-Class SVM for anomaly-based fraud detection.
    
    Uses SGDOneClassSVM with Nystroem kernel approximation to handle
    large datasets (7M+ samples) that would crash standard LibSVM.
    
    Args:
        nu: Expected upper bound on fraction of outliers (default: 0.1).
            Also serves as lower bound on fraction of support vectors.
    
    Note:
        The Nystroem transformer uses n_components=100 random features
        to approximate an RBF kernel with gamma=0.1.
    """
    
    def __init__(self, nu=0.1):
        # Pipeline: Scale -> Kernel Approximation -> Linear One-Class SVM
        self.pipeline = make_pipeline(
            StandardScaler(),
            Nystroem(gamma=0.1, random_state=42, n_components=100),  # Approx RBF kernel
            SGDOneClassSVM(nu=nu, random_state=42)
        )

    def fit(self, train_data: np.ndarray):
        """
        Fit One-Class SVM on training data.
        
        Args:
            train_data: Feature array of shape [N, F].
                        Can handle large N (7M+ samples).
        """
        self.pipeline.fit(train_data)

    def predict_score(self, test_data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for test data.
        
        Args:
            test_data: Feature array of shape [N, F].
            
        Returns:
            Anomaly scores (negated decision function, higher = more anomalous).
            
        Note:
            SGDOneClassSVM decision_function: >0 (inlier), <0 (outlier).
            We negate so that higher values indicate more anomalous samples.
        """
        return -self.pipeline.decision_function(test_data)