"""
LSTM with Time Delta Context for Sequence-Based Fraud Detection

Convenience module that provides an LSTM variant of the TimeContextRNNBase.
Inherits from TimeContextRNNBase with rnn_type='LSTM'.

See rnn_with_time_model.py for full architecture documentation.
"""

from models.rnn_with_time_model import TimeContextRNNBase


class LSTMWithTime(TimeContextRNNBase):
    """
    LSTM with time delta context for fraud classification.
    
    Uses LSTM cells within the TimeContextRNNBase architecture,
    combining feature projections with time delta embeddings.
    
    Args:
        input_size: Number of input features per time step.
        hidden_size: LSTM hidden dimension (default: 128).
        device: CUDA device for computation.
    """
    
    def __init__(self, input_size, hidden_size=128, device='cuda:0'):
        super().__init__(input_size, hidden_size, device=device, rnn_type='LSTM')

