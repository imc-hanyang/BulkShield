"""
Data Utilities - Dataset Classes and Data Loading Functions

This module provides data loading utilities and PyTorch Dataset implementations
for the SRT (Korea Train Express) fraud detection system. It handles user transaction
sequences stored as per-user CSV files and provides efficient lazy-loading mechanisms
to handle large-scale datasets without memory overflow.

Key Components:
    - LazyDataset: Memory-efficient dataset that loads files on-demand
    - TimeContextDataset: Dataset with temporal context (time delta) features
    - CombinedSequenceDataset: Dataset for combined feature sequences
    - Data loading functions: load_train_data(), load_test_data()

Data Structure:
    - Labels: Per-user label files indicating fraud (is_risk=1) or normal (is_risk=0)
    - Sequences: Per-user transaction sequence files with 28-day event windows

Usage:
    >>> from data_utils import load_train_data, LazyDataset, pad_collate_fn
    >>> train_paths, train_labels = load_train_data()
    >>> dataset = LazyDataset(train_paths, train_labels)
    >>> loader = DataLoader(dataset, batch_size=64, collate_fn=pad_collate_fn)
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

import used_raw_columns
from tqdm import tqdm
import re
import datetime
import math

# =============================================================================
# Data Path Configuration
# =============================================================================
# Base directories for label and feature data
BASE_LABEL_PATH = "/home/srt/Dataset/label"
BASE_SEQUENCE_PATH = "/home/srt/Dataset/feature"

# Training data paths
TRAIN_LABEL_PATH = os.path.join(BASE_LABEL_PATH, "label_train_per_user")
# Alternative: Rule-based labels
# TRAIN_LABEL_PATH = os.path.join(BASE_LABEL_PATH, "rulebased_label_train_per_user")
TRAIN_SEQUENCE_PATH = os.path.join(BASE_SEQUENCE_PATH, "sequence_data_28d_train")

# Test data paths
TEST_LABEL_PATH = os.path.join(BASE_LABEL_PATH, "label_test_per_user")
# Alternative: Rule-based labels
# TEST_LABEL_PATH = os.path.join(BASE_LABEL_PATH, "rulebased_label_test_per_user")
TEST_SEQUENCE_PATH = os.path.join(BASE_SEQUENCE_PATH, "sequence_data_28d_test")


class LazyDataset(Dataset):
    """
    Memory-efficient Lazy Loading Dataset for variable-length sequences.
    
    This dataset implements on-demand file loading to handle large-scale data
    without loading all files into memory upfront. Each file is read only when
    the corresponding sample is requested via __getitem__.
    
    Attributes:
        input_paths (list): List of file paths to sequence CSV files.
        labels (list): Corresponding fraud labels (0=normal, 1=fraud).
    
    Note:
        Uses csv.DictReader instead of pandas to avoid memory fragmentation
        and potential segfaults with large datasets.
    """
    
    def __init__(self, input_paths, labels):
        """
        Initialize LazyDataset with file paths and labels.
        
        Args:
            input_paths: List of paths to sequence CSV files.
            labels: List of integer labels (0 or 1) for each user.
        """
        self.input_paths = input_paths
        self.labels = labels

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        """
        Load and return a single sample.
        
        Args:
            idx: Index of the sample to load.
            
        Returns:
            tuple: (sequence_tensor, label) where sequence_tensor is [T, F]
                   with T=sequence length and F=number of features.
        """
        f_path = self.input_paths[idx]
        label = self.labels[idx]
        
        try:
            # Use csv module instead of pandas to avoid memory issues
            data_rows = []
            with open(f_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                target_cols = used_raw_columns.columns
                
                for row in reader:
                    # Extract only required columns and convert to float
                    row_vals = []
                    for col in target_cols:
                        val_str = row.get(col, '0')
                        try:
                            val = float(val_str)
                        except:
                            val = 0.0
                        row_vals.append(val)
                    data_rows.append(row_vals)
            
            if not data_rows:
                return torch.zeros((1, len(used_raw_columns.columns))), label

            tensor_x = torch.tensor(data_rows, dtype=torch.float32)
            return tensor_x, label
            
        except Exception:
            # Return empty tensor on error
            return torch.zeros((1, len(used_raw_columns.columns))), label


def pad_collate_fn(batch):
    """
    Collate function for DataLoader that handles variable-length sequences.
    
    Pads sequences to the same length within a batch and returns original
    sequence lengths for use with pack_padded_sequence.
    
    Args:
        batch: List of (sequence_tensor, label) tuples from LazyDataset.
        
    Returns:
        tuple: (padded_inputs, targets, lengths) where:
            - padded_inputs: [B, max_T, F] padded sequence tensor
            - targets: [B] label tensor
            - lengths: [B] original sequence lengths for each sample
    """
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.tensor(targets)
    return inputs_padded, targets, lengths


# =============================================================================
# Utility Functions
# =============================================================================

def extract_user_id_from_filename(filename: str) -> str:
    """
    Extract the 64-character hex user ID from a filename.
    
    User IDs are SHA-256 hashes used for anonymization.
    
    Args:
        filename: Filename containing the user ID.
        
    Returns:
        64-character hex string if found, None otherwise.
    """
    match = re.search(r'([a-f0-9]{64})', filename)
    if match:
        return match.group(1)
    return None


def collect_files_from_groups(base_folder: str, file_prefix: str) -> dict:
    """
    Scan grouped directories and collect file paths by user ID.
    
    Files are organized in group_* subdirectories for parallel processing.
    This function scans all groups and builds a user_id -> file_path mapping.
    
    Args:
        base_folder: Root directory containing group_* subdirectories.
        file_prefix: Prefix to filter files (e.g., 'label_' or 'seq_').
        
    Returns:
        dict: Mapping from user_id to absolute file path.
    """
    file_map = {}
    if not os.path.exists(base_folder):
        return {}
    
    group_dirs = [d for d in os.listdir(base_folder) if d.startswith('group_')]
    for group_dir in tqdm(group_dirs, desc=f"Scanning {os.path.basename(base_folder)}"):
        group_path = os.path.join(base_folder, group_dir)
        try:
            files = os.listdir(group_path)
            for f in files:
                if f.startswith(file_prefix) and f.endswith('.csv'):
                    user_id = extract_user_id_from_filename(f)
                    if user_id:
                        file_map[user_id] = os.path.join(group_path, f)
        except Exception:
            pass
    return file_map


import csv

def _get_file_lists(label_folder: str, sequence_folder: str):
    """
    Collect file paths and labels without loading full data into memory.
    
    This function scans the label and sequence directories, matches users
    that have both files available, and reads only the label values.
    Uses csv module instead of pandas to prevent segmentation faults
    with large datasets.
    
    Args:
        label_folder: Directory containing per-user label files.
        sequence_folder: Directory containing per-user sequence files.
        
    Returns:
        tuple: (input_paths, labels) where:
            - input_paths: List of sequence file paths
            - labels: List of corresponding fraud labels (0 or 1)
    """
    print(f"Collecting file paths from: {label_folder}")
    
    label_files = collect_files_from_groups(label_folder, 'label_')
    seq_files = collect_files_from_groups(sequence_folder, 'seq_')
    
    # Find users that have both label and sequence files
    common_users = sorted(list(set(label_files.keys()) & set(seq_files.keys())))
    print(f"  - Matched Users: {len(common_users)}")
    
    input_paths = []
    labels = []
    
    # Pre-read labels into memory (single int per user, minimal memory overhead)
    for user_id in tqdm(common_users, desc="Reading Labels"):
        try:
            lbl_path = label_files[user_id]
            
            # Use csv module for safe file reading
            with open(lbl_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only need first row
                    is_risk = int(row['is_risk'])
                    
                    input_paths.append(seq_files[user_id])
                    labels.append(is_risk)
                    break
        except:
            pass
            
    return input_paths, labels


def load_train_data():
    """
    Load training data file paths and labels.
    
    Returns:
        tuple: (input_paths, labels) for training set.
    """
    return _get_file_lists(TRAIN_LABEL_PATH, TRAIN_SEQUENCE_PATH)


def load_test_data():
    """
    Load test data file paths and labels.
    
    Returns:
        tuple: (input_paths, labels) for test set.
    """
    return _get_file_lists(TEST_LABEL_PATH, TEST_SEQUENCE_PATH)


# =============================================================================
# Feature Column Definitions
# Shared across TimeContextDataset, CombinedSequenceDataset, and time-context models
# =============================================================================

# Numeric features (31 columns) - continuous values that are log1p-scaled
NUMERIC_COLS = [
    # Transaction attributes
    "seat_cnt",               # Number of seats in transaction
    "buy_amt",                # Purchase amount (KRW)
    "refund_amt",             # Refund amount (KRW)
    "cancel_fee",             # Cancellation fee (KRW)
    "route_dist_km",          # Trip distance in kilometers
    
    # Time-related features
    "travel_time",            # Trip duration (minutes)
    "lead_time_buy",          # Time until departure at purchase (minutes)
    "lead_time_ref",          # Time until departure at refund (minutes)
    "hold_time",              # Ticket holding duration (minutes)
    "dep_hour",               # Departure hour (0-23)
    
    # Route purchase history
    "route_buy_cnt",          # Same-route purchases in window
    "fwd_dep_hour_median",    # Forward ticket departure hour (median)
    "rev_buy_cnt",            # Reverse-route purchases in window
    "rev_ratio",              # Reverse-to-forward purchase ratio
    
    # Completed trip statistics
    "completed_fwd_cnt",                  # Completed forward trips
    "completed_fwd_dep_interval_median",  # Forward departure interval (median)
    "completed_fwd_dep_hour_median",      # Forward departure hour (median)
    "completed_rev_cnt",                  # Completed reverse trips
    "completed_rev_dep_interval_median",  # Reverse departure interval (median)
    "completed_rev_dep_hour_median",      # Reverse departure hour (median)
    "unique_route_cnt",                   # Unique routes in window
    
    # Active ticket features
    "rev_dep_hour_median",    # Active reverse ticket departure hour (median)
    "rev_return_gap",         # Gap between arrival and reverse departure (minutes)
    
    # Fraud signal features
    "overlap_cnt",            # Overlapping time tickets count
    "same_route_cnt",         # Same-route active tickets count
    "rev_route_cnt",          # Reverse-route active tickets count
    "repeat_interval",        # Same-route re-purchase interval (median)
    "adj_seat_refund_flag",   # Adjacent seat refund indicator (0/1)
    
    # Refund history
    "recent_ref_cnt",         # Recent same-route refunds count
    "recent_ref_amt",         # Recent same-route refund amount
    "recent_ref_rate",        # Recent same-route refund ratio
]

# Categorical features (6 columns) - discrete values used as embeddings or raw integers
CATEGORICAL_COLS = [
    "dep_station_id",         # Departure station code
    "arr_station_id",         # Arrival station code
    "route_id",               # Route identifier (dep-arr pair hash)
    "train_id",               # Train number
    "action_type",            # Transaction type (1=purchase, 2=refund)
    "dep_dow",                # Departure day of week (0=Mon, 6=Sun)
]

# Supported timestamp formats for parsing
TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
]


def fast_parse_ts(ts_str: str):
    """
    Parse timestamp string with multiple format support.
    
    Tries multiple common datetime formats for robustness.
    
    Args:
        ts_str: Timestamp string to parse.
        
    Returns:
        datetime object if parsing succeeds, None otherwise.
    """
    if not ts_str:
        return None
    for fmt in TS_FORMATS:
        try:
            return datetime.datetime.strptime(ts_str, fmt)
        except:
            continue
    return None

class TimeContextDataset(Dataset):
    """
    Dataset with temporal context features for RNN/LSTM/GRU models.
    
    This dataset extends the basic sequence loading with time delta features.
    Events are sorted by timestamp, and the time difference between consecutive
    events is computed and discretized into buckets for embedding.
    
    The time delta captures the temporal dynamics of user behavior, which is
    critical for detecting patterns like rapid-fire transactions.
    
    Attributes:
        input_paths (list): Paths to sequence CSV files.
        labels (list): Fraud labels (0=normal, 1=fraud).
        max_len (int): Maximum sequence length (truncates from start).
        delta_bucket_size (int): Minutes per bucket for time discretization.
        delta_max_bucket (int): Maximum bucket index (caps the delta).
        feature_cols (list): Combined numeric and categorical column names.
    """
    
    def __init__(self, input_paths, labels, max_len=512, delta_bucket_size=10, delta_max_bucket=288):
        """
        Initialize TimeContextDataset.
        
        Args:
            input_paths: List of sequence file paths.
            labels: List of fraud labels.
            max_len: Maximum sequence length (default: 512).
            delta_bucket_size: Minutes per time bucket (default: 10).
            delta_max_bucket: Maximum bucket value (default: 288 = 48 hours).
        """
        self.input_paths = input_paths
        self.labels = labels
        self.max_len = max_len
        self.delta_bucket_size = delta_bucket_size
        self.delta_max_bucket = delta_max_bucket
        
        # Combine all columns for input features
        self.feature_cols = NUMERIC_COLS + CATEGORICAL_COLS

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        """
        Load a sample with time delta features.
        
        Args:
            idx: Sample index.
            
        Returns:
            tuple: (features, delta_indices, label) where:
                - features: [T, F] tensor of scaled features
                - delta_indices: [T] tensor of time bucket indices
                - label: Scalar label tensor
        """
        path = self.input_paths[idx]
        label = self.labels[idx]
        
        data_rows = []
        try:
            with open(path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # Parse timestamp for sorting and delta calculation
                    ts_val = None
                    if "timestamp" in r:
                        ts_val = fast_parse_ts(r["timestamp"])
                        
                    # Extract and scale features
                    feats = []
                    
                    # Numeric features with log1p scaling
                    for c in NUMERIC_COLS:
                        val = self.safe_float(r.get(c, 0))
                        val_scaled = math.log1p(abs(val))
                        if val < 0:
                            val_scaled = -val_scaled
                        feats.append(val_scaled)
                    
                    # Categorical features (raw integer values)
                    for c in CATEGORICAL_COLS:
                        val = self.safe_float(r.get(c, 0))
                        feats.append(val)
                        
                    data_rows.append({"_ts": ts_val, "feats": feats})
        except:
            pass
            
        if not data_rows:
            # Return empty tensors for missing/empty files
            return (
                torch.zeros(1, len(self.feature_cols)),
                torch.zeros(1, dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
            )
            
        # Sort events by timestamp (chronological order)
        max_ts = datetime.datetime.max
        data_rows.sort(key=lambda x: x["_ts"] if x["_ts"] else max_ts)
        
        # Truncate to max_len (keep most recent events)
        if len(data_rows) > self.max_len:
            data_rows = data_rows[-self.max_len:]
            
        T = len(data_rows)
        timestamps = [r["_ts"] for r in data_rows]
        features = [r["feats"] for r in data_rows]
        
        # Calculate time delta buckets between consecutive events
        delta_vals = [0] * T
        if T > 1 and any(timestamps):
            delta_vals[0] = 0  # First event has no previous event
            for i in range(1, T):
                t_curr = timestamps[i]
                t_prev = timestamps[i-1]
                if t_curr and t_prev:
                    diff_min = (t_curr - t_prev).total_seconds() / 60.0
                else:
                    diff_min = 0.0
                
                # Clamp to [0, 48 hours] and discretize into buckets
                diff_min = max(0.0, min(diff_min, 60.0 * 48))
                bucket = int(diff_min // self.delta_bucket_size)
                bucket = max(0, min(bucket, self.delta_max_bucket))
                delta_vals[i] = bucket
                
        return (
            torch.tensor(features, dtype=torch.float32), 
            torch.tensor(delta_vals, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def safe_float(self, x):
        """Safely convert value to float, returning 0.0 on failure."""
        try:
            return float(x)
        except:
            return 0.0

def collate_fn_with_time(batch):
    """
    Collate function for TimeContextDataset batches.
    
    Pads variable-length sequences and sorts by length for efficient
    packing in RNN-based models.
    
    Args:
        batch: List of (features, delta, label) tuples from TimeContextDataset.
        
    Returns:
        tuple: (feats_pad, deltas_pad, lengths, labels) where:
            - feats_pad: [B, max_T, F] padded feature tensor
            - deltas_pad: [B, max_T] padded time delta tensor
            - lengths: [B] original sequence lengths (descending order)
            - labels: [B] label tensor
    """
    # Sort by length for pack_padded_sequence (descending order)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    feats = [x[0] for x in batch]
    deltas = [x[1] for x in batch]
    labels = [x[2] for x in batch]
    lengths = torch.tensor([x.shape[0] for x in feats], dtype=torch.long)
    
    # Pad sequences to uniform length within batch
    feats_pad = pad_sequence(feats, batch_first=True, padding_value=0)
    deltas_pad = pad_sequence(deltas, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return feats_pad, deltas_pad, lengths, labels

# =============================================================================
# Combined Sequence Dataset (For TransformerCombined)
# =============================================================================

class CombinedSequenceDataset(Dataset):
    """
    Simplified sequence dataset for TransformerCombined model.
    
    Similar to TimeContextDataset but without time delta calculation.
    Suitable for models that don't require explicit temporal encoding
    or that compute their own positional embeddings.
    
    Features:
        - Sorts events by timestamp
        - Applies log1p scaling to numeric features
        - No time delta bucketization
    
    Attributes:
        input_paths (list): Paths to sequence CSV files.
        labels (list): Fraud labels (0=normal, 1=fraud).
        max_len (int): Maximum sequence length.
        feature_cols (list): Combined numeric and categorical column names.
    """
    
    def __init__(self, input_paths, labels, max_len=512):
        """
        Initialize CombinedSequenceDataset.
        
        Args:
            input_paths: List of sequence file paths.
            labels: List of fraud labels.
            max_len: Maximum sequence length (default: 512).
        """
        self.input_paths = input_paths
        self.labels = labels
        self.max_len = max_len
        self.feature_cols = NUMERIC_COLS + CATEGORICAL_COLS

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        """
        Load a sample without time delta features.
        
        Args:
            idx: Sample index.
            
        Returns:
            tuple: (features, label) where:
                - features: [T, F] tensor of scaled features
                - label: Scalar label tensor
        """
        path = self.input_paths[idx]
        label = self.labels[idx]
        
        data_rows = []
        try:
            with open(path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ts_val = None
                    if "timestamp" in r:
                        ts_val = fast_parse_ts(r["timestamp"])
                        
                    feats = []
                    # Numeric features (log1p scaled)
                    for c in NUMERIC_COLS:
                        val = self.safe_float(r.get(c, 0))
                        val_scaled = math.log1p(abs(val))
                        if val < 0:
                            val_scaled = -val_scaled
                        feats.append(val_scaled)
                    
                    # Categorical features (raw integer values)
                    for c in CATEGORICAL_COLS:
                        val = self.safe_float(r.get(c, 0))
                        feats.append(val)
                        
                    data_rows.append({"_ts": ts_val, "feats": feats})
        except:
            pass
            
        if not data_rows:
            return torch.zeros(1, len(self.feature_cols)), torch.tensor(label, dtype=torch.long)
            
        # Sort events by timestamp (chronological order)
        max_ts = datetime.datetime.max
        data_rows.sort(key=lambda x: x["_ts"] if x["_ts"] else max_ts)
        
        # Truncate to max_len (keep most recent events)
        if len(data_rows) > self.max_len:
            data_rows = data_rows[-self.max_len:]
            
        features = [r["feats"] for r in data_rows]
        
        return (
            torch.tensor(features, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.long)
        )

    def safe_float(self, x):
        """Safely convert value to float, returning 0.0 on failure."""
        try:
            return float(x)
        except:
            return 0.0

def collate_fn_combined_seq(batch):
    """
    Collate function for CombinedSequenceDataset batches.
    
    Pads variable-length sequences and returns lengths for optional
    mask creation in Transformer models.
    
    Args:
        batch: List of (features, label) tuples from CombinedSequenceDataset.
        
    Returns:
        tuple: (feats_pad, lengths, labels) where:
            - feats_pad: [B, max_T, F] padded feature tensor
            - lengths: [B] original sequence lengths (descending order)
            - labels: [B] label tensor
    
    Note:
        Sequences are sorted by length in descending order for compatibility
        with pack_padded_sequence. The calling code can use lengths to create
        attention masks (True for padding positions) if needed.
    """
    # Sort by length for efficient packing (descending order)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    feats = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    lengths = torch.tensor([x.shape[0] for x in feats], dtype=torch.long)
    
    # Pad sequences (padding positions will be masked later)
    feats_pad = pad_sequence(feats, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return feats_pad, lengths, labels
