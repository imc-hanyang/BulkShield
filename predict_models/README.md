# BulkShield Fraud Detection Models

This repository contains machine learning and deep learning models for detecting fraudulent ticket refund patterns in train reservation systems.

## Overview

BulkShield implements a multi-model fraud detection pipeline that analyzes user transaction sequences to identify suspicious refund patterns. The system combines:

- **Deep Learning Models**: Transformer, LSTM, RNN, GRU with attention mechanisms
- **Traditional ML Models**: HMM, ARIMA, SARIMA
- **One-Class Classification**: Isolation Forest, One-Class SVM, LOF
- **Rule-Based Baselines**: Threshold-based detection using refund statistics
- **LLM Integration**: Explainable fraud detection using Llama 3.1-70B

## Directory Structure

```
predict_models/
├── models/                         # Model definitions
│   ├── transformer_model.py        # Transformer classifier with attention
│   ├── lstm_model.py               # LSTM classifier
│   ├── rnn_model.py                # RNN classifier
│   ├── gru_model.py                # GRU classifier
│   └── simple_models.py            # Logistic Regression, SVR, MLP
│
├── # Core Utilities
├── data_utils.py                   # Data loading, Dataset classes, preprocessing
├── eval_utils.py                   # Evaluation metrics, result saving
├── dl_trainer.py                   # Deep learning training loop
├── prompt_schema.py                # LLM prompt column aliases
├── used_raw_columns.py             # Raw column definitions
│
├── # Main Training Scripts
├── transformer_main.py             # Transformer model training
├── lstm_main.py                    # LSTM model training
├── gmm_rnn_gru_main.py             # RNN and GRU model training
├── simple_models_main.py           # Simple models training
│
├── # LLM Integration
├── build_prompts_attn_jsonl.py     # Attention-based prompt generation
├── run_vllm_generate_from_jsonl.py # vLLM inference execution
├── transformer_result_attention_prompting.py # Attention prompting
│
├── # Analysis & Utilities
├── analyze_threshold.py            # Threshold analysis tool
├── extract_gmm_samples.py          # Sample extraction
└── ml_results/                     # Saved models and results
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch numpy scikit-learn tqdm pandas

# For LLM integration
pip install vllm transformers accelerate

# Optional
pip install statsmodels hmmlearn
```

### Hardware Requirements

- **Recommended**: NVIDIA H100 GPU with 80GB VRAM
- **Minimum**: NVIDIA GPU with 24GB VRAM (for smaller models)
- **CPU Cores**: 32+ cores recommended for data loading parallelization

## Configuration

### HuggingFace Token Setup (Required for LLM)

> ⚠️ **IMPORTANT**: Before using LLM integration scripts, you must set your HuggingFace token.

**Files requiring token configuration:**
- `build_prompts_attn_jsonl.py` (line 47)
- `transformer_result_attention_prompting.py` (line 41)

**Option 1: Environment Variable (Recommended)**
```bash
export HF_TOKEN="your_huggingface_token_here"
```

**Option 2: Update in code**
```python
# Replace VALID_TOKEN with your token
VALID_TOKEN = "hf_your_actual_token_here"
```

To obtain a HuggingFace token:
1. Create account at [huggingface.co](https://huggingface.co)
2. Request access to [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
3. Generate token at Settings → Access Tokens

## Data Format

### Sequence Data
Each user's transaction sequence is stored as a CSV file:

```
sequence_data_28d_train/
└── group_XX/
    └── seq_{user_id}_{date}.csv
```

**Required columns:**
- `timestamp`: Transaction datetime
- `seat_cnt`, `buy_amt`, `refund_amt`: Transaction amounts
- `route_dep_station`, `route_arr_station`: Route information
- See `used_raw_columns.py` for full column list

### Label Data
Per-user binary labels (0: Normal, 1: Fraud):

```
label/
├── gmm_label_train_per_user/
├── gmm_label_test_per_user/
├── rulebased_label_train_per_user/
└── rulebased_label_test_per_user/
```

## Usage

### 1. Train Transformer Model

```bash
# Main Transformer with attention
python transformer_main.py

# Combined Transformer architecture
python transformer_combined_main.py
```

### 2. Train RNN-family Models with Time Context

```bash
python rnn_with_time_main.py
python lstm_with_time_main.py
python gru_with_time_main.py
```

### 3. Train Traditional ML Models

```bash
python ml_main.py  # HMM, ARIMA, SARIMA
```

### 4. Train One-Class Classification Models

```bash
python occ_dl_main.py   # LSTM-AE, Deep SVDD
python occ_ml_main.py   # iForest, OC-SVM, LOF
```

### 5. Generate LLM Explanations

```bash
# Step 1: Build prompts with attention weights
CUDA_VISIBLE_DEVICES=0 python build_prompts_attn_jsonl.py

# Step 2: Run LLM inference
python run_vllm_generate_from_jsonl.py
```

## Model Architectures

### Transformer Classifier
- Multi-head self-attention with CLS token pooling
- Categorical embedding + Numeric projection + Time delta embedding
- Mixed precision training (AMP) for H100 optimization

## Evaluation Metrics

All models are evaluated using:
- **Macro F1 Score**: Primary metric for model selection
- **Precision / Recall**: Per-class performance
- **ROC-AUC / PR-AUC**: Threshold-independent metrics
- **Precision@K / Recall@K**: Top-K analysis

Results are automatically saved to `ml_results/{category}/{timestamp}/`.

## Rule-Based Detection Logic

Baseline detection uses simple threshold rules:
- **Refund Amount**: >= 1,000,000 KRW in 1-month window
- **Refund Rate**: >= 90% of total tickets refunded

## Output Structure

```
ml_results/
├── transformer/
│   └── 2024-01-15_10-30/
│       ├── transformer_best.pth      # Best model checkpoint
│       ├── vocabs.json               # Vocabulary mappings
│       ├── scores.npy                # Prediction scores
│       ├── labels.npy                # Ground truth labels
│       └── meta.json                 # Training metadata
└── latest -> 2024-01-15_10-30/       # Symlink to latest run
```

## Performance Optimization

### H100 GPU Optimization
- Batch size: 2048-4096
- Num workers: 32-60 (for data loading)
- Mixed precision training enabled
- Persistent workers for DataLoader

### Memory Efficiency
- Lazy loading via `LazyDataset`
- Chunked data processing
- GPU memory clearing between model runs

## License

This project is developed for academic research purposes.

## Citation

If you use this code in your research, please cite:
```
[Your paper citation here]
```
