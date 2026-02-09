
import re
import os
import collections

def parse_log_per_model(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {}

    # Dictionary to store best info per model type (rnn, gru, lstm, etc.)
    # Key: model_type (str), Value: dict of metrics
    best_metrics = {}

    current_model_full = "Unknown"
    current_model_type = "Unknown"
    current_epoch = "Unknown"
    current_roc_auc = "N/A"
    last_seen_epoch = None
    
    # regex
    re_report = re.compile(r"Evaluation Report:\s+(.*)") 
    # re_model_type tries to catch 'rnn', 'gru', 'lstm' from the full model string
    re_roc = re.compile(r"ROC-AUC\s+:\s+([\d\.]+)")
    re_macro = re.compile(r"macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # 1. Detect Report Header
            match_header = re_report.search(line)
            if match_header:
                model_str = match_header.group(1).strip()
                current_model_full = model_str
                
                # Determine model type
                lower_name = model_str.lower()
                if 'rnn' in lower_name:
                    current_model_type = 'rnn'
                elif 'gru' in lower_name:
                    current_model_type = 'gru'
                elif 'lstm' in lower_name:
                    current_model_type = 'lstm'
                else:
                    current_model_type = 'unknown'

                # Extract epoch if present in name
                m_ep = re.search(r"epoch(\d+)", lower_name)
                if m_ep:
                    current_epoch = m_ep.group(1)
                else:
                    current_epoch = "Unknown"
                
                current_roc_auc = "N/A"
                if current_epoch == "Unknown" and last_seen_epoch:
                    current_epoch = last_seen_epoch
                continue

            # 2. Extract Epoch from standalone line
            match_epoch_line = re.search(r"(?:\[Epoch|Epoch)\s+(\d+)", line)
            if match_epoch_line:
                last_seen_epoch = match_epoch_line.group(1)
            
            # 3. ROC-AUC
            match_roc = re_roc.search(line)
            if match_roc:
                current_roc_auc = float(match_roc.group(1))
                continue
                
            # 4. Macro Avg
            match_macro = re_macro.search(line)
            if match_macro:
                p_macro = float(match_macro.group(1))
                r_macro = float(match_macro.group(2))
                f1_macro = float(match_macro.group(3))
                
                # Initialize if not present
                if current_model_type not in best_metrics:
                    best_metrics[current_model_type] = {
                        "f1_macro": -1.0
                    }
                
                # Update if better
                if f1_macro > best_metrics[current_model_type]["f1_macro"]:
                    best_metrics[current_model_type] = {
                        "file": filepath,
                        "model_type": current_model_type,
                        "full_model_name": current_model_full,
                        "epoch": current_epoch,
                        "f1_macro": f1_macro,
                        "roc_auc": current_roc_auc,
                        "precision_macro": p_macro,
                        "recall_macro": r_macro
                    }

    return best_metrics

print(f"{'Log File':<35} | {'Type':<5} | {'Full Name':<15} | {'Epoch':<5} | {'F1 Macro':<8} | {'ROC-AUC':<8} | {'Precision':<8} | {'Recall':<8}")
print("-" * 120)

path_list = [
    "logs/gmm_rnn_gru.log",
    "logs/rulebased_rnn_gru.log"
]

for fpath in path_list:
    results = parse_log_per_model(fpath)
    base_name = os.path.basename(fpath)
    
    if not results:
        print(f"{base_name:<35} | No Data")
        continue
        
    # Print RNN result specifically if requested, or all found?
    # User asked for RNN specifically. Let's print all found types to be safe and clear.
    for m_type, info in results.items():
        if m_type == 'rnn': # Highlight or just print
             print(f"{base_name:<35} | {m_type:<5} | {info['full_model_name']:<15} | {info['epoch']:<5} | {info['f1_macro']:<8} | {info['roc_auc']:<8} | {info['precision_macro']:<8} | {info['recall_macro']:<8}")
        else:
             # Also print others? The user specifically asked for RNN.
             # But might be useful to see if RNN is missing.
             # "이 두 파일에서 rnn 모데에 대한..."
             pass

