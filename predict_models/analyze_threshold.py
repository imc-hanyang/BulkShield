"""
Threshold Analysis Script

This module loads saved model results and evaluates performance across
various classification thresholds. Provides comprehensive analysis including:
    - ROC-AUC and PR-AUC (threshold-independent)
    - Precision@K and Recall@K for top-K analysis
    - Optimal threshold search for F1/Precision/Recall
    - Interactive threshold testing mode

Usage:
    python analyze_threshold.py --latest         # Analyze latest results
    python analyze_threshold.py --path <path>    # Analyze specific model
    python analyze_threshold.py --path <path> --threshold 0.5
    python analyze_threshold.py --list           # List available results
"""
import os
import json
import numpy as np
import argparse
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

RESULTS_BASE_DIR = "/home/srt/ml_results"


def load_results(model_path: str):
    """저장된 결과 로드"""
    scores = np.load(os.path.join(model_path, 'scores.npy'))
    labels = np.load(os.path.join(model_path, 'labels.npy'))
    
    meta_path = os.path.join(model_path, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        meta = {}
    
    return scores, labels, meta


def evaluate_with_threshold(y_true, y_scores, threshold):
    """특정 threshold로 평가"""
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    return metrics


def evaluate_at_k(y_true, y_scores, k_percent):
    """상위 K%를 Fraud로 분류하여 평가"""
    threshold = np.percentile(y_scores, 100 - k_percent)
    y_pred = (y_scores >= threshold).astype(int)
    
    n_predicted = np.sum(y_pred)
    n_true_positive = np.sum((y_pred == 1) & (y_true == 1))
    
    precision_at_k = n_true_positive / n_predicted if n_predicted > 0 else 0
    recall_at_k = n_true_positive / np.sum(y_true) if np.sum(y_true) > 0 else 0
    
    return {
        'k_percent': k_percent,
        'threshold': threshold,
        'n_predicted': int(n_predicted),
        'n_true_positive': int(n_true_positive),
        'precision@k': precision_at_k,
        'recall@k': recall_at_k,
    }


def find_best_threshold(y_true, y_scores, metric='f1'):
    """최적 threshold 탐색"""
    thresholds = np.percentile(y_scores, np.arange(1, 100))
    best_score = 0
    best_threshold = 0.5
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score


def analyze_model(model_path: str):
    """모델 결과 종합 분석"""
    print(f"\n{'=' * 60}")
    print(f"  📊 Threshold Analysis: {model_path}")
    print(f"{'=' * 60}")
    
    scores, labels, meta = load_results(model_path)
    
    print(f"\n  [데이터 정보]")
    print(f"  총 샘플: {len(labels)}")
    print(f"  Fraud: {np.sum(labels)} ({100*np.mean(labels):.2f}%)")
    print(f"  Score 범위: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    # ROC-AUC, PR-AUC (threshold 무관)
    print(f"\n  [Threshold-Free 메트릭]")
    print(f"  {'─' * 40}")
    try:
        roc_auc = roc_auc_score(labels, scores)
        print(f"  ROC-AUC: {roc_auc:.4f}")
    except:
        print(f"  ROC-AUC: N/A")
    
    try:
        prec, rec, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(rec, prec)
        print(f"  PR-AUC:  {pr_auc:.4f}")
    except:
        print(f"  PR-AUC: N/A")
    
    # Precision@K, Recall@K
    print(f"\n  [Top-K 분석]")
    print(f"  {'─' * 40}")
    print(f"  {'K%':<6} {'Threshold':<12} {'Prec@K':<10} {'Recall@K':<10} {'TP':<8}")
    print(f"  {'─' * 40}")
    
    for k in [1, 2, 5, 10, 20]:
        result = evaluate_at_k(labels, scores, k)
        print(f"  {k:<6} {result['threshold']:<12.4f} {result['precision@k']:<10.4f} {result['recall@k']:<10.4f} {result['n_true_positive']:<8}")
    
    # 최적 Threshold 탐색
    print(f"\n  [최적 Threshold 탐색]")
    print(f"  {'─' * 40}")
    
    for metric in ['f1', 'recall', 'precision']:
        best_t, best_score = find_best_threshold(labels, scores, metric)
        print(f"  Best for {metric:<10}: threshold={best_t:.4f}, {metric}={best_score:.4f}")
    
    # 다양한 Threshold 테스트
    print(f"\n  [Threshold별 성능]")
    print(f"  {'─' * 40}")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'─' * 40}")
    
    for percentile in [50, 60, 70, 80, 90, 95]:
        t = np.percentile(scores, percentile)
        metrics = evaluate_with_threshold(labels, scores, t)
        print(f"  {t:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    print(f"\n{'=' * 60}\n")


def list_available_results():
    """저장된 결과 목록 출력"""
    print("\n📁 Available Results:")
    print("=" * 60)
    
    if not os.path.exists(RESULTS_BASE_DIR):
        print("  No results found.")
        return []
    
    paths = []
    # [변경] 구조: Category -> Timestamp -> Model
    for category in sorted(os.listdir(RESULTS_BASE_DIR)):
        cat_path = os.path.join(RESULTS_BASE_DIR, category)
        if not os.path.isdir(cat_path):
            continue
            
        print(f"\n  📂 [{category}]")
        
        for timestamp in sorted(os.listdir(cat_path), reverse=True):
            if timestamp == 'latest':
                continue
                
            ts_path = os.path.join(cat_path, timestamp)
            if not os.path.isdir(ts_path):
                continue
                
            print(f"     └── 📅 {timestamp}/")
            
            for model in os.listdir(ts_path):
                model_path = os.path.join(ts_path, model)
                if os.path.isdir(model_path):
                    print(f"         └── {model}/")
                    paths.append(model_path)
    
    print("=" * 60)
    return paths



def test_custom_threshold(scores, labels, threshold):
    """특정 Threshold 테스트 및 출력"""
    print(f"\n  🔍 Custom Threshold Test: {threshold}")
    print(f"  {'─' * 40}")
    
    metrics = evaluate_with_threshold(labels, scores, threshold)
    
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    
    # 예측 통계
    y_pred = (scores >= threshold).astype(int)
    n_fraud_pred = np.sum(y_pred)
    n_fraud_real = np.sum(labels)
    n_correct = np.sum((y_pred == 1) & (labels == 1))
    
    print(f"  Predicted Fraud: {n_fraud_pred} / {len(labels)} ({(n_fraud_pred/len(labels))*100:.2f}%)")
    print(f"  Actual Caught  : {n_correct} / {n_fraud_real} (Recall: {n_correct/n_fraud_real:.4f})")
    print(f"  {'─' * 40}")


def main():
    parser = argparse.ArgumentParser(description="Threshold Analysis Tool")
    parser.add_argument('--path', type=str, help='특정 모델 결과 경로')
    parser.add_argument('--latest', action='store_true', help='최신 결과 전체 분석')
    parser.add_argument('--list', action='store_true', help='저장된 결과 목록')
    parser.add_argument('--threshold', type=float, help='특정 Threshold 값으로 테스트')
    parser.add_argument('--interactive', action='store_true', help='대화형 Threshold 테스트 모드')
    args = parser.parse_args()
    
    if args.list:
        list_available_results()
        return
    
    # 모델 경로 결정
    target_paths = []
    if args.path:
        target_paths = [args.path]
    elif args.latest:
        # [변경] 모든 카테고리의 latest 링크를 확인
        for category in os.listdir(RESULTS_BASE_DIR):
            cat_latest = os.path.join(RESULTS_BASE_DIR, category, 'latest')
            if os.path.exists(cat_latest) and os.path.isdir(cat_latest):
                # 그 안의 모델들 추가
                for model in os.listdir(cat_latest):
                    model_path = os.path.join(cat_latest, model)
                    if os.path.isdir(model_path):
                        target_paths.append(model_path)

    
    if not target_paths:
        # 기본: 목록 출력 후 종료
        list_available_results()
        print("\n사용법:")
        print("  python analyze_threshold.py --latest        # 최신 결과 전체 분석")
        print("  python analyze_threshold.py --path <경로>   # 특정 모델 분석")
        print("  python analyze_threshold.py --path <경로> --threshold 0.5  # 특정 값 테스트")
        print("  python analyze_threshold.py --path <경로> --interactive    # 대화형 모드")
        return

    # 분석 실행
    for path in target_paths:
        if args.interactive:
            # 대화형 모드 (단일 모델 권장)
            print(f"\n[{path}] 모델 로드 중...")
            scores, labels, _ = load_results(path)
            
            # 기본 범위 출력
            print(f"  Score Range: {np.min(scores):.4f} ~ {np.max(scores):.4f}")
            
            while True:
                try:
                    user_input = input("\nTest Threshold (or 'q' to quit): ")
                    if user_input.lower() == 'q':
                        break
                    t = float(user_input)
                    test_custom_threshold(scores, labels, t)
                except ValueError:
                    print("유효한 숫자를 입력하세요.")
        
        elif args.threshold is not None:
            # 특정 Threshold 테스트
            print(f"\n📂 Model: {path}")
            scores, labels, _ = load_results(path)
            test_custom_threshold(scores, labels, args.threshold)
            
        else:
            # 기본 종합 분석
            analyze_model(path)


if __name__ == "__main__":
    main()
