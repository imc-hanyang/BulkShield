import numpy as np
import itertools
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 기존 모듈 import
# from data_utils import load_train_data

# 경고 무시
warnings.filterwarnings("ignore")

CONFIG = {
    'output_file': 'best_params.json',  # 결과 저장 파일
    'sample_size': 20000,  # 튜닝에 사용할 최대 데이터 포인트 (속도 조절)
    'seasonal_period': 7,  # SARIMA 주기 (도메인 지식 기반 고정)
    'n_jobs': 8  # 병렬 처리 워커 수
}


import pandas as pd
import torch
import data_utils
from data_utils import collect_files_from_groups, TRAIN_LABEL_PATH, TRAIN_SEQUENCE_PATH
import used_raw_columns
from tqdm import tqdm

def get_representative_data():
    """
    135만 명 데이터를 스트리밍 방식으로 읽어서 
    '정상(Label=0)' 데이터만 모음 (메모리 최적화)
    """
    print("\n[Data Stream] 파일 목록 스캔 중...")
    label_files = collect_files_from_groups(TRAIN_LABEL_PATH, 'label_')
    seq_files = collect_files_from_groups(TRAIN_SEQUENCE_PATH, 'seq_')
    
    # 매칭되는 유저 찾기
    common_users = list(set(label_files.keys()) & set(seq_files.keys()))
    # 랜덤성을 위해 섞기 (앞쪽 유저만 뽑히는 편향 방지)
    np.random.shuffle(common_users)
    
    print(f"[Data Stream] 매칭된 유저: {len(common_users)}명. 데이터 수집 시작...")
    
    normal_data = []
    current_size = 0
    target_size = CONFIG['sample_size']
    
    for user_id in tqdm(common_users, desc="Streaming Data"):
        if current_size >= target_size:
            break
            
        try:
            # print(f"[DEBUG] Processing {user_id}", flush=True) 
            
            # 1. Label 확인
            df_lbl = pd.read_csv(label_files[user_id])
            if len(df_lbl) == 0: continue
            is_risk = int(df_lbl['is_risk'].iloc[0])
            
            # 비정상 데이터는 건너뜀
            if is_risk != 0:
                continue
                
            # 2. Sequence 로드
            # print(f"[DEBUG] Reading sequence for {user_id}", flush=True)
            df_seq = pd.read_csv(seq_files[user_id])
            if len(df_seq) == 0: continue
            
            # 컬럼 필터 및 숫자 변환
            df_seq = df_seq[used_raw_columns.columns]
            df_seq = df_seq.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # 1차원으로 펴서 수집
            features_flat = df_seq.values.flatten()
            normal_data.extend(features_flat)
            current_size += len(features_flat)
            
            # print(f"[DEBUG] Done {user_id}, Current size: {current_size}", flush=True)
            
        except Exception as e:
            print(f"[Error] {user_id}: {e}")
            pass
            
    # 데이터가 너무 많으면 자르기
    data_concat = np.array(normal_data)
    if len(data_concat) > target_size:
        data_concat = data_concat[:target_size]
        
    print(f"\n[Data Info] 수집 완료: {len(data_concat)} points (목표: {target_size})")
    return data_concat


# =============================================================================
# ARIMA 병렬 처리용 함수
# =============================================================================
def _fit_arima(args):
    """단일 ARIMA 조합 fitting (병렬 처리용)"""
    data, order = args
    if order == (0, 0, 0):
        return None
    try:
        model = ARIMA(data, order=order)
        res = model.fit()
        return (order, res.aic)
    except:
        return None


def grid_search_arima(data):
    print("\n>>> ARIMA (p, d, q) Grid Search 시작...")

    # 탐색 범위
    p_values = [0, 1, 2, 3]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    combinations = list(itertools.product(p_values, d_values, q_values))
    print(f"    총 {len(combinations)}개 조합 탐색 (병렬: {CONFIG['n_jobs']} workers)")

    best_aic = float("inf")
    best_order = (1, 0, 0)

    # 병렬 처리
    args_list = [(data, order) for order in combinations]
    with ProcessPoolExecutor(max_workers=CONFIG['n_jobs']) as executor:
        futures = {executor.submit(_fit_arima, args): args[1] for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                order, aic = result
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    print(f"   Update Best: {best_order} (AIC: {best_aic:.2f})")

    print(f"[Done] Best ARIMA Order: {best_order}")
    return best_order


# =============================================================================
# SARIMA 병렬 처리용 함수
# =============================================================================
def _fit_sarima(args):
    """단일 SARIMA 조합 fitting (병렬 처리용)"""
    data, arima_order, seasonal_order = args
    try:
        model = SARIMAX(data, order=arima_order, seasonal_order=seasonal_order)
        res = model.fit(disp=False)
        return (seasonal_order, res.aic)
    except:
        return None


def grid_search_sarima(data, best_arima_order):
    """
    SARIMA Grid Search
    
    [핵심] ARIMA에서 찾은 (p,d,q)를 '고정'하고, 계절성 (P,D,Q,s)만 탐색
    
    SARIMA 모델: SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s))
    - order: 비계절성 파트 → ARIMA에서 이미 최적화된 값 사용
    - seasonal_order: 계절성 파트 → 여기서 탐색
    
    이렇게 하는 이유:
    1. 전체 (p,d,q,P,D,Q) 조합은 너무 많음 (수백 개)
    2. 비계절성 패턴은 ARIMA로 충분히 캡처됨
    3. SARIMA는 추가로 '주기적 패턴'만 학습하면 됨
    """
    print(f"\n>>> SARIMA Seasonal (P, D, Q, s={CONFIG['seasonal_period']}) Grid Search 시작...")
    print(f"    비계절성 order는 ARIMA 결과 사용: {best_arima_order}")

    P_values = [0, 1, 2]
    D_values = [0, 1]
    Q_values = [0, 1, 2]
    s = CONFIG['seasonal_period']

    combinations = list(itertools.product(P_values, D_values, Q_values))
    print(f"    총 {len(combinations)}개 조합 탐색 (병렬: {CONFIG['n_jobs']} workers)")

    best_aic = float("inf")
    best_seasonal = (1, 0, 0, s)

    # 병렬 처리
    args_list = [(data, best_arima_order, (p, d, q, s)) for p, d, q in combinations]
    with ProcessPoolExecutor(max_workers=CONFIG['n_jobs']) as executor:
        futures = {executor.submit(_fit_sarima, args): args[2] for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                seasonal_order, aic = result
                if aic < best_aic:
                    best_aic = aic
                    best_seasonal = seasonal_order
                    print(f"   Update Best Seasonal: {best_seasonal} (AIC: {best_aic:.2f})")

    print(f"[Done] Best SARIMA Seasonal Order: {best_seasonal}")
    return best_seasonal


def main():
    try:
        # 1. 데이터 준비
        data = get_representative_data()

        # 2. ARIMA 튜닝 (비계절성 파라미터 탐색)
        best_arima = grid_search_arima(data)

        # 3. SARIMA 튜닝 (계절성 파라미터 탐색, ARIMA 결과 활용)
        best_sarima_seasonal = grid_search_sarima(data, best_arima)

        # 4. 결과 저장
        result = {
            "arima_order": best_arima,
            "sarima_order": best_arima,  # SARIMA의 비계절성 파트는 ARIMA와 동일
            "sarima_seasonal": best_sarima_seasonal
        }

        with open(CONFIG['output_file'], 'w') as f:
            json.dump(result, f, indent=4)

        print(f"\n[Success] 최적 파라미터가 '{CONFIG['output_file']}'에 저장되었습니다.")
        print(json.dumps(result, indent=4))

    except Exception as e:
        print(f"[Error] {e}")


if __name__ == "__main__":
    main()