
import os
import glob
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime

# 설정
INPUT_DIR = "/home/srt/Dataset/passenger_split_final/group_0"
TEST_FILES = [
    "000a090fcd1e7e31bee24519d9e33d7d3aac266986ceb3499d37e229a530882b.csv",
    "000c4ffd5dc99940f2592018bcf8c35d906f1d860e4673a878c6d89341eef03d.csv"
]
REFUND_THRESHOLD = 1_000_000
REFUND_RATE_THRESHOLD = 0.90

def parse_dot_date(date_str):
    try:
        return pd.to_datetime(str(date_str).strip(), format="%Y.%m.%d")
    except:
        return pd.NaT

def process_file(filename):
    file_path = os.path.join(INPUT_DIR, filename)
    print(f"\nProcessing: {filename}")
    
    # 1. Load Data (Raw)
    try:
        # 데이터가 클 수 있으니 필요한 컬럼만 읽거나, dtype 지정
        # Raw 데이터 컬럼: 출발일자, 반환금액, 고객관리번호 ...
        # csv 모듈 대신 pandas read_csv 사용 (편의성)
        # 인코딩: 한글 데이터이므로 utf-8 아니면 cp949. 아까 head는 잘 보였음.
        df = pd.read_csv(file_path, on_bad_lines='skip')
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 컬럼 확인
    if '출발일자' not in df.columns or '반환금액' not in df.columns:
        print("Required columns missing!")
        print(df.columns)
        return

    # 2. Preprocess Date
    df['출발일자_dt'] = df['출발일자'].apply(parse_dot_date)
    df = df.dropna(subset=['출발일자_dt'])
    
    if df.empty:
        print("Date parsing failed or empty.")
        return

    # 3. Determine Window (End Date = Max Departure Date)
    end_date = df['출발일자_dt'].max()
    start_date = end_date - relativedelta(months=1) + pd.Timedelta(days=1)
    
    print(f"  - Window: {start_date.date()} ~ {end_date.date()}")
    
    # 4. Filter Window
    mask = (df['출발일자_dt'] >= start_date) & (df['출발일자_dt'] <= end_date)
    window_df = df[mask].copy()
    
    print(f"  - Total Rows: {len(df)} -> Window Rows: {len(window_df)}")
    
    if window_df.empty:
        print("Window is empty.")
        return

    # 5. Aggregate Stats
    total_issued = len(window_df)
    
    # 반환금액 처리 (콤마 제거 등 필요할 수 있음)
    # head 출력에서 0, 27700 숫자 그대로 보임.
    # 혹시 문자열일 수 있으니 변환
    window_df['반환금액'] = pd.to_numeric(window_df['반환금액'], errors='coerce').fillna(0)
    
    total_refund_amt = window_df['반환금액'].sum()
    
    # 반환 횟수: 반환금액 > 0 인 건수 (가장 확실)
    # 혹은 반환일자가 있는 경우?
    # 여기선 '반환금액 > 0'으로 집계
    refund_cnt = (window_df['반환금액'] > 0).sum()
    
    refund_rate = refund_cnt / total_issued if total_issued > 0 else 0.0
    
    # 6. Rule-Based Labeling
    is_risk = (total_refund_amt >= REFUND_THRESHOLD) and (refund_rate >= REFUND_RATE_THRESHOLD)
    
    result = {
        'User': str(window_df['고객관리번호'].iloc[0]) if '고객관리번호' in window_df else 'Unknown',
        'End_Date': end_date.strftime("%Y%m%d"),
        'Refund_Amt': total_refund_amt,
        'Refund_Rate': refund_rate,
        'Issued_Cnt': total_issued,
        'Refund_Cnt': refund_cnt,
        'Is_Risk': int(is_risk)
    }
    
    print("  - Result:", result)

# 실행
for f in TEST_FILES:
    process_file(f)
