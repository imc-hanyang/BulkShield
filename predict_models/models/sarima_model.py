import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# 수렴 경고 및 데이터 부족 경고 무시
warnings.filterwarnings("ignore")


class SARIMAModel:
    """
    SARIMA (Seasonal ARIMA) 기반 이상 탐지기

    특징:
    - 데이터 내의 주기적인 패턴(Seasonality)을 학습함.
    - 예: 계절성 주기(s)가 3이라면, 3스텝마다 반복되는 규칙성을 찾음.

    전략:
    1. 정상 데이터로 Global SARIMA 파라미터를 학습(Fit).
    2. 테스트 시, 파라미터를 고정(Fix)하고 입력 시퀀스에 적용(Apply).
    3. 예측 오차(Residual)가 크면 이상치로 판단.
    """

    def __init__(self, order=(1, 0, 0), seasonal_order=(1, 0, 0, 3)):
        """
        order: (p, d, q) - 비계절성 파라미터
        seasonal_order: (P, D, Q, s) - 계절성 파라미터
         - s: 주기 (예: 로그 10개 중 3개 단위로 패턴이 있다면 3)
         - N=10으로 짧기 때문에 s는 3~4 정도로 작게 잡는 것이 유리함.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model_res = None  # 학습된 결과 객체 저장

    def train(self, X_list):
        """
        정상 시퀀스들을 연결하여 전반적인 주기성과 자기회귀 패턴을 학습함.
        """
        print(f"[SARIMA] 학습 시작 (Order={self.order}, Seasonal={self.seasonal_order})...")

        # 1차원으로 연결 (Flatten)
        X_concat = np.concatenate([x.flatten() for x in X_list])

        # 데이터가 너무 많으면 학습 속도가 매우 느리므로 샘플링 (최대 5만 개)
        if len(X_concat) > 50000:
            X_concat = X_concat[:50000]

        # SARIMAX 모델 정의 및 학습
        # disp=False: 로그 출력 끄기
        model = SARIMAX(X_concat, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model_res = model.fit(disp=False)

        print(f"[SARIMA] 학습 완료. AIC: {self.fitted_model_res.aic:.2f}")

    def predict_score(self, X_list):
        """
        학습된 파라미터를 사용하여 테스트 데이터의 잔차(Residual)를 계산함.
        잔차 제곱 평균(MSE)을 이상 점수로 반환.
        """
        if self.fitted_model_res is None:
            raise ValueError("모델이 학습되지 않았음")

        scores = []

        # SARIMA는 구조가 복잡하여 수동 연산이 어려우므로
        # statsmodels의 apply() 메서드를 사용하여 파라미터를 적용함.
        # apply(): 기존 fit 결과를 새로운 데이터에 적용하여 filtering만 수행 (학습 X, 속도 빠름)

        # 배치 처리가 안 되므로 루프를 돌려야 함 (속도 이슈 가능성 있음)
        for i, x in enumerate(X_list):
            try:
                ts = x.flatten()
                # 기존 학습된 모델의 파라미터를 새 데이터(ts)에 적용
                res = self.fitted_model_res.apply(ts)

                # 잔차(resid) 가져오기
                # 초기 몇 개는 예측이 불안정할 수 있으므로 뒤쪽 데이터 위주로 볼 수도 있음
                mse = np.mean(res.resid ** 2)
                scores.append(mse)
            except Exception as e:
                # 에러 발생 시 (데이터가 너무 짧거나 수렴 실패)
                scores.append(np.inf)

            if i % 1000 == 0 and i > 0:
                print(f"[SARIMA] {i}/{len(X_list)} 처리 중...", end='\r')

        print(f"[SARIMA] 예측 완료 (총 {len(X_list)}개)        ")
        return np.array(scores)