import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# 수렴 경고 무시
warnings.filterwarnings("ignore")


class ARIMAModel:
    """
    ARIMA 기반 이상 탐지기
    전략:
    1. 정상 데이터(Normal)로 ARIMA 모델 파라미터(p,d,q)를 학습함.
    2. 테스트 시, 해당 파라미터를 고정한 상태에서 입력 윈도우의 '예측 오차(Residual)'를 계산함.
    3. 오차가 클수록 이상치(Anomaly)로 판단함.
    """

    def __init__(self, order=(1, 0, 0)):
        # order=(p, d, q)
        # N이 짧고(10), 상태값이므로 복잡한 모델보다는 AR(1) 같은 단순 모델이 유리함
        self.order = order
        self.fitted_params = None
        self.resid_mean = 0
        self.resid_std = 1

    def train(self, X_list):
        """
        정상 데이터들의 패턴을 학습하여 최적의 계수(Coefficient)를 찾음.
        데이터가 너무 끊겨있으므로, 대표적인 긴 정상 시퀀스들을 샘플링하여 학습하거나
        전체를 연결(Concatenate)하여 학습함 (여기선 연결 방식 사용).
        """
        print(f"[ARIMA] 학습 시작 (Order={self.order})...")

        # 1차원으로 쭉 펴서 학습 (점프가 있겠지만 전체적인 '정상'의 자기상관성을 학습하기 위함)
        # HMM과 달리 ARIMA는 flatten해서 넣어야 함
        X_concat = np.concatenate([x.flatten() for x in X_list])

        # 데이터가 너무 크면 학습이 느리므로 최대 10만 개 정도로 자름 (H100이라도 statsmodels는 CPU 연산)
        if len(X_concat) > 100000:
            X_concat = X_concat[:100000]

        model = ARIMA(X_concat, order=self.order)
        model_fit = model.fit()

        # 학습된 파라미터 저장
        self.fitted_params = model_fit.params

        # 정상 데이터에서의 잔차 분포 저장 (Standardization 용도)
        self.resid_mean = np.mean(model_fit.resid)
        self.resid_std = np.std(model_fit.resid)

        print(f"[ARIMA] 학습 완료. 파라미터: {self.fitted_params}")

    def predict_score(self, X_list):
        """
        각 윈도우에 대해 '정상 모델'을 적용했을 때의 예측 오차(MSE)를 반환함.
        오차가 클수록 이상 패턴임.
        """
        if self.fitted_params is None:
            raise ValueError("모델이 학습되지 않았음")

        scores = []

        # apply() 메서드를 사용하여 파라미터를 고정하고 새로운 데이터에 적용
        # 속도를 위해 루프 최소화가 필요하지만, statsmodels 구조상 개별 적용 필요
        # H100 서버라도 CPU 병목이 생길 수 있는 구간임 -> 간단한 AR 로직으로 직접 구현도 가능하나
        # 여기서는 라이브러리의 apply 기능을 사용함.

        print("[ARIMA] 예측 오차 계산 중...")

        # 윈도우 단위 처리는 statsmodels 오버헤드가 큼.
        # 따라서 수동으로 AR(p) 예측 오차를 계산하는 것이 훨씬 빠름.
        # 여기서는 order=(1,0,0) 즉 AR(1)이라고 가정하고 고속 연산 구현함.

        # AR(1) 수식: y_t = const + coeff * y_{t-1} + error
        if self.order == (1, 0, 0):
            const = self.fitted_params[0]
            ar_coeff = self.fitted_params[1]
            # sigma2 = self.fitted_params[2]

            X_batch = np.array([x.flatten() for x in X_list])
            # 예측값: Prev * coeff + const
            preds = X_batch[:, :-1] * ar_coeff + const
            actuals = X_batch[:, 1:]
            mse = np.mean((actuals - preds) ** 2, axis=1)
            scores = mse

        elif self.order == (0, 0, 1):
            # MA(1) 수식: y_t = const + error_t + theta * error_{t-1}
            # error_t = y_t - const - theta * error_{t-1}
            # Recursive calculation needed

            const = self.fitted_params[0]
            ma_coeff = self.fitted_params[1]
            
            X_batch = np.array([x.flatten() for x in X_list])
            # (Batch, Seq_Len)
            
            # Recursive calculation using Numpy is hard to fully vectorize across time without loop,
            # but we can loop over time steps which is small (10), while vectorizing over batch (470k).
            # This is much faster than looping over batch.
            
            batch_size, seq_len = X_batch.shape
            errors = np.zeros_like(X_batch)
            
            # t=0: error_0 = y_0 - const (assume error_{-1}=0)
            errors[:, 0] = X_batch[:, 0] - const
            
            for t in range(1, seq_len):
                # error_t = y_t - const - theta * error_{t-1}
                errors[:, t] = X_batch[:, t] - const - ma_coeff * errors[:, t-1]
                
            # Score = Mean Squared Error of residuals (errors)
            scores = np.mean(errors ** 2, axis=1)

        else:
            # 그 외 Order는 statsmodels 사용하되, re-fitting 없이 apply 사용 시도
            # (statsmodels 버전에 따라 apply가 없을 수 있으므로 예외처리)
            scores = []
            print(f"[ARIMA] Warning: Order {self.order} slow loop prediction...")
            for x in X_list:
                try:
                    ts = x.flatten()
                    # Re-fitting is too slow. Just return 0 or do simple approximation if critical.
                    # For now, stick to original logic but warn.
                    res = ARIMA(ts, order=self.order).fit(start_params=self.fitted_params)
                    scores.append(np.mean(res.resid ** 2))
                except:
                    scores.append(np.inf)
            scores = np.array(scores)

        return scores