import numpy as np
from hmmlearn import hmm
import joblib


class HMMModel:
    """
    Hidden Markov Model 기반 이상 탐지기임
    정상 시퀀스의 패턴을 학습하고, Log-Likelihood를 통해 이상치를 탐지함
    """

    def __init__(self, n_components=4, n_iter=100, random_state=42):
        # n_components: 잠재 상태(State)의 개수임
        # GaussianHMM: 피처가 연속형 수치라고 가정함 (state만 있다면 MultinomialHMM 사용 가능)
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            implementation="log",
            verbose=True  # 진행 상황 출력
        )
        self.is_fitted = False

    def train(self, X_list):
        """
        X_list: (N, F) 형태의 Numpy Array들이 담긴 리스트임
        HMM은 정상 데이터(Label=0)만으로 학습하는 것이 일반적임
        """
        # hmmlearn은 전체 데이터를 하나로 합치고, 길이를 따로 줘야 함
        X_concat = np.concatenate(X_list)
        lengths = [x.shape[0] for x in X_list]

        print(f"[HMM] 학습 시작함 (총 {len(X_list)}개 시퀀스)...")
        self.model.fit(X_concat, lengths)
        self.is_fitted = True
        print("[HMM] 학습 완료됨")

    def predict_score(self, X_list):
        """
        각 시퀀스별 Log Likelihood를 계산하여 반환함
        점수가 낮을수록(음의 무한대) 이상치임.
        일관성을 위해 (-1)을 곱해 '점수가 높을수록 이상'하도록 변환함
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았음")

        scores = []
        for x in X_list:
            # score()는 Log Likelihood를 반환함
            try:
                score = self.model.score(x)
                scores.append(-score)  # 부호 반전 (높을수록 이상)
            except:
                scores.append(np.inf)  # 에러 시 매우 이상한 것으로 간주

        return np.array(scores)