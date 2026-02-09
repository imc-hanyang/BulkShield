import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data_utils import pad_collate_fn



class DeepSVDDNetwork(nn.Module):
    """LSTM을 사용하여 시퀀스를 압축하는 네트워크"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Deep SVDD should not have bias terms to prevent trivial solution (mapping to c=constant)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bias=False)
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)


    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        # 마지막 hidden state 사용
        feature = self.projection(hidden[-1])
        return feature


class DeepSVDDModel:
    def __init__(self, input_dim, hidden_dim=32, epochs=10, batch_size=256, device='cuda:3'):
        self.device = device
        self.net = DeepSVDDNetwork(input_dim, hidden_dim).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.c = None  # Hypersphere center
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-6)

    def _init_center(self, loader):
        """데이터의 초기 Forward Pass를 통해 중심점 c 초기화"""
        n_samples = 0
        c = torch.zeros(self.net.projection.out_features, device=self.device)
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                x, _, _ = batch # pad_collate_fn returns (x, y, lengths)

                x = x.to(self.device)
                outputs = self.net(x)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        # 중심점 c가 너무 0에 가까우면 trivial solution이 되므로 약간의 노이즈 추가 가능
        c[(abs(c) < 0.1) & (c < 0)] = -0.1
        c[(abs(c) < 0.1) & (c > 0)] = 0.1
        return c

    def fit(self, train_data):
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                            collate_fn=pad_collate_fn, num_workers=4)


        # 중심점 초기화
        if self.c is None:
            self.c = self._init_center(loader)

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i, batch in enumerate(loader):
                x, _, _ = batch # Unpack

                x = x.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(x)
                # 거리의 제곱 최소화 ( ||f(x) - c||^2 )
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                loss = torch.mean(dist)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if (i + 1) % 100 == 0:
                     print(f"[Deep SVDD] Epoch {epoch + 1} | Batch {i + 1}/{len(loader)} | Loss: {loss.item():.4f}")
            print(f"[Deep SVDD] Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(loader):.4f}")

    def predict_score(self, test_data) -> np.ndarray:
        loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                            collate_fn=pad_collate_fn, num_workers=4)


        self.net.eval()
        scores = []
        with torch.no_grad():
            for batch in loader:
                x, _, _ = batch

                x = x.to(self.device)
                outputs = self.net(x)
                # 중심점과의 거리가 곧 이상치 점수
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores.extend(dist.cpu().numpy())
        return np.array(scores)