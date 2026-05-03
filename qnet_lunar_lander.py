import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, n_features: int = 8, n_actions: int = 4):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        # Fully Connected Layer 정의
        self.fc1 = nn.Linear(n_features, 128)   # 입력층(s) -> 은닉층 1
        self.fc2 = nn.Linear(128, 128)          # 은닉층 1 -> 은닉층 2
        self.fc3 = nn.Linear(128, n_actions)    # 은닉층 2 -> 출력층(Q)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력이 numpy array인 경우 tensor로 변환
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)

        # ReLU 활성화 함수를 사용하여 FC Layer를 통과
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Q(x, A) 반환
        return x 

    def get_action(self, obs: torch.Tensor, epsilon: float = 0.1) -> int:
        # epsilon-greedy 정책
        if random.random() < epsilon:   # Exploration
            action = random.randrange(0, self.n_actions) # 무작위 행동 선택
        else:   # Exploitation
            q_values = self.forward(obs)            # Q(obs, A)
            action = torch.argmax(q_values, dim=-1) # Q가 최대가 되는 행동 선택
            action = action.item()                  # tensor -> int
        return action  
