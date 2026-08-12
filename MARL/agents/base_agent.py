import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, device="cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.device = torch.device(device)

    @abstractmethod
    def select_action(self, obs):
        pass

    @abstractmethod
    def update(self, batch):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
