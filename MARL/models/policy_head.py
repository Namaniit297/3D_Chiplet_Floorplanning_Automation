import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    MLP policy head: embedding → action logits → Categorical distribution
    """

    def __init__(self, hidden_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_distribution(self, x):
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)
