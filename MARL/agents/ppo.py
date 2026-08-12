import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPO:
    def __init__(
        self,
        policy_net,
        value_net,
        lr=3e-4,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        device="cpu",
    ):
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.device = device

        self.optimizer = optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()), lr=lr
        )

    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, obs, actions, log_probs_old, returns, advantages):
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            dist = self.policy_net.get_distribution(obs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            values_pred = self.value_net(obs).squeeze()
            value_loss = nn.MSELoss()(values_pred, returns)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
