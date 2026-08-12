import torch
import numpy as np
from agents.base_agent import BaseAgent
from agents.ppo import PPO
from models.vertical_coordinator import VerticalCoordinator
from models.policy_head import PolicyHead
from models.value_head import ValueHead


class VerticalAgent(BaseAgent):
    """
    Inter-layer coordination agent (Phase II).
    Aggregates per-layer embeddings and emits cross-layer reward shaping signals.
    """

    def __init__(
        self,
        n_layers,
        layer_embed_dim,
        hidden_dim=256,
        lr=3e-4,
        device="cpu",
    ):
        act_dim = n_layers
        super().__init__(obs_dim=hidden_dim, act_dim=act_dim, lr=lr, device=device)
        self.n_layers = n_layers

        self.coordinator = VerticalCoordinator(n_layers, layer_embed_dim, hidden_dim).to(device)
        self.policy = PolicyHead(hidden_dim, act_dim).to(device)
        self.value = ValueHead(hidden_dim).to(device)

        self.ppo = PPO(self.policy, self.value, lr=lr, device=device)
        self.trajectory = []

    def select_action(self, layer_embeddings):
        """
        layer_embeddings: list of tensors, one per layer
        """
        stacked = torch.stack(layer_embeddings, dim=0).unsqueeze(0).to(self.device)
        context = self.coordinator(stacked)
        dist = self.policy.get_distribution(context)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(context)
        return action.item(), log_prob.item(), value.item(), context.detach()

    def store_transition(self, transition):
        self.trajectory.append(transition)

    def update(self, batch=None):
        if not self.trajectory:
            return {}
        obs = [t["embedding"].squeeze(0).numpy() for t in self.trajectory]
        actions = [t["action"] for t in self.trajectory]
        log_probs = [t["log_prob"] for t in self.trajectory]
        rewards = [t["reward"] for t in self.trajectory]
        values = [t["value"] for t in self.trajectory]
        dones = [t["done"] for t in self.trajectory]
        next_value = self.trajectory[-1].get("next_value", 0.0)

        advantages, returns = self.ppo.compute_gae(rewards, values, dones, next_value)
        info = self.ppo.update(
            np.array(obs), actions, log_probs, returns, advantages
        )
        self.trajectory = []
        return info

    def state_dict(self):
        return {
            "coordinator": self.coordinator.state_dict(),
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }

    def load_state_dict(self, sd):
        self.coordinator.load_state_dict(sd["coordinator"])
        self.policy.load_state_dict(sd["policy"])
        self.value.load_state_dict(sd["value"])
