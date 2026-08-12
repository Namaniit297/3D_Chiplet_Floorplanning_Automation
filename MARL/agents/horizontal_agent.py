import torch
import numpy as np
from agents.base_agent import BaseAgent
from agents.ppo import PPO
from models.gnn_encoder import GNNEncoder
from models.policy_head import PolicyHead
from models.value_head import ValueHead


class HorizontalAgent(BaseAgent):
    """
    Intra-layer RL agent (Phase I).
    Each layer gets one HorizontalAgent that places chiplets within that layer.
    """

    def __init__(
        self,
        layer_id,
        node_feat_dim,
        edge_feat_dim,
        grid_size,
        hidden_dim=256,
        lr=3e-4,
        device="cpu",
    ):
        act_dim = grid_size * grid_size
        super().__init__(
            obs_dim=hidden_dim, act_dim=act_dim, lr=lr, device=device
        )
        self.layer_id = layer_id
        self.grid_size = grid_size

        self.encoder = GNNEncoder(node_feat_dim, edge_feat_dim, hidden_dim).to(device)
        self.policy = PolicyHead(hidden_dim, act_dim).to(device)
        self.value = ValueHead(hidden_dim).to(device)

        self.ppo = PPO(self.policy, self.value, lr=lr, device=device)
        self.trajectory = []

    def encode(self, graph_obs):
        return self.encoder(graph_obs)

    def select_action(self, graph_obs, valid_mask=None):
        embedding = self.encode(graph_obs)
        dist = self.policy.get_distribution(embedding)

        if valid_mask is not None:
            mask = torch.BoolTensor(valid_mask).to(self.device)
            logits = dist.logits.clone()
            logits[~mask] = float("-inf")
            dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(embedding)
        return action.item(), log_prob.item(), value.item(), embedding.detach()

    def store_transition(self, transition):
        self.trajectory.append(transition)

    def update(self, batch=None):
        if not self.trajectory:
            return {}
        obs = [t["embedding"].numpy() for t in self.trajectory]
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
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd["encoder"])
        self.policy.load_state_dict(sd["policy"])
        self.value.load_state_dict(sd["value"])
