import numpy as np
from memory.prioritized_replay import PrioritizedReplayBuffer


class IERB:
    """
    Intelligent Experience Relay Buffer.
    Dynamically routes experiences to priority levels and supports
    curriculum-weighted sampling.
    """

    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.buffer = PrioritizedReplayBuffer(capacity, alpha, beta)
        self.episode_rewards = []
        self.curriculum_phase = 0

    def _compute_priority(self, td_error, reward, done):
        base = abs(td_error) + 1e-6
        bonus = 0.1 * abs(reward)
        terminal_bonus = 0.2 if done else 0.0
        return float(np.clip(base + bonus + terminal_bonus, 0.0, 1.0))

    def add(self, obs, action, reward, next_obs, done, td_error=0.0):
        priority = self._compute_priority(td_error, reward, done)
        self.buffer.add(
            {"obs": obs, "action": action, "reward": reward,
             "next_obs": next_obs, "done": done},
            priority,
        )

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def update_priorities(self, idxs, td_errors):
        priorities = [abs(e) + 1e-6 for e in td_errors]
        self.buffer.update_priorities(idxs, priorities)

    def advance_curriculum(self):
        self.curriculum_phase += 1

    def __len__(self):
        return len(self.buffer)
