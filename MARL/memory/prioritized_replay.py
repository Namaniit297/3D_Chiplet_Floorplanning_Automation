import numpy as np
import random


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.ptr = 0
        self.size = 0

    def update(self, idx, priority):
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def add(self, priority, data):
        self.data[self.ptr] = data
        self.update(self.ptr, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, value):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Hierarchical 3-level prioritized experience replay.
    Level 0 (hot): recent high-priority samples
    Level 1 (warm): mid-priority
    Level 2 (cold): background replay
    """

    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=1e-4):
        cap_per_level = capacity // 3
        self.trees = [SumTree(cap_per_level) for _ in range(3)]
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.thresholds = [0.7, 0.4, 0.0]  # priority thresholds per level

    def _level(self, priority):
        if priority >= self.thresholds[0]:
            return 0
        elif priority >= self.thresholds[1]:
            return 1
        return 2

    def add(self, experience, priority):
        level = self._level(priority)
        self.trees[level].add(priority ** self.alpha, experience)

    def sample(self, batch_size):
        samples, idxs, weights = [], [], []
        per_level = batch_size // 3
        totals = [t.total_priority for t in self.trees]
        total = sum(totals) + 1e-8

        for level, tree in enumerate(self.trees):
            n = per_level if level < 2 else batch_size - 2 * per_level
            if tree.size == 0 or tree.total_priority == 0:
                continue
            seg = tree.total_priority / n
            for i in range(n):
                lo, hi = seg * i, seg * (i + 1)
                val = random.uniform(lo, hi)
                idx, prio, data = tree.sample(val)
                prob = prio / (total + 1e-8)
                w = (1.0 / (tree.size * prob + 1e-8)) ** (1 - self.beta)
                samples.append(data)
                idxs.append((level, idx))
                weights.append(w)

        self.beta = min(1.0, self.beta + self.beta_increment)
        max_w = max(weights) if weights else 1.0
        weights = [w / max_w for w in weights]
        return samples, idxs, weights

    def update_priorities(self, idxs, priorities):
        for (level, idx), p in zip(idxs, priorities):
            self.trees[level].update(idx, p ** self.alpha)

    def __len__(self):
        return sum(t.size for t in self.trees)
