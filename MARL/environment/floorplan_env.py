import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Data
from environment.thermal_model import ThermalModel
from environment.cooling import MicrochannelCooling
from environment.congestion import compute_congestion, congestion_penalty
from environment.wirelength import compute_hpwl, normalize_hpwl
import yaml, json


class FloorplanEnv(gym.Env):
    """
    3D Chiplet Floorplanning Gymnasium Environment.
    - State: GNN-encoded graph of chiplet netlist per layer
    - Action: (chiplet_id, grid_cell_x, grid_cell_y)
    - Reward: composite of HPWL, thermal, congestion penalties
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path, netlist_path, power_map_path):
        super().__init__()
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        with open(netlist_path) as f:
            self.netlist = json.load(f)

        self.power_map = np.load(power_map_path)

        self.n_layers = self.cfg["n_layers"]
        self.grid_size = self.cfg["grid_size"]
        self.n_chiplets = len(self.netlist["chiplets"])

        self.thermal = ThermalModel(self.grid_size, self.n_layers)
        self.cooling = MicrochannelCooling()

        self.positions = {}
        self.current_layer = 0
        self.step_count = 0
        self.max_steps = self.n_chiplets * self.n_layers

        self.observation_space = gym.spaces.Dict(
            {
                "graph": gym.spaces.Discrete(1),
                "layer": gym.spaces.Discrete(self.n_layers),
            }
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_chiplets, self.grid_size, self.grid_size]
        )

    def _build_graph(self, layer):
        """Build a PyG Data object for the current layer."""
        chiplets = self.netlist["chiplets"]
        n = len(chiplets)
        node_feats = []
        for c in chiplets:
            pos = self.positions.get(c["id"], (-1, -1, -1))
            feat = [
                c.get("width", 1.0),
                c.get("height", 1.0),
                c.get("power", 0.0),
                float(pos[0]) / self.grid_size,
                float(pos[1]) / self.grid_size,
                float(pos[2] if len(pos) > 2 else -1),
                float(c.get("type_id", 0)),
            ]
            node_feats.append(feat)

        x = torch.tensor(node_feats, dtype=torch.float)

        edges = self.netlist.get("nets", [])
        src, dst, ew = [], [], []
        for net in edges:
            for i in range(len(net) - 1):
                src.append(net[i])
                dst.append(net[i + 1])
                ew.append([1.0])

        if src:
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            edge_attr = torch.tensor(ew + ew, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)

        batch = torch.zeros(n, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    def _compute_reward(self):
        nets = self.netlist.get("nets", [])
        hpwl = compute_hpwl(nets, self.positions, (1, 1))
        norm_hpwl = normalize_hpwl(hpwl, self.grid_size, self.grid_size)

        _, max_cong = compute_congestion(nets, self.positions, self.grid_size)
        cong_pen = congestion_penalty(max_cong)

        t_pen = self.thermal.thermal_penalty(self.power_map)
        cool_pen = self.cooling.penalty(self.power_map)

        reward = -(0.4 * norm_hpwl + 0.3 * cong_pen + 0.2 * t_pen + 0.1 * cool_pen)
        return reward, {
            "hpwl": norm_hpwl,
            "congestion": cong_pen,
            "thermal": t_pen,
            "cooling": cool_pen,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = {}
        self.current_layer = 0
        self.step_count = 0
        graph = self._build_graph(self.current_layer)
        return {"graph": graph, "layer": self.current_layer}, {}

    def step(self, action):
        chiplet_id, gx, gy = int(action[0]), int(action[1]), int(action[2])
        self.positions[chiplet_id] = (gx, gy, self.current_layer)
        self.step_count += 1

        placed_in_layer = sum(
            1 for p in self.positions.values() if p[2] == self.current_layer
        )
        layer_capacity = self.cfg.get("chiplets_per_layer", self.n_chiplets)
        if placed_in_layer >= layer_capacity:
            self.current_layer = min(self.current_layer + 1, self.n_layers - 1)

        terminated = self.step_count >= self.max_steps
        reward, info = self._compute_reward()
        graph = self._build_graph(self.current_layer)
        obs = {"graph": graph, "layer": self.current_layer}
        return obs, reward, terminated, False, info

    def render(self):
        print(f"Layer {self.current_layer} | Placed: {len(self.positions)} chiplets")
