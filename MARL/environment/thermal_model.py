import numpy as np


class ThermalModel:
    """
    Simplified RC-network thermal simulator (PACT-inspired).
    Models steady-state temperature given a power density map.
    """

    def __init__(self, grid_size, n_layers, thermal_res=0.1, ambient_temp=45.0):
        self.grid_size = grid_size
        self.n_layers = n_layers
        self.R = thermal_res
        self.T_amb = ambient_temp

    def simulate(self, power_maps):
        """
        power_maps: (n_layers, grid_size, grid_size) float array [Watts]
        Returns: temperature map (n_layers, grid_size, grid_size)
        """
        temp_maps = np.zeros_like(power_maps, dtype=np.float32)
        for layer in range(self.n_layers):
            # Simple resistive model: T = T_amb + R * P
            temp_maps[layer] = self.T_amb + self.R * power_maps[layer]
            # Inter-layer coupling (vertical heat spreading)
            if layer > 0:
                temp_maps[layer] += 0.05 * temp_maps[layer - 1]
        return temp_maps

    def max_temperature(self, power_maps):
        return float(self.simulate(power_maps).max())

    def thermal_penalty(self, power_maps, threshold=85.0):
        T_max = self.max_temperature(power_maps)
        if T_max <= threshold:
            return 0.0
        return (T_max - threshold) / threshold
