import numpy as np


class MicrochannelCooling:
    """
    Microchannel liquid cooling model.
    Computes effective thermal resistance reduction based on flow rate.
    """

    def __init__(self, base_resistance=0.1, flow_rate=1.0):
        self.base_R = base_resistance
        self.flow_rate = flow_rate

    def effective_resistance(self):
        """Higher flow rate → lower effective resistance."""
        return self.base_R / (1.0 + 0.5 * self.flow_rate)

    def cooling_effect(self, power_map):
        """Returns cooled temperature reduction map."""
        R_eff = self.effective_resistance()
        return R_eff * power_map

    def penalty(self, power_map, temp_threshold=85.0, ambient=45.0):
        cooled = ambient + self.cooling_effect(power_map)
        excess = np.maximum(cooled - temp_threshold, 0)
        return float(excess.mean())
