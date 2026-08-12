import subprocess
import json
import os
import numpy as np
from hardware.apu_core  import APUCore
from hardware.pe_array  import PEArray
from hardware.noc       import NoC


class HardwareSimulator:
    """
    Full hardware simulator:
    - Uses C++ cycle_sim for thermal/cache/NoC
    - Falls back to pure-Python models if binary not built
    """

    BINARY = "build/flame_hw_sim"

    def __init__(self, n_apu_cores=4, n_chiplets=16, n_layers=2,
                 grid_size=16, sim_cycles=100_000, use_cpp=True):
        self.n_cores   = n_apu_cores
        self.n_chiplets= n_chiplets
        self.n_layers  = n_layers
        self.grid_size = grid_size
        self.sim_cycles= sim_cycles
        self.use_cpp   = use_cpp and os.path.exists(self.BINARY)

        self.apus     = [APUCore(i) for i in range(n_apu_cores)]
        self.pe_array = PEArray()
        self.noc      = NoC(n_chiplets)

    # ── C++ backed run ─────────────────────────────────────────────────────
    def _run_cpp(self, power_map: np.ndarray):
        power_flat = power_map.flatten().tolist()
        cfg = {
            "n_cores":    self.n_cores,
            "n_chiplets": self.n_chiplets,
            "n_layers":   self.n_layers,
            "grid_size":  self.grid_size,
            "sim_cycles": self.sim_cycles,
            "power_map":  power_flat,
        }
        cfg_path = "/tmp/flame_sim_cfg.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        result = subprocess.run(
            [self.BINARY, "--config", cfg_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"C++ sim failed:\n{result.stderr}")
        return json.loads(result.stdout)

    # ── Pure-Python fallback ───────────────────────────────────────────────
    def _run_python(self, ops_list, mem_ops_list,
                    traffic_pairs, power_map):
        from environment.thermal_model import ThermalModel
        from environment.cooling       import MicrochannelCooling

        apu_results = [
            apu.run_workload(ops, mops)
            for apu, ops, mops in zip(self.apus, ops_list, mem_ops_list)
        ]
        pe_result = self.pe_array.dispatch(np.random.rand(8, 8))

        if traffic_pairs:
            for src, dst, vol in traffic_pairs:
                self.noc.inject_traffic(src, dst, vol)
        noc_result = self.noc.simulate()

        thermal = ThermalModel(self.grid_size, self.n_layers)
        cooling = MicrochannelCooling()
        temp_maps = thermal.simulate(power_map)
        cooled    = cooling.apply_cooling(power_map)

        return {
            "apu":      apu_results,
            "pe_array": pe_result,
            "noc":      noc_result,
            "thermal": {
                "max_temp":  float(temp_maps.max()),
                "avg_temp":  float(temp_maps.mean()),
                "alert":     bool(temp_maps.max() >= 85.0),
                "throttle":  bool(temp_maps.max() >= 95.0),
            },
            "cache": {
                "l1_hit_rate": 0.85,  # placeholder for Python fallback
                "l2_hit_rate": 0.72,
            }
        }

    def run(self, ops_list=None, mem_ops_list=None,
            traffic_pairs=None, power_map=None):
        if power_map is None:
            power_map = np.random.rand(self.n_layers, self.grid_size) * 80.0
        if ops_list is None:
            ops_list = [1e9] * self.n_cores
        if mem_ops_list is None:
            mem_ops_list = [5e8] * self.n_cores

        if self.use_cpp:
            return self._run_cpp(power_map)
        return self._run_python(ops_list, mem_ops_list, traffic_pairs, power_map)

    def reset(self):
        for apu in self.apus:
            apu.reset()
        self.pe_array.reset()
        self.noc.reset()
