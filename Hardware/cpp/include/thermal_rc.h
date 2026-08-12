#pragma once
#include <vector>
#include "sim_types.h"

class ThermalRC {
public:
    ThermalRC(int n_layers, int grid_size,
              float R_lateral  = 0.08f,   // °C/W lateral
              float R_vertical = 0.15f,   // °C/W vertical (inter-layer)
              float C_thermal  = 0.002f,  // J/°C capacitance
              float ambient    = 45.0f);

    // Steady-state solve (Gauss-Seidel)
    ThermalState solve_steady(
        const std::vector<std::vector<float>>& power_map,
        int max_iter = 500, float tol = 0.01f);

    // Transient solve (explicit Euler)
    ThermalState solve_transient(
        const std::vector<std::vector<float>>& power_map,
        float dt = 1e-6f, int steps = 1000);

private:
    int   n_layers_, grid_size_;
    float R_lat_, R_vert_, C_, T_amb_;

    inline int idx(int layer, int node) const {
        return layer * grid_size_ + node;
    }
};
