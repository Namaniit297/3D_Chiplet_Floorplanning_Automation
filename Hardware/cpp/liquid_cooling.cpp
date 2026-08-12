#include "include/liquid_cooling.h"
#include <cmath>
#include <algorithm>

LiquidCooling::LiquidCooling(
    float flow_rate_ml_s, float channel_width_um,
    float channel_depth_um, float n_channels,
    float fluid_conductivity, float fluid_density,
    float fluid_cp)
    : Q_ml_s_(flow_rate_ml_s),
      W_ch_(channel_width_um * 1e-6f),
      D_ch_(channel_depth_um * 1e-6f),
      N_ch_(n_channels), k_f_(fluid_conductivity),
      rho_(fluid_density), cp_(fluid_cp) {}

float LiquidCooling::heat_transfer_coeff() const {
    // Dittus-Boelter for turbulent flow; use simplified Nu=4.36 for laminar
    float D_h   = 2.0f * W_ch_ * D_ch_ / (W_ch_ + D_ch_);  // hydraulic diameter
    float Q_m3s = Q_ml_s_ * 1e-6f / N_ch_;                   // per channel
    float A_ch  = W_ch_ * D_ch_;
    float v     = Q_m3s / (A_ch + 1e-20f);
    float Re    = rho_ * v * D_h / 1e-3f;  // assume μ=1e-3 Pa·s (water)
    float Nu    = (Re > 2300.0f)
                ? 0.023f * std::pow(Re, 0.8f) * std::pow(6.9f, 0.4f)  // Pr≈6.9 water
                : 4.36f;
    return Nu * k_f_ / D_h;
}

float LiquidCooling::pumping_power() const {
    float Q_m3s = Q_ml_s_ * 1e-6f;
    float dP    = 1000.0f;   // assumed pressure drop [Pa]
    return Q_m3s * dP;
}

float LiquidCooling::effective_resistance() const {
    float h = heat_transfer_coeff();
    float A_total = N_ch_ * W_ch_ * 0.01f;  // 1 cm channel length
    return 1.0f / (h * A_total + 1e-12f);
}

std::vector<std::vector<float>> LiquidCooling::apply_cooling(
    const std::vector<std::vector<float>>& temp_map,
    const std::vector<std::vector<float>>& power_map)
{
    float R_eff = effective_resistance();
    auto cooled = temp_map;
    for (size_t l = 0; l < temp_map.size(); l++)
        for (size_t n = 0; n < temp_map[l].size(); n++)
            cooled[l][n] = temp_map[l][n] - R_eff * power_map[l][n] * 0.5f;
    return cooled;
}
