#pragma once
#include <vector>

class LiquidCooling {
public:
    LiquidCooling(float flow_rate_ml_s    = 1.5f,
                  float channel_width_um  = 100.0f,
                  float channel_depth_um  = 200.0f,
                  float n_channels        = 50,
                  float fluid_conductivity= 0.6f,   // W/m·K (water)
                  float fluid_density     = 997.0f, // kg/m³
                  float fluid_cp          = 4182.0f);// J/kg·K

    // Compute cooled temperature map after microchannel cooling
    std::vector<std::vector<float>> apply_cooling(
        const std::vector<std::vector<float>>& temp_map,
        const std::vector<std::vector<float>>& power_map);

    // Nusselt number based heat transfer coefficient
    float heat_transfer_coeff() const;

    // Total pumping power [W]
    float pumping_power() const;

    // Effective thermal resistance [°C/W]
    float effective_resistance() const;

private:
    float Q_ml_s_, W_ch_, D_ch_, N_ch_;
    float k_f_, rho_, cp_;
};
