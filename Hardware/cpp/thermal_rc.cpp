#include "include/thermal_rc.h"
#include <algorithm>
#include <cmath>
#include <numeric>

ThermalRC::ThermalRC(int n_layers, int grid_size,
                     float R_lateral, float R_vertical,
                     float C_thermal, float ambient)
    : n_layers_(n_layers), grid_size_(grid_size),
      R_lat_(R_lateral), R_vert_(R_vertical),
      C_(C_thermal), T_amb_(ambient) {}

ThermalState ThermalRC::solve_steady(
    const std::vector<std::vector<float>>& power_map,
    int max_iter, float tol)
{
    int N = n_layers_ * grid_size_;
    std::vector<float> T(N, T_amb_);

    for (int iter = 0; iter < max_iter; iter++) {
        float max_delta = 0.0f;
        for (int l = 0; l < n_layers_; l++) {
            for (int n = 0; n < grid_size_; n++) {
                float P     = power_map[l][n];
                float T_old = T[idx(l, n)];
                float sum_g = 0.0f, sum_gT = 0.0f;

                // Lateral neighbors
                if (n > 0) {
                    float g = 1.0f / R_lat_;
                    sum_g  += g; sum_gT += g * T[idx(l, n-1)];
                }
                if (n < grid_size_ - 1) {
                    float g = 1.0f / R_lat_;
                    sum_g  += g; sum_gT += g * T[idx(l, n+1)];
                }
                // Vertical neighbors
                if (l > 0) {
                    float g = 1.0f / R_vert_;
                    sum_g  += g; sum_gT += g * T[idx(l-1, n)];
                }
                if (l < n_layers_ - 1) {
                    float g = 1.0f / R_vert_;
                    sum_g  += g; sum_gT += g * T[idx(l+1, n)];
                }
                // Ambient boundary
                float g_amb = 1.0f / (R_lat_ * 4.0f);
                sum_g  += g_amb;
                sum_gT += g_amb * T_amb_;

                float T_new = (P + sum_gT) / (sum_g + 1e-12f);
                max_delta = std::max(max_delta, std::abs(T_new - T_old));
                T[idx(l, n)] = T_new;
            }
        }
        if (max_delta < tol) break;
    }

    ThermalState state;
    state.temp_map.resize(n_layers_, std::vector<float>(grid_size_));
    state.max_temp = T_amb_;
    float sum_t = 0.0f;

    for (int l = 0; l < n_layers_; l++)
        for (int n = 0; n < grid_size_; n++) {
            float t = T[idx(l, n)];
            state.temp_map[l][n] = t;
            state.max_temp = std::max(state.max_temp, t);
            sum_t += t;
        }

    state.avg_temp   = sum_t / float(N);
    state.alert      = state.max_temp >= 85.0f;
    state.throttle   = state.max_temp >= 95.0f;
    return state;
}

ThermalState ThermalRC::solve_transient(
    const std::vector<std::vector<float>>& power_map,
    float dt, int steps)
{
    int N = n_layers_ * grid_size_;
    std::vector<float> T(N, T_amb_);

    for (int step = 0; step < steps; step++) {
        std::vector<float> dT(N, 0.0f);
        for (int l = 0; l < n_layers_; l++) {
            for (int n = 0; n < grid_size_; n++) {
                float flux = power_map[l][n];
                float Ti   = T[idx(l, n)];

                auto add_flux = [&](float Tj, float R) {
                    flux += (Tj - Ti) / R;
                };
                if (n > 0)              add_flux(T[idx(l, n-1)],   R_lat_);
                if (n < grid_size_ - 1) add_flux(T[idx(l, n+1)],   R_lat_);
                if (l > 0)              add_flux(T[idx(l-1, n)],    R_vert_);
                if (l < n_layers_ - 1)  add_flux(T[idx(l+1, n)],    R_vert_);
                add_flux(T_amb_, R_lat_ * 4.0f);  // ambient

                dT[idx(l, n)] = flux / C_;
            }
        }
        for (int i = 0; i < N; i++)
            T[i] += dt * dT[i];
    }

    ThermalState state;
    state.temp_map.resize(n_layers_, std::vector<float>(grid_size_));
    state.max_temp = T_amb_;
    for (int l = 0; l < n_layers_; l++)
        for (int n = 0; n < grid_size_; n++) {
            state.temp_map[l][n] = T[idx(l, n)];
            state.max_temp = std::max(state.max_temp, T[idx(l, n)]);
        }
    state.alert    = state.max_temp >= 85.0f;
    state.throttle = state.max_temp >= 95.0f;
    return state;
}
