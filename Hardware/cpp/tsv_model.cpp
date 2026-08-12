#include "include/tsv_model.h"
#include <numeric>
#include <algorithm>
#include <cmath>

TSVModel::TSVModel(int n_tsvs, float tsv_resistance,
                   float tsv_capacitance, float tsv_pitch_um)
    : n_tsvs_(n_tsvs), R_tsv_(tsv_resistance),
      C_tsv_(tsv_capacitance), pitch_(tsv_pitch_um) {}

std::vector<float> TSVModel::compute_temp_drop(
    const std::vector<float>& heat_flux_W) const
{
    std::vector<float> drops(heat_flux_W.size());
    for (size_t i = 0; i < heat_flux_W.size(); i++)
        drops[i] = heat_flux_W[i] * R_tsv_;
    return drops;
}

float TSVModel::congestion_penalty(
    const std::vector<float>& temp_drops,
    float threshold_C) const
{
    int exceed = 0;
    for (float d : temp_drops)
        if (d > threshold_C) ++exceed;
    return float(exceed) / float(temp_drops.size() + 1e-9f);
}

float TSVModel::bandwidth_derating(float avg_temp_drop) const {
    // Empirical: BW derates 0.5% per °C above 5°C drop
    float excess = std::max(0.0f, avg_temp_drop - 5.0f);
    return std::max(0.5f, 1.0f - 0.005f * excess);
}
