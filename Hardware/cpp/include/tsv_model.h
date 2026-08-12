#pragma once
#include <vector>

class TSVModel {
public:
    TSVModel(int n_tsvs, float tsv_resistance = 0.05f,
             float tsv_capacitance = 1e-12f,
             float tsv_pitch_um = 10.0f);

    // Compute temperature delta across each TSV given vertical heat flux
    std::vector<float> compute_temp_drop(
        const std::vector<float>& heat_flux_W) const;

    // Congestion penalty: fraction of TSVs exceeding thermal threshold
    float congestion_penalty(
        const std::vector<float>& temp_drops,
        float threshold_C = 5.0f) const;

    // Bandwidth derating due to thermal expansion
    float bandwidth_derating(float avg_temp_drop) const;

private:
    int   n_tsvs_;
    float R_tsv_, C_tsv_, pitch_;
};
