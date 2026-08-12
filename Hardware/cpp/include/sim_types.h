#pragma once
#include <cstdint>
#include <vector>
#include <string>

struct SimConfig {
    int   n_cores       = 4;
    int   n_chiplets    = 16;
    int   n_layers      = 2;
    int   grid_size     = 16;
    int   sim_cycles    = 100000;
    float freq_ghz      = 2.0f;
    float ambient_temp  = 45.0f;
    float tsv_res       = 0.05f;   // °C/W per TSV
    float flow_rate     = 1.5f;    // ml/s
};

struct ThermalState {
    std::vector<std::vector<float>> temp_map;  // [layer][node]
    float max_temp   = 0.0f;
    float avg_temp   = 0.0f;
    bool  alert      = false;
    bool  throttle   = false;
};

struct CacheStats {
    uint64_t hits        = 0;
    uint64_t misses      = 0;
    uint64_t evictions   = 0;
    double   hit_rate()  const {
        return hits + misses > 0 ? double(hits) / (hits + misses) : 0.0;
    }
};

struct NoCStats {
    uint64_t total_flits  = 0;
    uint64_t stall_cycles = 0;
    double   avg_latency  = 0.0;
    double   throughput   = 0.0;
};

struct CoreStats {
    uint64_t cycles        = 0;
    uint64_t instrs        = 0;
    double   ipc           = 0.0;
    double   utilization   = 0.0;
    float    temperature   = 0.0f;
};
