#include <iostream>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>   // header-only JSON (add to deps)

#include "include/sim_types.h"
#include "include/thermal_rc.h"
#include "include/tsv_model.h"
#include "include/liquid_cooling.h"
#include "include/cache_sim.h"
#include "include/noc_sim.h"

using json = nlohmann::json;

struct SimResult {
    ThermalState thermal;
    std::vector<CacheStats> l1_stats;
    CacheStats              l2_stats;
    NoCStats                noc_stats;
    std::vector<CoreStats>  core_stats;
};

SimResult run_simulation(const SimConfig& cfg,
                         const std::vector<std::vector<float>>& power_map)
{
    SimResult result;

    // ── Thermal ────────────────────────────────────────────────────────────
    ThermalRC thermal(cfg.n_layers, cfg.grid_size,
                      0.08f, 0.15f, 0.002f, cfg.ambient_temp);
    ThermalState ts = thermal.solve_steady(power_map);

    LiquidCooling cooling(cfg.flow_rate);
    ts.temp_map = cooling.apply_cooling(ts.temp_map, power_map);
    ts.max_temp = 0.0f;
    for (auto& row : ts.temp_map)
        for (float t : row)
            ts.max_temp = std::max(ts.max_temp, t);
    ts.alert    = ts.max_temp >= 85.0f;
    ts.throttle = ts.max_temp >= 95.0f;
    result.thermal = ts;

    // ── TSV ────────────────────────────────────────────────────────────────
    TSVModel tsv(64, cfg.tsv_res);
    std::vector<float> tsv_flux(64);
    for (int i = 0; i < 64; i++)
        tsv_flux[i] = power_map[0][i % cfg.grid_size] * 0.1f;
    auto tsv_drops = tsv.compute_temp_drop(tsv_flux);
    float tsv_pen  = tsv.congestion_penalty(tsv_drops);

    // ── Caches ─────────────────────────────────────────────────────────────
    std::vector<L1Cache> l1s(cfg.n_cores, L1Cache(32, 64, 1, 4));
    L2Cache l2(512, 64, 4, 12, cfg.n_cores);

    // ── NoC ────────────────────────────────────────────────────────────────
    int mesh_dim = int(std::ceil(std::sqrt(double(cfg.n_chiplets))));
    NoCSimulator noc(mesh_dim, 2, 8, 128);

    // ── Core simulation (simplified instruction stream) ────────────────────
    result.core_stats.resize(cfg.n_cores);

    std::vector<uint32_t> addrs = {
        0x1000, 0x2000, 0x1040, 0x3000, 0x1000,
        0x4000, 0x2040, 0x1080, 0x5000, 0x1000
    };

    for (int cycle = 0; cycle < cfg.sim_cycles; cycle++) {
        for (int c = 0; c < cfg.n_cores; c++) {
            uint32_t addr    = addrs[(cycle + c) % addrs.size()];
            bool     is_wr   = (cycle % 5 == 0);
            bool     l1_hit  = l1s[c].access(addr, is_wr);
            if (!l1_hit) l2.access(c, addr, is_wr);

            result.core_stats[c].cycles++;
            result.core_stats[c].instrs += (cycle % 3 == 0) ? 2 : 1;
            result.core_stats[c].temperature = ts.temp_map[0][c % cfg.grid_size];
        }

        // Inject NoC traffic every 10 cycles
        if (cycle % 10 == 0) {
            int sx = cycle % mesh_dim, sy = (cycle / mesh_dim) % mesh_dim;
            int dx = (sx + 1) % mesh_dim, dy = sy;
            noc.inject(sx, sy, dx, dy, cycle, cycle);
        }
        noc.step(cycle);
    }

    for (int c = 0; c < cfg.n_cores; c++) {
        result.l1_stats.push_back(l1s[c].stats());
        result.core_stats[c].ipc = double(result.core_stats[c].instrs)
                                  / double(result.core_stats[c].cycles + 1);
        result.core_stats[c].utilization =
            double(l1s[c].stats().hits + l1s[c].stats().misses)
            / double(cfg.sim_cycles + 1);
    }
    result.l2_stats  = l2.stats();
    result.noc_stats = noc.stats();

    return result;
}

int main(int argc, char* argv[]) {
    SimConfig cfg;
    cfg.n_cores    = 4;
    cfg.n_chiplets = 16;
    cfg.n_layers   = 2;
    cfg.grid_size  = 16;
    cfg.sim_cycles = 100000;

    // Build synthetic power map
    std::vector<std::vector<float>> pmap(cfg.n_layers,
        std::vector<float>(cfg.grid_size, 0.0f));
    for (int l = 0; l < cfg.n_layers; l++)
        for (int n = 0; n < cfg.grid_size; n++)
            pmap[l][n] = float((l * cfg.grid_size + n) % 100) * 0.75f;

    SimResult res = run_simulation(cfg, pmap);

    // ── Report ─────────────────────────────────────────────────────────────
    std::cout << "\n=== FLAME_297 Hardware Simulation Report ===\n";
    std::cout << "Thermal: T_max=" << res.thermal.max_temp
              << "°C  Alert=" << res.thermal.alert
              << "  Throttle=" << res.thermal.throttle << "\n";
    for (int c = 0; c < cfg.n_cores; c++) {
        auto& cs = res.core_stats[c];
        auto& ls = res.l1_stats[c];
        std::cout << "Core " << c
                  << ": IPC=" << cs.ipc
                  << " Util=" << cs.utilization
                  << " T=" << cs.temperature << "°C"
                  << " L1_HR=" << ls.hit_rate() << "\n";
    }
    std::cout << "L2  : HR=" << res.l2_stats.hit_rate()
              << " Misses=" << res.l2_stats.misses << "\n";
    std::cout << "NoC : Flits=" << res.noc_stats.total_flits
              << " AvgLat=" << res.noc_stats.avg_latency
              << " Stalls=" << res.noc_stats.stall_cycles << "\n";

    return 0;
}
