#pragma once
#include <vector>
#include <unordered_map>
#include <list>
#include "sim_types.h"

class L1Cache {
public:
    L1Cache(int size_kb = 32, int line_bytes = 64,
            int ways = 1, int latency_cycles = 4);
    bool access(uint32_t addr, bool is_write);
    CacheStats stats() const { return stats_; }
    void reset();

private:
    int size_kb_, line_bytes_, ways_, latency_;
    int n_sets_;
    struct Line { uint32_t tag; bool valid; bool dirty; };
    std::vector<std::vector<Line>> sets_;
    CacheStats stats_;
    uint32_t get_tag(uint32_t addr)  const;
    uint32_t get_set(uint32_t addr)  const;
};

class L2Cache {
public:
    L2Cache(int size_kb = 512, int line_bytes = 64,
            int ways = 4, int latency_cycles = 12,
            int n_ports = 4);
    bool access(int port_id, uint32_t addr, bool is_write);
    CacheStats stats() const { return stats_; }
    void reset();

private:
    int size_kb_, line_bytes_, ways_, latency_, n_ports_;
    int n_sets_;
    struct Line {
        uint32_t tag;
        bool valid, dirty;
        int lru_counter;
    };
    std::vector<std::vector<Line>> sets_;
    CacheStats stats_;
    uint32_t get_tag(uint32_t addr) const;
    uint32_t get_set(uint32_t addr) const;
    int find_lru(int set_idx) const;
};
