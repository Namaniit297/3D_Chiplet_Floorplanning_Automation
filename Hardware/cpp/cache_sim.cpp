#include "include/cache_sim.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ─────────────────────────── L1 Cache ────────────────────────────────────

L1Cache::L1Cache(int size_kb, int line_bytes, int ways, int latency)
    : size_kb_(size_kb), line_bytes_(line_bytes),
      ways_(ways), latency_(latency)
{
    n_sets_ = (size_kb * 1024) / (line_bytes * ways);
    sets_.resize(n_sets_, std::vector<Line>(ways, {0, false, false}));
}

uint32_t L1Cache::get_tag(uint32_t addr) const {
    int off = int(std::log2(line_bytes_));
    int idx = int(std::log2(n_sets_));
    return addr >> (off + idx);
}

uint32_t L1Cache::get_set(uint32_t addr) const {
    int off = int(std::log2(line_bytes_));
    return (addr >> off) & (n_sets_ - 1);
}

bool L1Cache::access(uint32_t addr, bool is_write) {
    uint32_t tag = get_tag(addr);
    uint32_t set = get_set(addr);
    for (auto& line : sets_[set]) {
        if (line.valid && line.tag == tag) {
            if (is_write) line.dirty = true;
            stats_.hits++;
            return true;
        }
    }
    // Miss: install (LRU = first invalid or way 0)
    stats_.misses++;
    for (auto& line : sets_[set]) {
        if (!line.valid) {
            line = {tag, true, is_write};
            return false;
        }
    }
    // Evict way 0
    if (sets_[set][0].dirty) stats_.evictions++;
    sets_[set][0] = {tag, true, is_write};
    return false;
}

void L1Cache::reset() {
    for (auto& s : sets_) for (auto& l : s) l = {0, false, false};
    stats_ = {};
}

// ─────────────────────────── L2 Cache ────────────────────────────────────

L2Cache::L2Cache(int size_kb, int line_bytes, int ways,
                 int latency, int n_ports)
    : size_kb_(size_kb), line_bytes_(line_bytes),
      ways_(ways), latency_(latency), n_ports_(n_ports)
{
    n_sets_ = (size_kb * 1024) / (line_bytes * ways);
    sets_.resize(n_sets_, std::vector<Line>(ways, {0, false, false, 0}));
}

uint32_t L2Cache::get_tag(uint32_t addr) const {
    int off = int(std::log2(line_bytes_));
    int idx = int(std::log2(n_sets_));
    return addr >> (off + idx);
}

uint32_t L2Cache::get_set(uint32_t addr) const {
    int off = int(std::log2(line_bytes_));
    return (addr >> off) & (n_sets_ - 1);
}

int L2Cache::find_lru(int set_idx) const {
    int lru_way = 0, min_ctr = sets_[set_idx][0].lru_counter;
    for (int w = 1; w < ways_; w++)
        if (sets_[set_idx][w].lru_counter < min_ctr) {
            min_ctr = sets_[set_idx][w].lru_counter;
            lru_way = w;
        }
    return lru_way;
}

bool L2Cache::access(int /*port_id*/, uint32_t addr, bool is_write) {
    uint32_t tag = get_tag(addr);
    uint32_t set = get_set(addr);
    static int global_ctr = 0;
    ++global_ctr;

    for (auto& line : sets_[set]) {
        if (line.valid && line.tag == tag) {
            if (is_write) line.dirty = true;
            line.lru_counter = global_ctr;
            stats_.hits++;
            return true;
        }
    }
    stats_.misses++;
    int evict = find_lru(set);
    if (sets_[set][evict].dirty) stats_.evictions++;
    sets_[set][evict] = {tag, true, is_write, global_ctr};
    return false;
}

void L2Cache::reset() {
    for (auto& s : sets_)
        for (auto& l : s) l = {0, false, false, 0};
    stats_ = {};
}
