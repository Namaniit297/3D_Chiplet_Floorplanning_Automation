#pragma once
#include <vector>
#include <queue>
#include "sim_types.h"

struct Flit {
    int src_x, src_y, dest_x, dest_y;
    int payload;
    int inject_cycle;
};

class NoCSimulator {
public:
    NoCSimulator(int mesh_dim = 4, int vc_count = 2,
                 int buffer_depth = 8, int link_bw_gbps = 128);

    void inject(int src_x, int src_y, int dst_x, int dst_y,
                int payload, int cycle);
    void step(int cycle);       // advance one cycle
    NoCStats stats() const;
    void reset();

private:
    int dim_, vc_, buf_depth_, bw_;
    struct RouterPort {
        std::queue<Flit> buf;
    };
    // [x][y][port(0=L,1=N,2=S,3=E,4=W)]
    std::vector<std::vector<std::vector<RouterPort>>> routers_;

    NoCStats stats_;
    double total_latency_;
    int    delivered_;

    int xy_route_port(int cx, int cy, int dx, int dy) const;
};
