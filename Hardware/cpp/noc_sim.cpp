#include "include/noc_sim.h"
#include <algorithm>
#include <cmath>

NoCSimulator::NoCSimulator(int mesh_dim, int vc_count,
                           int buffer_depth, int link_bw_gbps)
    : dim_(mesh_dim), vc_(vc_count),
      buf_depth_(buffer_depth), bw_(link_bw_gbps),
      total_latency_(0.0), delivered_(0)
{
    routers_.resize(dim_, std::vector<std::vector<RouterPort>>(
        dim_, std::vector<RouterPort>(5)));
}

int NoCSimulator::xy_route_port(int cx, int cy, int dx, int dy) const {
    if      (dx > cx) return 3;  // EAST
    else if (dx < cx) return 4;  // WEST
    else if (dy > cy) return 1;  // NORTH
    else if (dy < cy) return 2;  // SOUTH
    else              return 0;  // LOCAL
}

void NoCSimulator::inject(int sx, int sy, int dx, int dy,
                          int payload, int cycle)
{
    Flit f{sx, sy, dx, dy, payload, cycle};
    if (sx < dim_ && sy < dim_)
        routers_[sx][sy][0].buf.push(f);
    stats_.total_flits++;
}

void NoCSimulator::step(int cycle) {
    // Move flits one hop per cycle
    std::vector<std::vector<std::vector<RouterPort>>> next = routers_;

    for (int x = 0; x < dim_; x++) {
        for (int y = 0; y < dim_; y++) {
            for (int p = 0; p < 5; p++) {
                auto& port = routers_[x][y][p];
                if (port.buf.empty()) continue;
                Flit f = port.buf.front();

                int out_p = xy_route_port(x, y, f.dest_x, f.dest_y);
                if (out_p == 0) {
                    // Delivered
                    total_latency_ += double(cycle - f.inject_cycle);
                    delivered_++;
                    port.buf.pop();
                    continue;
                }
                // Compute next router
                int nx = x, ny = y;
                if (out_p == 3) nx++;
                else if (out_p == 4) nx--;
                else if (out_p == 1) ny++;
                else if (out_p == 2) ny--;

                if (nx >= 0 && nx < dim_ && ny >= 0 && ny < dim_) {
                    auto& nport = next[nx][ny][0];
                    if ((int)nport.buf.size() < buf_depth_) {
                        nport.buf.push(f);
                        port.buf.pop();
                    } else {
                        stats_.stall_cycles++;
                    }
                }
            }
        }
    }
    routers_ = next;
}

NoCStats NoCSimulator::stats() const {
    NoCStats s = stats_;
    s.avg_latency  = delivered_ > 0 ? total_latency_ / delivered_ : 0.0;
    s.throughput   = delivered_ > 0
        ? double(delivered_) / double(stats_.total_flits + 1) * bw_
        : 0.0;
    return s;
}

void NoCSimulator::reset() {
    for (auto& row : routers_)
        for (auto& col : row)
            for (auto& p : col)
                while (!p.buf.empty()) p.buf.pop();
    stats_       = {};
    total_latency_= 0.0;
    delivered_   = 0;
}
