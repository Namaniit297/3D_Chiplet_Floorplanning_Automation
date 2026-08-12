// Top-level APU: N cores + PE array + L1 per core + Shared L2 + NoC + Thermal monitor
`timescale 1ns/1ps
module top_apu #(
    parameter N_CORES    = 4,
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32
)(
    input  logic clk,
    input  logic rst_n,

    // DRAM interface
    output logic [ADDR_WIDTH-1:0]  dram_addr,
    output logic [DATA_WIDTH-1:0]  dram_wdata,
    output logic                   dram_we,
    output logic                   dram_req,
    input  logic [DATA_WIDTH-1:0]  dram_rdata,
    input  logic                   dram_ack,

    // Global thermal outputs
    output logic [7:0]  max_chip_temp,
    output logic        global_thermal_alert,
    output logic        global_throttle,

    // Performance summary
    output logic [63:0] total_cycles,
    output logic [63:0] total_instrs,
    output logic [31:0] l2_hits,
    output logic [31:0] l2_misses
);

    // -----------------------------------------------------------------------
    // Instruction stimulus (simplified: tie to known pattern for simulation)
    // -----------------------------------------------------------------------
    logic [31:0] instr_in    [0:N_CORES-1];
    logic        instr_valid [0:N_CORES-1];
    logic        instr_ready [0:N_CORES-1];

    // -----------------------------------------------------------------------
    // L1 ↔ L2 wires
    // -----------------------------------------------------------------------
    logic [ADDR_WIDTH-1:0]  l1_l2_addr  [0:N_CORES-1];
    logic [DATA_WIDTH-1:0]  l1_l2_wdata [0:N_CORES-1];
    logic                   l1_l2_we    [0:N_CORES-1];
    logic                   l1_l2_req   [0:N_CORES-1];
    logic [DATA_WIDTH-1:0]  l2_l1_rdata [0:N_CORES-1];
    logic                   l2_l1_ack   [0:N_CORES-1];

    // -----------------------------------------------------------------------
    // Thermal sensor wires
    // -----------------------------------------------------------------------
    logic [7:0] core_temp [0:N_CORES-1];

    // -----------------------------------------------------------------------
    // APU Cores
    // -----------------------------------------------------------------------
    genvar g;
    generate
        for (g = 0; g < N_CORES; g++) begin : CORES
            logic [31:0] mem_addr_w;
            logic [DATA_WIDTH-1:0] mem_wdata_w, mem_rdata_w;
            logic mem_we_w, mem_req_w, mem_ack_w;
            logic [63:0] cc_w, ir_w;
            logic [31:0] util_w;

            apu_core #(.CORE_ID(g)) u_core (
                .clk         (clk),
                .rst_n       (rst_n),
                .instr_in    (instr_in[g]),
                .instr_valid (instr_valid[g]),
                .instr_ready (instr_ready[g]),
                .mem_addr    (mem_addr_w),
                .mem_wdata   (mem_wdata_w),
                .mem_we      (mem_we_w),
                .mem_req     (mem_req_w),
                .mem_rdata   (mem_rdata_w),
                .mem_ack     (mem_ack_w),
                .temp_code   (core_temp[g]),
                .thermal_alert(),
                .cycle_count (cc_w),
                .instr_retired(ir_w),
                .util_percent (util_w)
            );

            // L1 Cache per core
            l1_cache u_l1 (
                .clk       (clk),
                .rst_n     (rst_n),
                .cpu_addr  (mem_addr_w),
                .cpu_wdata (mem_wdata_w),
                .cpu_we    (mem_we_w),
                .cpu_req   (mem_req_w),
                .cpu_rdata (mem_rdata_w),
                .cpu_ack   (mem_ack_w),
                .cpu_hit   (),
                .l2_addr   (l1_l2_addr[g]),
                .l2_wdata  (l1_l2_wdata[g]),
                .l2_we     (l1_l2_we[g]),
                .l2_req    (l1_l2_req[g]),
                .l2_rdata  (l2_l1_rdata[g]),
                .l2_ack    (l2_l1_ack[g]),
                .hit_count (),
                .miss_count()
            );
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Shared L2 Cache
    // -----------------------------------------------------------------------
    l2_cache_shared #(.NUM_PORTS(N_CORES)) u_l2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .l1_addr    (l1_l2_addr),
        .l1_wdata   (l1_l2_wdata),
        .l1_we      (l1_l2_we),
        .l1_req     (l1_l2_req),
        .l1_rdata   (l2_l1_rdata),
        .l1_ack     (l2_l1_ack),
        .dram_addr  (dram_addr),
        .dram_wdata (dram_wdata),
        .dram_we    (dram_we),
        .dram_req   (dram_req),
        .dram_rdata (dram_rdata),
        .dram_ack   (dram_ack),
        .total_hits (l2_hits),
        .total_misses(l2_misses)
    );

    // -----------------------------------------------------------------------
    // Thermal Monitor
    // -----------------------------------------------------------------------
    thermal_monitor #(.N_SENSORS(N_CORES)) u_therm (
        .clk              (clk),
        .rst_n            (rst_n),
        .temp_in          (core_temp),
        .max_temp         (max_chip_temp),
        .avg_temp         (),
        .thermal_alert    (global_thermal_alert),
        .thermal_throttle (global_throttle),
        .sensor_alert     ()
    );

    // Simplified: sum cycles/instrs from first core only (extend as needed)
    assign total_cycles = CORES[0].cc_w;
    assign total_instrs = CORES[0].ir_w;

endmodule
