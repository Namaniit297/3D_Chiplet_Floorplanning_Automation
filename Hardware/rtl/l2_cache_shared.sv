// Shared L2 Cache — 4-way set-associative, shared across all APU cores
`timescale 1ns/1ps
module l2_cache_shared #(
    parameter CACHE_SIZE_KB = 512,
    parameter LINE_SIZE_B   = 64,
    parameter WAYS          = 4,
    parameter DATA_WIDTH    = 32,
    parameter ADDR_WIDTH    = 32,
    parameter NUM_PORTS     = 4   // One per APU core
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Per-core L1 interfaces (arbitrated)
    input  logic [ADDR_WIDTH-1:0]   l1_addr   [0:NUM_PORTS-1],
    input  logic [DATA_WIDTH-1:0]   l1_wdata  [0:NUM_PORTS-1],
    input  logic                    l1_we     [0:NUM_PORTS-1],
    input  logic                    l1_req    [0:NUM_PORTS-1],
    output logic [DATA_WIDTH-1:0]   l1_rdata  [0:NUM_PORTS-1],
    output logic                    l1_ack    [0:NUM_PORTS-1],

    // DRAM interface
    output logic [ADDR_WIDTH-1:0]   dram_addr,
    output logic [DATA_WIDTH-1:0]   dram_wdata,
    output logic                    dram_we,
    output logic                    dram_req,
    input  logic [DATA_WIDTH-1:0]   dram_rdata,
    input  logic                    dram_ack,

    // Stats
    output logic [31:0]             total_hits,
    output logic [31:0]             total_misses
);

    localparam NUM_SETS  = (CACHE_SIZE_KB * 1024) / (LINE_SIZE_B * WAYS);
    localparam IDX_BITS  = $clog2(NUM_SETS);
    localparam OFF_BITS  = $clog2(LINE_SIZE_B / (DATA_WIDTH/8));
    localparam TAG_BITS  = ADDR_WIDTH - IDX_BITS - OFF_BITS;

    // Cache storage (ways × sets)
    logic [TAG_BITS-1:0]    tags   [0:WAYS-1][0:NUM_SETS-1];
    logic [DATA_WIDTH-1:0]  data   [0:WAYS-1][0:NUM_SETS-1];
    logic                   valid  [0:WAYS-1][0:NUM_SETS-1];
    logic                   dirty  [0:WAYS-1][0:NUM_SETS-1];
    logic [1:0]             lru    [0:NUM_SETS-1]; // 2-bit LRU way pointer

    // Arbitration: round-robin
    logic [1:0] grant;
    logic [1:0] arb_ptr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) arb_ptr <= '0;
        else begin
            // Advance to next requesting port
            arb_ptr <= arb_ptr + 1;
        end
    end

    // Find grant: first requesting port from arb_ptr
    always_comb begin
        grant = arb_ptr;
        for (int k = 0; k < NUM_PORTS; k++) begin
            int idx = (arb_ptr + k) % NUM_PORTS;
            if (l1_req[idx]) begin
                grant = 2'(idx);
                break;
            end
        end
    end

    // Lookup logic (combinatorial for granted port)
    logic [TAG_BITS-1:0] g_tag;
    logic [IDX_BITS-1:0] g_idx;
    logic [1:0]          hit_way;
    logic                hit;

    assign g_tag = l1_addr[grant][ADDR_WIDTH-1 : IDX_BITS+OFF_BITS];
    assign g_idx = l1_addr[grant][IDX_BITS+OFF_BITS-1 : OFF_BITS];

    always_comb begin
        hit     = 1'b0;
        hit_way = '0;
        for (int w = 0; w < WAYS; w++) begin
            if (valid[w][g_idx] && tags[w][g_idx] == g_tag) begin
                hit     = 1'b1;
                hit_way = 2'(w);
            end
        end
    end

    logic [31:0] hit_cnt, miss_cnt;

    typedef enum logic [1:0] {IDLE, SERVE, FILL, EVICT} l2_state_t;
    l2_state_t state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            hit_cnt    <= '0;
            miss_cnt   <= '0;
            dram_req   <= 1'b0;
            for (int p = 0; p < NUM_PORTS; p++) l1_ack[p] <= 1'b0;
            for (int w = 0; w < WAYS; w++)
                for (int s = 0; s < NUM_SETS; s++) begin
                    valid[w][s] <= 1'b0;
                    dirty[w][s] <= 1'b0;
                end
        end else begin
            for (int p = 0; p < NUM_PORTS; p++) l1_ack[p] <= 1'b0;

            case (state)
                IDLE: begin
                    if (|l1_req) state <= SERVE;
                end

                SERVE: begin
                    if (hit) begin
                        hit_cnt <= hit_cnt + 1;
                        if (l1_we[grant]) begin
                            data[hit_way][g_idx]  <= l1_wdata[grant];
                            dirty[hit_way][g_idx] <= 1'b1;
                        end else begin
                            l1_rdata[grant] <= data[hit_way][g_idx];
                        end
                        lru[g_idx]   <= ~hit_way[0];  // Simple LRU update
                        l1_ack[grant]<= 1'b1;
                        state        <= IDLE;
                    end else begin
                        miss_cnt <= miss_cnt + 1;
                        // Evict LRU way if dirty
                        if (dirty[lru[g_idx]][g_idx]) begin
                            dram_addr  <= {tags[lru[g_idx]][g_idx], g_idx, {OFF_BITS{1'b0}}};
                            dram_wdata <= data[lru[g_idx]][g_idx];
                            dram_we    <= 1'b1;
                            dram_req   <= 1'b1;
                            state      <= EVICT;
                        end else begin
                            dram_addr <= l1_addr[grant];
                            dram_we   <= 1'b0;
                            dram_req  <= 1'b1;
                            state     <= FILL;
                        end
                    end
                end

                EVICT: begin
                    if (dram_ack) begin
                        dram_req  <= 1'b0;
                        dram_addr <= l1_addr[grant];
                        dram_we   <= 1'b0;
                        dram_req  <= 1'b1;
                        dirty[lru[g_idx]][g_idx] <= 1'b0;
                        state     <= FILL;
                    end
                end

                FILL: begin
                    if (dram_ack) begin
                        dram_req <= 1'b0;
                        data [lru[g_idx]][g_idx] <= dram_rdata;
                        tags [lru[g_idx]][g_idx] <= g_tag;
                        valid[lru[g_idx]][g_idx] <= 1'b1;
                        dirty[lru[g_idx]][g_idx] <= l1_we[grant];
                        if (!l1_we[grant])
                            l1_rdata[grant] <= dram_rdata;
                        l1_ack[grant] <= 1'b1;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

    assign total_hits   = hit_cnt;
    assign total_misses = miss_cnt;

endmodule
