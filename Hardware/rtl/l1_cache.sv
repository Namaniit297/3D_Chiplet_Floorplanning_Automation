// L1 Cache — Direct-mapped, write-back, configurable size
`timescale 1ns/1ps
module l1_cache #(
    parameter CACHE_SIZE_KB = 32,
    parameter LINE_SIZE_B   = 64,
    parameter DATA_WIDTH    = 32,
    parameter ADDR_WIDTH    = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // CPU-side interface
    input  logic [ADDR_WIDTH-1:0]   cpu_addr,
    input  logic [DATA_WIDTH-1:0]   cpu_wdata,
    input  logic                    cpu_we,
    input  logic                    cpu_req,
    output logic [DATA_WIDTH-1:0]   cpu_rdata,
    output logic                    cpu_ack,
    output logic                    cpu_hit,

    // L2-side interface
    output logic [ADDR_WIDTH-1:0]   l2_addr,
    output logic [DATA_WIDTH-1:0]   l2_wdata,
    output logic                    l2_we,
    output logic                    l2_req,
    input  logic [DATA_WIDTH-1:0]   l2_rdata,
    input  logic                    l2_ack,

    // Performance counters
    output logic [31:0]             hit_count,
    output logic [31:0]             miss_count
);

    localparam NUM_LINES = (CACHE_SIZE_KB * 1024) / LINE_SIZE_B;
    localparam IDX_BITS  = $clog2(NUM_LINES);
    localparam OFF_BITS  = $clog2(LINE_SIZE_B / (DATA_WIDTH/8));
    localparam TAG_BITS  = ADDR_WIDTH - IDX_BITS - OFF_BITS;

    // Cache arrays
    logic [TAG_BITS-1:0]   tag_array   [0:NUM_LINES-1];
    logic [DATA_WIDTH-1:0] data_array  [0:NUM_LINES-1];
    logic                  valid_array [0:NUM_LINES-1];
    logic                  dirty_array [0:NUM_LINES-1];

    logic [TAG_BITS-1:0]  req_tag;
    logic [IDX_BITS-1:0]  req_idx;
    logic [OFF_BITS-1:0]  req_off;
    logic                 tag_match;
    logic [31:0]          hit_cnt, miss_cnt;

    assign req_tag = cpu_addr[ADDR_WIDTH-1 : IDX_BITS+OFF_BITS];
    assign req_idx = cpu_addr[IDX_BITS+OFF_BITS-1 : OFF_BITS];
    assign req_off = cpu_addr[OFF_BITS-1:0];
    assign tag_match = valid_array[req_idx] && (tag_array[req_idx] == req_tag);

    typedef enum logic [1:0] {IDLE, HIT_STATE, MISS_FILL, WRITEBACK} cache_state_t;
    cache_state_t state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            cpu_ack   <= 1'b0;
            cpu_hit   <= 1'b0;
            l2_req    <= 1'b0;
            hit_cnt   <= '0;
            miss_cnt  <= '0;
            for (int i = 0; i < NUM_LINES; i++) begin
                valid_array[i] <= 1'b0;
                dirty_array[i] <= 1'b0;
            end
        end else begin
            cpu_ack <= 1'b0;
            case (state)
                IDLE: begin
                    if (cpu_req) begin
                        if (tag_match) begin
                            cpu_hit   <= 1'b1;
                            hit_cnt   <= hit_cnt + 1;
                            state     <= HIT_STATE;
                        end else begin
                            cpu_hit  <= 1'b0;
                            miss_cnt <= miss_cnt + 1;
                            // Writeback if dirty
                            if (dirty_array[req_idx]) begin
                                l2_addr  <= {tag_array[req_idx], req_idx, {OFF_BITS{1'b0}}};
                                l2_wdata <= data_array[req_idx];
                                l2_we    <= 1'b1;
                                l2_req   <= 1'b1;
                                state    <= WRITEBACK;
                            end else begin
                                l2_addr <= cpu_addr;
                                l2_we   <= 1'b0;
                                l2_req  <= 1'b1;
                                state   <= MISS_FILL;
                            end
                        end
                    end
                end

                HIT_STATE: begin
                    if (cpu_we) begin
                        data_array[req_idx]  <= cpu_wdata;
                        dirty_array[req_idx] <= 1'b1;
                    end else begin
                        cpu_rdata <= data_array[req_idx];
                    end
                    cpu_ack <= 1'b1;
                    state   <= IDLE;
                end

                MISS_FILL: begin
                    if (l2_ack) begin
                        l2_req              <= 1'b0;
                        data_array[req_idx] <= l2_rdata;
                        tag_array[req_idx]  <= req_tag;
                        valid_array[req_idx]<= 1'b1;
                        dirty_array[req_idx]<= cpu_we;
                        if (cpu_we)
                            data_array[req_idx] <= cpu_wdata;
                        else
                            cpu_rdata <= l2_rdata;
                        cpu_ack <= 1'b1;
                        state   <= IDLE;
                    end
                end

                WRITEBACK: begin
                    if (l2_ack) begin
                        l2_req  <= 1'b0;
                        l2_addr <= cpu_addr;
                        l2_we   <= 1'b0;
                        l2_req  <= 1'b1;
                        state   <= MISS_FILL;
                    end
                end
            endcase
        end
    end

    assign hit_count  = hit_cnt;
    assign miss_count = miss_cnt;

endmodule
