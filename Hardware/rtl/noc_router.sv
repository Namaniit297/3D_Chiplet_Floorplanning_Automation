// NoC Router — 5-port wormhole router (N/S/E/W/Local)
// Used for chiplet-to-chiplet communication
`timescale 1ns/1ps
module noc_router #(
    parameter FLIT_WIDTH  = 64,
    parameter VC_COUNT    = 2,
    parameter BUFFER_DEEP = 8,
    parameter ROUTER_ID_X = 0,
    parameter ROUTER_ID_Y = 0,
    parameter MESH_DIM    = 4
)(
    input  logic                   clk,
    input  logic                   rst_n,

    // 5 ports: 0=Local, 1=North, 2=South, 3=East, 4=West
    input  logic [FLIT_WIDTH-1:0]  flit_in  [0:4],
    input  logic                   flit_valid_in  [0:4],
    output logic                   flit_credit_out[0:4],

    output logic [FLIT_WIDTH-1:0]  flit_out [0:4],
    output logic                   flit_valid_out [0:4],
    input  logic                   flit_credit_in [0:4],

    // Performance counters
    output logic [31:0]            flit_count,
    output logic [31:0]            stall_count
);

    // Flit format: [63:56]=dest_x, [55:48]=dest_y, [47:0]=payload
    localparam LOCAL=0, NORTH=1, SOUTH=2, EAST=3, WEST=4;

    // Input buffers (FIFOs per port)
    logic [FLIT_WIDTH-1:0] buf_data [0:4][0:BUFFER_DEEP-1];
    logic [2:0]            buf_head [0:4];
    logic [2:0]            buf_tail [0:4];
    logic [3:0]            buf_cnt  [0:4];

    logic [FLIT_WIDTH-1:0] active_flit [0:4];
    logic                  active_val  [0:4];
    logic [2:0]            route_port  [0:4];  // output port for each input

    // XY routing
    function automatic logic [2:0] xy_route(
        input logic [7:0] dest_x,
        input logic [7:0] dest_y
    );
        if      (dest_x > ROUTER_ID_X) return EAST;
        else if (dest_x < ROUTER_ID_X) return WEST;
        else if (dest_y > ROUTER_ID_Y) return NORTH;
        else if (dest_y < ROUTER_ID_Y) return SOUTH;
        else                           return LOCAL;
    endfunction

    logic [31:0] flit_cnt, stall_cnt;

    // Enqueue incoming flits
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int p = 0; p < 5; p++) begin
                buf_head[p] <= '0;
                buf_tail[p] <= '0;
                buf_cnt[p]  <= '0;
                flit_valid_out[p] <= 1'b0;
            end
            flit_cnt  <= '0;
            stall_cnt <= '0;
        end else begin
            // Enqueue
            for (int p = 0; p < 5; p++) begin
                if (flit_valid_in[p] && buf_cnt[p] < BUFFER_DEEP) begin
                    buf_data[p][buf_tail[p]] <= flit_in[p];
                    buf_tail[p] <= buf_tail[p] + 1;
                    buf_cnt[p]  <= buf_cnt[p]  + 1;
                    flit_cnt    <= flit_cnt + 1;
                end else if (flit_valid_in[p] && buf_cnt[p] == BUFFER_DEEP) begin
                    stall_cnt <= stall_cnt + 1;
                end
                flit_credit_out[p] <= (buf_cnt[p] < BUFFER_DEEP);
            end

            // Dequeue + route (simple round-robin arbitration)
            for (int p = 0; p < 5; p++) begin
                if (buf_cnt[p] > 0) begin
                    logic [7:0] dx, dy;
                    logic [2:0] out_p;
                    dx    = buf_data[p][buf_head[p]][63:56];
                    dy    = buf_data[p][buf_head[p]][55:48];
                    out_p = xy_route(dx, dy);
                    if (flit_credit_in[out_p]) begin
                        flit_out[out_p]       <= buf_data[p][buf_head[p]];
                        flit_valid_out[out_p] <= 1'b1;
                        buf_head[p]           <= buf_head[p] + 1;
                        buf_cnt[p]            <= buf_cnt[p]  - 1;
                    end else begin
                        flit_valid_out[out_p] <= 1'b0;
                    end
                end
            end
        end
    end

    assign flit_count  = flit_cnt;
    assign stall_count = stall_cnt;

endmodule
