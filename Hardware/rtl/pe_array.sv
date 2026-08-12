// 8×8 Processing Element array — SIMD MAC grid for RL policy inference
`timescale 1ns/1ps
module pe_array #(
    parameter ROWS       = 8,
    parameter COLS       = 8,
    parameter DATA_WIDTH = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,
    input  logic [DATA_WIDTH-1:0]         weight_in [0:ROWS-1][0:COLS-1],
    input  logic [DATA_WIDTH-1:0]         act_in    [0:ROWS-1],
    output logic [DATA_WIDTH-1:0]         result    [0:ROWS-1][0:COLS-1],
    output logic                          done,
    output logic [7:0]                    pe_util_map [0:ROWS-1][0:COLS-1]
);

    // Systolic array: each PE accumulates act × weight
    logic [DATA_WIDTH-1:0] pe_acc   [0:ROWS-1][0:COLS-1];
    logic [DATA_WIDTH-1:0] act_pipe [0:ROWS-1][0:COLS-1];
    logic                  active   [0:ROWS-1][0:COLS-1];
    logic [3:0]            cycle_ctr;

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : ROW_GEN
            for (c = 0; c < COLS; c++) begin : COL_GEN
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        pe_acc[r][c]    <= '0;
                        act_pipe[r][c]  <= '0;
                        active[r][c]    <= 1'b0;
                    end else if (start) begin
                        // Pipeline activation horizontally
                        act_pipe[r][c] <= (c == 0) ? act_in[r] : act_pipe[r][c-1];
                        pe_acc[r][c]   <= pe_acc[r][c] + act_pipe[r][c] * weight_in[r][c];
                        active[r][c]   <= 1'b1;
                    end else begin
                        active[r][c] <= 1'b0;
                    end
                end
                assign result[r][c]      = pe_acc[r][c];
                assign pe_util_map[r][c] = active[r][c] ? 8'd100 : 8'd0;
            end
        end
    endgenerate

    // Done after COLS pipeline cycles
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_ctr <= '0;
        else if (start)
            cycle_ctr <= cycle_ctr + 1;
        else
            cycle_ctr <= '0;
    end
    assign done = (cycle_ctr == 4'(COLS));

endmodule
