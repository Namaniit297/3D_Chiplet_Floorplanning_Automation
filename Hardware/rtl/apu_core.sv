// APU Core — Manycore processing element for RL policy inference
// Supports configurable ISA width, FP32/INT8 MAC units, local scratchpad

`timescale 1ns/1ps
module apu_core #(
    parameter CORE_ID       = 0,
    parameter DATA_WIDTH    = 32,
    parameter SCRATCHPAD_KB = 16,
    parameter MAC_UNITS     = 8
)(
    input  logic                   clk,
    input  logic                   rst_n,

    // Instruction interface
    input  logic [31:0]            instr_in,
    input  logic                   instr_valid,
    output logic                   instr_ready,

    // Memory interface (to L1 cache)
    output logic [31:0]            mem_addr,
    output logic [DATA_WIDTH-1:0]  mem_wdata,
    output logic                   mem_we,
    output logic                   mem_req,
    input  logic [DATA_WIDTH-1:0]  mem_rdata,
    input  logic                   mem_ack,

    // Thermal monitor output
    output logic [7:0]             temp_code,      // 8-bit encoded temp
    output logic                   thermal_alert,

    // Performance counters
    output logic [63:0]            cycle_count,
    output logic [63:0]            instr_retired,
    output logic [31:0]            util_percent    // 0-100 fixed point
);

    // -----------------------------------------------------------------------
    // Pipeline registers
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        FETCH, DECODE, EXECUTE, MEM_ACCESS, WRITEBACK
    } pipe_stage_t;

    logic [31:0] pc, pc_next;
    logic [31:0] ir;                        // instruction register
    logic [DATA_WIDTH-1:0] alu_result;
    logic [DATA_WIDTH-1:0] reg_file [0:31]; // 32 general-purpose registers
    logic [DATA_WIDTH-1:0] acc_reg;         // MAC accumulator

    // -----------------------------------------------------------------------
    // Scratchpad SRAM (synthesisable single-port)
    // -----------------------------------------------------------------------
    localparam SP_DEPTH = (SCRATCHPAD_KB * 1024) / (DATA_WIDTH / 8);
    logic [DATA_WIDTH-1:0] scratchpad [0:SP_DEPTH-1];

    // -----------------------------------------------------------------------
    // MAC unit array  (FP32 multiply-accumulate)
    // -----------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] mac_a [0:MAC_UNITS-1];
    logic [DATA_WIDTH-1:0] mac_b [0:MAC_UNITS-1];
    logic [DATA_WIDTH-1:0] mac_out[0:MAC_UNITS-1];
    logic                  mac_valid;

    genvar gi;
    generate
        for (gi = 0; gi < MAC_UNITS; gi++) begin : MAC_ARRAY
            // Simplified: integer multiply for synthesis (replace with FP IP)
            assign mac_out[gi] = mac_a[gi] * mac_b[gi];
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Simple decode (RISC-V RV32I subset + custom MAC opcode)
    // -----------------------------------------------------------------------
    logic [6:0]  opcode;
    logic [4:0]  rd, rs1, rs2;
    logic [11:0] imm12;
    logic [2:0]  funct3;

    assign opcode = ir[6:0];
    assign rd     = ir[11:7];
    assign rs1    = ir[19:15];
    assign rs2    = ir[24:20];
    assign funct3 = ir[14:12];
    assign imm12  = ir[31:20];

    // -----------------------------------------------------------------------
    // Pipeline FSM
    // -----------------------------------------------------------------------
    pipe_stage_t stage;
    logic [63:0] cycle_cnt_r;
    logic [63:0] instr_ret_r;
    logic [31:0] active_cycles;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage         <= FETCH;
            pc            <= 32'h0000_0000;
            cycle_cnt_r   <= '0;
            instr_ret_r   <= '0;
            active_cycles <= '0;
            acc_reg       <= '0;
            for (int i = 0; i < 32; i++) reg_file[i] <= '0;
        end else begin
            cycle_cnt_r <= cycle_cnt_r + 1;

            case (stage)
                FETCH: begin
                    if (instr_valid) begin
                        ir    <= instr_in;
                        stage <= DECODE;
                        active_cycles <= active_cycles + 1;
                    end
                end

                DECODE: begin
                    // Operand fetch happens combinatorially
                    stage <= EXECUTE;
                end

                EXECUTE: begin
                    case (opcode)
                        7'b0110011: begin  // R-type ALU
                            case (funct3)
                                3'b000: alu_result <= reg_file[rs1] + reg_file[rs2]; // ADD
                                3'b010: alu_result <= reg_file[rs1] - reg_file[rs2]; // SUB
                                3'b110: alu_result <= reg_file[rs1] | reg_file[rs2]; // OR
                                3'b111: alu_result <= reg_file[rs1] & reg_file[rs2]; // AND
                                default: alu_result <= '0;
                            endcase
                            stage <= WRITEBACK;
                        end

                        7'b0010011: begin  // I-type ALU
                            alu_result <= reg_file[rs1] + {{20{imm12[11]}}, imm12};
                            stage <= WRITEBACK;
                        end

                        7'b0000011: begin  // LOAD
                            mem_addr <= reg_file[rs1] + {{20{imm12[11]}}, imm12};
                            mem_we   <= 1'b0;
                            mem_req  <= 1'b1;
                            stage    <= MEM_ACCESS;
                        end

                        7'b0100011: begin  // STORE
                            mem_addr  <= reg_file[rs1] + {{20{imm12[11]}}, imm12};
                            mem_wdata <= reg_file[rs2];
                            mem_we    <= 1'b1;
                            mem_req   <= 1'b1;
                            stage     <= MEM_ACCESS;
                        end

                        7'b1111111: begin  // Custom MAC opcode
                            mac_a[0] <= reg_file[rs1];
                            mac_b[0] <= reg_file[rs2];
                            acc_reg  <= acc_reg + mac_out[0];
                            stage    <= WRITEBACK;
                        end

                        default: stage <= WRITEBACK;
                    endcase
                end

                MEM_ACCESS: begin
                    if (mem_ack) begin
                        mem_req <= 1'b0;
                        if (!mem_we)
                            reg_file[rd] <= mem_rdata;
                        stage <= WRITEBACK;
                    end
                end

                WRITEBACK: begin
                    if (opcode == 7'b0110011 || opcode == 7'b0010011)
                        reg_file[rd] <= alu_result;
                    instr_ret_r <= instr_ret_r + 1;
                    stage <= FETCH;
                end
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Thermal monitor: simple ring-oscillator proxy
    // Uses active_cycles/cycle_cnt ratio as thermal proxy
    // -----------------------------------------------------------------------
    always_comb begin
        if (cycle_cnt_r == 0)
            util_percent = 0;
        else
            util_percent = (active_cycles * 100) / cycle_cnt_r[31:0];

        // Encode temperature: T = 40 + util * 0.6 → 8-bit (max ~100°C)
        temp_code = 8'(40 + (util_percent * 6) / 10);
        thermal_alert = (temp_code > 8'd85);
    end

    assign cycle_count   = cycle_cnt_r;
    assign instr_retired = instr_ret_r;
    assign instr_ready   = (stage == FETCH);

endmodule
