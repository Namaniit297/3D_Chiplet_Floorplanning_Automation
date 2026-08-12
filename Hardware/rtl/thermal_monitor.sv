// On-chip thermal sensor with alert + throttle output
`timescale 1ns/1ps
module thermal_monitor #(
    parameter N_SENSORS    = 4,
    parameter SAMPLE_RATE  = 1000,  // cycles between samples
    parameter ALERT_THRESH = 85,    // °C
    parameter THROT_THRESH = 95     // °C
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic [7:0]        temp_in [0:N_SENSORS-1],  // encoded °C
    output logic [7:0]        max_temp,
    output logic [7:0]        avg_temp,
    output logic              thermal_alert,
    output logic              thermal_throttle,
    output logic [N_SENSORS-1:0] sensor_alert
);

    logic [7:0]  max_t, avg_t;
    logic [15:0] sum_t;
    logic [31:0] sample_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_t      <= '0;
            avg_t      <= '0;
            sample_cnt <= '0;
        end else begin
            sample_cnt <= sample_cnt + 1;
            if (sample_cnt == SAMPLE_RATE) begin
                sample_cnt <= '0;
                sum_t = '0;
                max_t = '0;
                for (int s = 0; s < N_SENSORS; s++) begin
                    sum_t = sum_t + temp_in[s];
                    if (temp_in[s] > max_t) max_t = temp_in[s];
                end
                avg_t <= 8'(sum_t / N_SENSORS);
                max_t <= max_t;
            end
        end
    end

    always_comb begin
        for (int s = 0; s < N_SENSORS; s++)
            sensor_alert[s] = (temp_in[s] >= ALERT_THRESH);
        thermal_alert    = (max_t >= ALERT_THRESH);
        thermal_throttle = (max_t >= THROT_THRESH);
    end

    assign max_temp = max_t;
    assign avg_temp = avg_t;

endmodule
