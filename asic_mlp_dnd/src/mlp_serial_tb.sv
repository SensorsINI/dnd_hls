// This file is part of https://github.com/SensorsINI/dnd_hls. 
// This intellectual property is licensed under the terms of the project license available at the root of the project.
module mlp_serial_tb;
  timeunit 1ns/1ps;

  localparam  CLK_PERIOD = 10,
              N1 = 98,
              N2 = 20,
              P  = 2,
              W_X = 4,
              W_K = 4,
              W_Y = 17;

  localparam  N_BEATS       = (1+N1/2)/2;

  logic clk=0, rst=1, in_vld=0, out_vld;
  logic [P   -1:0][W_X-1:0] in_mag;
  logic [P   -1:0][1:0]     in_pol;
  logic [W_Y   :0]          out;
  int status, file;

  initial forever #(CLK_PERIOD/2) clk <= ~clk;
  mlp_serial #(.N1(N1),.N2(N2),.P(P),.W_X(W_X),.W_Y(W_Y)) dut (.*);

  initial begin

    repeat (2) @(posedge clk);
    rst <= 0;
    repeat (2) @(posedge clk);

    file = $fopen("input.txt", "r");
    repeat (N_BEATS) begin
      @(posedge clk); #1

      in_vld = 1;
      for (int p=0; p < P; p++) begin
        status = $fscanf(file, "%d\r", in_mag[p]);
        status = $fscanf(file, "%d\r", in_pol[p]);
      end
    end

    @(posedge clk); #1
    {in_vld, in_mag, in_pol} = '0;

    wait (out_vld);
    wait (!out_vld);
    $finish();
  end

endmodule