module mlp_tb;
  timeunit 1ns/1ps;

  localparam  CLK_PERIOD = 10,
              N1 = 98,
              N2 = 10,
              W_X = 4,
              W_K = 4,
              W_Y = 16;

  localparam  W_SUM_FC1     = W_X + W_K + $clog2(N1/2),
              W_SUM_FC1_POL = 1 + W_K + $clog2(N1/2),
              W_SUM_FC2     = W_X + W_K + $clog2(N2);

  logic clk=0;
  logic [N1/2-1:0][W_X-1:0] in_mag;
  logic [N1/2-1:0]          in_pol;
  logic [W_Y -1:0]          out;
  int status;

  initial forever #(CLK_PERIOD/2) clk <= ~clk;
  mlp #(.N1(N1),.N2(N2),.W_X(W_X),.W_K(W_K),.W_Y(W_Y)) dut (.*);

  initial begin
    status = std::randomize(in_mag);
    status = std::randomize(in_pol);

    repeat (100) @(posedge clk);
  end

endmodule