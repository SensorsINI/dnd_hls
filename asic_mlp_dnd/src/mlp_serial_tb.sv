module mlp_serial_tb;
  timeunit 1ns/1ps;

  localparam  CLK_PERIOD = 10,
              N1 = 98,
              N2 = 10,
              P  = 2,
              W_X = 4,
              W_K = 4,
              W_Y = 16;

  localparam  N_BEATS       = (1+N1/2)/2,
              W_SUM_FC1     = W_X + W_K + $clog2(N1/2),
              W_SUM_FC1_POL = 1   + W_K + $clog2(N1/2),
              W_SUM_FC2     = W_X + W_K + $clog2(N2);

  logic clk=0, rst=1, in_vld=0, out_vld;
  logic [P   -1:0][W_X-1:0] in_mag;
  logic [P   -1:0][1:0]     in_pol;
  logic [W_Y -1:0]          out;
  int status;

  logic [W_X-1:0] queue_mag [$];
  logic [    1:0] queue_pol [$];
  logic [W_X-1:0] out_mag;
  logic [    1:0] out_pol;

  initial forever #(CLK_PERIOD/2) clk <= ~clk;
  mlp_serial #(.N1(N1),.N2(N2),.P(P),.W_X(W_X),.W_K(W_K),.W_Y(W_Y)) dut (.*);

  wire [N2-1:0][N1/2:0][W_K-1:0] weights_n1_mag;
  wire [N2-1:0][N1/2:0][W_K-1:0] weights_n1_pol;
  wire [N2  :0][W_K-1:0] weights_n2;
  wire [2**W_K-1:0][W_Y-1:0] tanh;
  luts LUTS (.*);

  logic [N2-1:0][W_SUM_FC1 -1:0] fc1_out_exp = '0;

  initial begin

    repeat (2) @(posedge clk);
    rst <= 0;
    repeat (2) @(posedge clk);

    repeat (N_BEATS) begin
      @(posedge clk); #1

      in_vld = 1;
      // in_mag = {W_K'(1), W_K'(1)};
      in_pol = 0;
      status = std::randomize(in_mag);
      // status = std::randomize(in_pol);
      for (int p=0; p < P; p++) begin
        queue_mag.push_front(in_mag[p]);
        queue_pol.push_front(in_pol[p]);
      end
    end

    @(posedge clk); #1
    {in_vld, in_mag, in_pol} = '0;

    wait (dut.add_last);

    for (int n1=0; n1 < N1/2+1; n1++) begin
      out_mag = queue_mag.pop_back();
      out_pol = queue_pol.pop_back();
      for (int n2=0; n2 < N2; n2++)
        fc1_out_exp[n2] = $signed(fc1_out_exp[n2]) + $signed(weights_n1_mag[n2][n1]) * $signed(out_mag) + $signed(weights_n1_pol[n2][n1]) * $signed(out_pol);
    end
  end

endmodule