module top #(

  parameter   DVS_WIDTH       = 346, 
              DVS_HEIGHT      = 260, 
              WORD_SIZE       = 18,
              CAVIAR_X_Y_BITS = 9,
              TIMESTAMP_BITS  = 16,
              POLARITY_BITS   = 2,
              N1 = 98,
              N2 = 10,
              P  = 2,
              W_X = 4,
              W_K = 4,
              W_Y = 16,
  localparam W_ADDR          = $clog2(DVS_WIDTH*DVS_HEIGHT)
)(
  input  logic clk,
  input  logic rst_n, 
  // cavier
  input logic [2*CAVIAR_X_Y_BITS:0]cavier_in,                           
  input logic cavier_in_vld,
  input logic [TIMESTAMP_BITS-1:0]current_timestamp,
  input logic current_timestamp_vld,
  // memory
  input  logic read_data_mem_vld1,
  input  logic read_data_mem_vld2,
  input  logic [WORD_SIZE-1:0]read_data1_mem,
  input  logic [WORD_SIZE-1:0]read_data2_mem,
  output logic [WORD_SIZE-1:0]write_data_mem,
  output logic rw, 
  output logic cen,
  output logic [W_ADDR-1:0]addr_port1,
  output logic [W_ADDR-1:0]addr_port2,

  // MLP
  output logic [W_Y -1:0] out,
  output logic            out_vld
);

  logic [CAVIAR_X_Y_BITS-1:0]addr_port1_x;
  logic [CAVIAR_X_Y_BITS-1:0]addr_port1_y;
  logic [CAVIAR_X_Y_BITS-1:0]addr_port2_x;
  logic [CAVIAR_X_Y_BITS-1:0]addr_port2_y;
  assign addr_port1 = addr_port1_x + DVS_WIDTH*addr_port1_y;
  assign addr_port2 = addr_port2_x + DVS_WIDTH*addr_port2_y;


  logic [TIMESTAMP_BITS-1:0]MLPout1;
  logic [POLARITY_BITS -1:0]MLPout2;
  logic [TIMESTAMP_BITS-1:0]MLPout3;
  logic [POLARITY_BITS -1:0]MLPout4;
  logic MLPvld;
  logic done;

  create_mlp_activations #(
    .DVS_WIDTH  (DVS_WIDTH ), 
    .DVS_HEIGHT (DVS_HEIGHT), 
    .WORD_SIZE  (WORD_SIZE ),
    .CAVIAR_X_Y_BITS (CAVIAR_X_Y_BITS),
    .TIMESTAMP_BITS  (TIMESTAMP_BITS ),
    .POLARITY_BITS   (POLARITY_BITS  ))
    CREATE_ACT (.*);

  wire in_vld = MLPvld;
  wire [P-1:0][W_X-1:0] in_mag = {W_X'(MLPout3/8), W_X'(MLPout1/8)};
  wire [P-1:0][1:0]     in_pol = {MLPout4, MLPout2};

  mlp_serial #(
   .N1 (N1 ),
   .N2 (N2 ),
   .P  (P  ),
   .W_X(W_X),
   .W_K(W_K),
   .W_Y(W_Y)
  ) MLP (.rst(!rst_n), .*);

endmodule