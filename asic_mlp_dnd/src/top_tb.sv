module top_tb;
  timeunit 1ns/1ps;
  localparam  DVS_WIDTH = 346,
              DVS_HEIGHT = 260,
              WORD_SIZE = 18,
              CAVIAR_X_Y_BITS = 9,
              TIMESTAMP_BITS = 16,
              POLARITY_BITS = 2,
              N1 = 98,
              N2 = 10,
              P  = 2,
              W_X = 4,
              W_K = 4,
              W_Y = 16,
              W_ADDR = $clog2(DVS_WIDTH*DVS_HEIGHT);

  //logic [WORD_SIZE-1:0]input_file_buffer[0:DVS_WIDTH-1][0:DVS_HEIGHT-1];
  logic [2*CAVIAR_X_Y_BITS:0]cavier_in;
  logic cavier_in_vld;
  logic [TIMESTAMP_BITS-1:0]current_timestamp;
  logic current_timestamp_vld;
  logic clk;
  logic rst_n;
  logic [WORD_SIZE-1:0]read_data1_mem;
  logic [WORD_SIZE-1:0]read_data2_mem;
  logic [WORD_SIZE-1:0]write_data_mem;
  logic read_data_mem_vld1;
  logic read_data_mem_vld2;
  logic rw; 
  logic cen;
  logic [W_ADDR-1:0]addr_port1;
  logic [W_ADDR-1:0]addr_port2;

  logic [W_Y -1:0] out;
  logic            out_vld;

  integer qk_file;
  integer qk_scan_file;
  int captured_data;

  logic [WORD_SIZE-1:0] memory [2**W_ADDR-1:0];

  top #(
    .DVS_WIDTH (DVS_WIDTH), 
    .DVS_HEIGHT(DVS_HEIGHT), 
    .WORD_SIZE (WORD_SIZE), 
    .CAVIAR_X_Y_BITS(CAVIAR_X_Y_BITS), 
    .TIMESTAMP_BITS (TIMESTAMP_BITS), 
    .POLARITY_BITS  (POLARITY_BITS),
    .N1  (N1 ),
    .N2  (N2 ),
    .P   (P  ),
    .W_X (W_X),
    .W_K (W_K),
    .W_Y (W_Y)
    ) DUT (.*);

  // clock generator
  always begin
      #1 clk = 1'b1;
      #1 clk = 1'b0;
  end

  //initial $readmemh("/home/keerthivasan/Downloads/hls4ml/create_mlp_activations/input_data.txt", Memory);

  initial begin 

    //Populating Memory 
    for(int i=0; i<2**W_ADDR; i++) begin
        memory[i] = i;
    end   

    @(posedge clk) #10ps rst_n = 0;
    for (int m = 0; m < 2; m++) @(posedge clk);
    #10ps
    rst_n = 1;
    for (int m = 0; m < 2; m++) @(posedge clk);
    #10ps


    // START PROCESSOR
    cavier_in_vld = 1'b1;
    current_timestamp_vld = 1'b1;
    cavier_in = 19'h01009; //choosing at random 
    current_timestamp = 100; //choosing at random 

    $display("POLARITY :%d, at X:%d, Y:%d\n",cavier_in[0], cavier_in[18:10], cavier_in[9:1]);

    @(posedge clk) #10ps;
    cavier_in_vld = 1'b0;
    current_timestamp_vld = 1'b0;

    wait (out_vld);

    @(posedge clk);
    $finish;

  end    

  always @(posedge clk, negedge rst_n) begin
    if (cen && !rw) begin //read
      read_data1_mem <= memory[addr_port1];
      read_data2_mem <= memory[addr_port1];
      read_data_mem_vld1 <= 1;
      read_data_mem_vld2 <= 1;
    end else if (cen && rw) begin //write
      memory[addr_port1] <= write_data_mem;
      read_data_mem_vld1 <= 0;
      read_data_mem_vld2 <= 0;
    end else begin
      read_data_mem_vld1 <= 0;
      read_data_mem_vld2 <= 0;
    end      
  end


endmodule






