module create_mlp_activations_tb();

parameter integer DVS_WIDTH = 346; 
parameter integer DVS_HEIGHT = 260; 
parameter integer WORD_SIZE = 18;
parameter integer CAVIAR_X_Y_BITS = 9; 
parameter integer TIMESTAMP_BITS = 16; 
parameter integer POLARITY_BITS = 2; 

//logic [WORD_SIZE-1:0]input_file_buffer[0:DVS_WIDTH-1][0:DVS_HEIGHT-1];
logic [WORD_SIZE-1:0]Memory[0:7][0:7];
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
logic [CAVIAR_X_Y_BITS-1:0]addr_port1_x;
logic [CAVIAR_X_Y_BITS-1:0]addr_port1_y;
logic [CAVIAR_X_Y_BITS-1:0]addr_port2_x;
logic [CAVIAR_X_Y_BITS-1:0]addr_port2_y;

logic [TIMESTAMP_BITS-1:0]MLPout1;
logic [POLARITY_BITS-1:0]MLPout2;
logic [TIMESTAMP_BITS-1:0]MLPout3;
logic [POLARITY_BITS-1:0]MLPout4;
logic done;

integer qk_file;
integer qk_scan_file;
int captured_data;

create_mlp_activations #(.DVS_WIDTH(DVS_WIDTH), .DVS_HEIGHT(DVS_HEIGHT), .WORD_SIZE(WORD_SIZE), .CAVIAR_X_Y_BITS(CAVIAR_X_Y_BITS), .TIMESTAMP_BITS(TIMESTAMP_BITS), .POLARITY_BITS(POLARITY_BITS)) create_mlp_activations_inst 
 (clk, rst_n, read_data_mem_vld1, read_data_mem_vld2, read_data1_mem, read_data2_mem, write_data_mem, rw, cen, addr_port1_x, addr_port1_y, addr_port2_x, addr_port2_y, cavier_in, cavier_in_vld, current_timestamp, current_timestamp_vld, MLPout1, MLPout2, MLPout3, MLPout4, MLPvld, done);

// clock generator
always begin
    #1 clk = 1'b1;
    #1 clk = 1'b0;
end

//initial $readmemh("/home/keerthivasan/Downloads/hls4ml/create_mlp_activations/input_data.txt", Memory);

initial begin 

//Populating Memory 
for( int i = 0; i < 8; i++)begin
    for( int j = 0; j < 8; j++)begin
		  Memory[i][j] = i + j;
        //$display("%d\n",Memory[i][j]);
    end
end   

@(posedge clk) rst_n = 0;
for (int m = 0; m < 2; m++) @(posedge clk);
rst_n = 1;
for (int m = 0; m < 2; m++) @(posedge clk);


// START PROCESSOR
cavier_in_vld = 1'b1;
current_timestamp_vld = 1'b1;
cavier_in = 19'h01009; //choosing at random 
current_timestamp = 100; //choosing at random 

$display("POLARITY :%d, at X:%d, Y:%d\n",cavier_in[0], cavier_in[18:10], cavier_in[9:1]);

for (int m = 0; m < 2; m++) @(posedge clk);
cavier_in_vld = 1'b0;
current_timestamp_vld = 1'b0;

//#1000 $finish; 
wait (done == 1);

end    


always @(posedge clk, negedge rst_n) begin
/*
if ( !rst_n )begin
    for( int i = 0; i < 8; i++)begin
        for( int j = 0; j < 8; i++)begin
            //qk_scan_file = $fscanf(qk_file, "%d\n", captured_data);
            Memory[i][j] = i;
            //printf("POLARITY :%d, at X:%d, Y:%d\n",captured_data[0], captured_data[18:10], captured_data[9:1] );
        end
    end    
end  
*/ 
if (cen && !rw) begin //read
        read_data1_mem <=  Memory[addr_port1_x][addr_port1_y];
        read_data_mem_vld1 <= 1;
        read_data2_mem <=  Memory[addr_port2_x][addr_port2_y];
        read_data_mem_vld2 <= 1;
    end    
else if ( cen && rw)begin //write
        Memory[addr_port1_x][addr_port1_y] <= write_data_mem;
        read_data_mem_vld1 <= 0;
        read_data_mem_vld2 <= 0;
    end  
else begin
        read_data_mem_vld1 <= 0;
        read_data_mem_vld2 <= 0;
    end      
end


endmodule






