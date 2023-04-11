module create_mlp_activations #(parameter integer DVS_WIDTH = 346, DVS_HEIGHT = 260, WORD_SIZE = 18,CAVIAR_X_Y_BITS = 9,TIMESTAMP_BITS = 16,POLARITY_BITS = 2)( 
input clk,
input rst_n, 
input read_data_mem_vld1,
input read_data_mem_vld2,
input [WORD_SIZE-1:0]read_data1_mem,
input [WORD_SIZE-1:0]read_data2_mem,
output [WORD_SIZE-1:0]write_data_mem,
output rw, 
output cen,
output logic [CAVIAR_X_Y_BITS-1:0]addr_port1_x,
output logic [CAVIAR_X_Y_BITS-1:0]addr_port1_y,
output logic [CAVIAR_X_Y_BITS-1:0]addr_port2_x,
output logic [CAVIAR_X_Y_BITS-1:0]addr_port2_y,

//output [DVS_WIDTH-1:0][DVS_HEIGHT-1:0]address,
input [2*CAVIAR_X_Y_BITS:0]cavier_in,                           
input cavier_in_vld,
input [TIMESTAMP_BITS-1:0]current_timestamp,
input current_timestamp_vld,
output [TIMESTAMP_BITS-1:0]MLPout1,
output [POLARITY_BITS -1:0]MLPout2,
output [TIMESTAMP_BITS-1:0]MLPout3,
output [POLARITY_BITS-1:0]MLPout4,
output MLPvld,
output done
);


enum logic [1:0] {IDLE, LOAD_COMPUTE, STORE} current_state,next_state;

logic [2*CAVIAR_X_Y_BITS:0]cavier_in_reg;
logic [TIMESTAMP_BITS-1:0]current_timestamp_reg;
logic [4:0]count_upd;
logic [4:0]count_reg;
logic [CAVIAR_X_Y_BITS-1:0]addr1_x_reg;
logic [CAVIAR_X_Y_BITS-1:0]addr2_x_reg;
logic [CAVIAR_X_Y_BITS-1:0]addr1_y_reg;
logic [CAVIAR_X_Y_BITS-1:0]addr2_y_reg;
logic read_data_mem_vld1_reg;
logic read_data_mem_vld2_reg;
logic [WORD_SIZE-1:0]read_data1_mem_reg;
logic [WORD_SIZE-1:0]read_data2_mem_reg;
logic MLPvld_reg;

logic [CAVIAR_X_Y_BITS-1:0]input_addr_addr_calc_x;
logic [CAVIAR_X_Y_BITS-1:0]input_addr_addr_calc_y;

logic [CAVIAR_X_Y_BITS-1:0]addr1_x;
logic [CAVIAR_X_Y_BITS-1:0]addr2_x;
logic [CAVIAR_X_Y_BITS-1:0]addr1_y;
logic [CAVIAR_X_Y_BITS-1:0]addr2_y;
logic cen_reg;

logic [2-1:0] count_modulo;

assign cen = cen_reg;
addr_calc #(.CAVIAR_X_Y_BITS(CAVIAR_X_Y_BITS)) addr_calc_inst( .clk, .count_modulo(count_modulo), .input_addr_x(input_addr_addr_calc_x), .input_addr_y(input_addr_addr_calc_y), .addr1_x_out(addr1_x), .addr2_x_out(addr2_x), .addr1_y_out(addr1_y),.addr2_y_out(addr2_y) );
age_calc #(.TIMESTAMP_BITS(TIMESTAMP_BITS),.POLARITY_BITS(POLARITY_BITS), .WORD_SIZE(WORD_SIZE)) age_calc_inst( .read_data1(read_data1_mem_reg), .read_data2(read_data2_mem_reg), .out1(MLPout1), .out2(MLPout2), .out3(MLPout3), .out4(MLPout4));

//Counter update logic 
//assign count_upd = (read_data_mem_vld1 & read_data_mem_vld2) ? count_reg+1 : count_reg;

//Input address to addr_calc module 
assign input_addr_addr_calc_x = ( count_reg ==0 ) ? cavier_in_reg[2*CAVIAR_X_Y_BITS:CAVIAR_X_Y_BITS+1]:addr1_x_reg;
assign input_addr_addr_calc_y = ( count_reg ==0 ) ? cavier_in_reg[CAVIAR_X_Y_BITS:1]: addr1_y_reg;

//assign MLPvld = read_data_mem_vld1_reg & read_data_mem_vld2_reg;

//Assigning ADDRESS ports of memory  
always_ff @(posedge clk) begin
addr_port1_x <= (current_state == STORE)?cavier_in_reg[2*CAVIAR_X_Y_BITS:CAVIAR_X_Y_BITS+1]: addr1_x;
addr_port2_x <=  addr2_x;
addr_port1_y <= (current_state == STORE)?cavier_in_reg[CAVIAR_X_Y_BITS:1]: addr1_y;
addr_port2_y <= addr2_y;
end

//Assigning read/Write and chip enable
assign rw = (next_state == STORE)? 1:0;
//assign cen = (current_state == IDLE)?0 : 1;

assign write_data_mem = (current_state == STORE)?current_timestamp_reg: 0;
assign done = (current_state == STORE)?1: 0;

//Assigning MLPout valid 
assign MLPvld = MLPvld_reg;

always_ff@(posedge clk, negedge rst_n)begin
    if ( !rst_n )begin
        current_state <= IDLE;
        count_reg <=0;
        read_data_mem_vld1_reg <=0;
        read_data_mem_vld2_reg <=0;
        addr1_x_reg <=0;
        addr1_y_reg <=0;
        addr2_x_reg <=0;
        addr2_y_reg <=0;
        current_timestamp_reg <=0;
        cavier_in_reg <= 0;
        read_data1_mem_reg <=0;
        read_data2_mem_reg <=0;
        count_modulo <= 0;
    end    
    else begin
        current_state <= next_state;
        count_reg <= count_upd;
        count_modulo <= count_upd     == 0                      ? 0 : 
                        count_upd % 7 == 3                      ? 1 :
                        count_upd % 7 == 4 || count_upd %7 == 0 ? 2 :
                                                                  3;

        if( current_state == LOAD_COMPUTE)begin
            read_data_mem_vld1_reg <= read_data_mem_vld1;
            read_data_mem_vld2_reg <= read_data_mem_vld2;
        end    
        addr1_x_reg <= addr1_x;
        addr1_y_reg <= addr1_y;
        addr2_x_reg <= addr2_x;
        addr2_y_reg <= addr2_y;
        
        if ( cavier_in_vld & current_timestamp_vld) begin
            current_timestamp_reg <= current_timestamp;
            cavier_in_reg <= cavier_in;
        end 

        if ( read_data_mem_vld1)   
            read_data1_mem_reg <= read_data1_mem;
        if ( read_data_mem_vld2)   
            read_data2_mem_reg <= read_data2_mem;
        if (current_state != IDLE )
            cen_reg <=1;
        else    
            cen_reg <=0;     
        if ( current_state != LOAD_COMPUTE )
            MLPvld_reg <=0;
        else
            MLPvld_reg <= read_data_mem_vld1 & read_data_mem_vld2;

    end    
end

always_comb begin
    case(current_state)
    IDLE:begin
        count_upd = 0;
        if ( cavier_in_vld & current_timestamp_vld) begin
            next_state = LOAD_COMPUTE;
        end    
        else begin
            next_state = IDLE;   
        end    
    end    
    LOAD_COMPUTE:begin
        if (read_data_mem_vld1 & read_data_mem_vld2)
            count_upd = count_reg+1 ;
        else 
            count_upd = count_reg;
            
        if( count_reg > 24) begin
            next_state = STORE; 
        end    
        else begin
            next_state = LOAD_COMPUTE; 
        end    
    end    
    STORE:begin
        count_upd = 0;
        next_state = IDLE;
    end   

    endcase
end






endmodule 