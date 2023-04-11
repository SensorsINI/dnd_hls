module addr_calc #(parameter CAVIAR_X_Y_BITS = 9 )(
input clk,
input [1:0]count_modulo,
input [CAVIAR_X_Y_BITS-1:0]input_addr_x,
input [CAVIAR_X_Y_BITS-1:0]input_addr_y,
output [CAVIAR_X_Y_BITS-1:0]addr1_x_out,
output [CAVIAR_X_Y_BITS-1:0]addr1_y_out,
output [CAVIAR_X_Y_BITS-1:0]addr2_x_out,
output [CAVIAR_X_Y_BITS-1:0]addr2_y_out
);


logic [CAVIAR_X_Y_BITS-1:0]addr1_x;
logic [CAVIAR_X_Y_BITS-1:0]addr1_y;
logic [CAVIAR_X_Y_BITS-1:0]addr2_x;
logic [CAVIAR_X_Y_BITS-1:0]addr2_y;

assign addr1_x_out = addr1_x;
assign addr1_y_out = addr1_y;
assign addr2_x_out = addr2_x;
assign addr2_y_out = addr2_y;

always_ff @(posedge clk) begin
    unique case (count_modulo)
        2'd0: begin
        addr1_x <= input_addr_x - 3;
        addr2_x <= input_addr_x - 2;
        addr1_y <= input_addr_y - 3;
        addr2_y <= input_addr_y - 3;
        end
        2'd1: begin
        addr1_x <= input_addr_x + 2;
        addr2_x <= input_addr_x - 4;
        addr1_y <= input_addr_y;
        addr2_y <= input_addr_y + 1;
        end
        2'd2: begin
        addr1_x <= input_addr_x - 5;
        addr2_x <= input_addr_x - 4;
        addr1_y <= input_addr_y + 1;
        addr2_y <= input_addr_y + 1;
        end
        2'd3: begin
        addr1_x <= input_addr_x + 2;
        addr2_x <= input_addr_x + 3;
        addr1_y <= input_addr_y;
        addr2_y <= input_addr_y;
        end
    endcase    
end

endmodule