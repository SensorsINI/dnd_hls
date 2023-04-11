module age_calc #(parameter TIMESTAMP_BITS = 16,POLARITY_BITS = 2,WORD_SIZE = 18 )( 
    input [WORD_SIZE-1:0]read_data1, 
    input [WORD_SIZE-1:0]read_data2, 
    output [TIMESTAMP_BITS-1:0]out1, 
    output [POLARITY_BITS -1:0]out2, 
    output [TIMESTAMP_BITS-1:0]out3, 
    output [POLARITY_BITS -1:0]out4);

logic [TIMESTAMP_BITS-1:0]out1_reg;
logic [TIMESTAMP_BITS-1:0]out3_reg;
assign out2 = read_data1[POLARITY_BITS -1:0];
assign out4 = read_data2[POLARITY_BITS -1:0];
assign out1 = out1_reg;
assign out3 = out3_reg;

always_comb begin
if (read_data1[(WORD_SIZE-1): (WORD_SIZE-11)] > 6'b111111 )begin
    out1_reg = {6'b111111,read_data1[(WORD_SIZE - TIMESTAMP_BITS + 5):(WORD_SIZE-TIMESTAMP_BITS)],4'b0000};
    out3_reg = {6'b111111,read_data2[(WORD_SIZE - TIMESTAMP_BITS + 5):(WORD_SIZE-TIMESTAMP_BITS)],4'b0000};
end    
else begin
    out1_reg = {read_data1[(WORD_SIZE-TIMESTAMP_BITS + 11):(WORD_SIZE-TIMESTAMP_BITS)],4'b0000};
    out3_reg = {read_data2[(WORD_SIZE-TIMESTAMP_BITS + 11):(WORD_SIZE-TIMESTAMP_BITS)],4'b0000};
end
end    
endmodule


