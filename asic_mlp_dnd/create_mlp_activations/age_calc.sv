// This file is part of https://github.com/SensorsINI/dnd_hls. 
// This intellectual property is licensed under the terms of the project license available at the root of the project.
module age_calc #(parameter TIMESTAMP_BITS = 16,POLARITY_BITS = 2,WORD_SIZE = 18 )( 
    input [WORD_SIZE-1:0]read_data1, 
    input [WORD_SIZE-1:0]read_data2, 
    input [TIMESTAMP_BITS-1:0]ts_tau_diff,
    input [TIMESTAMP_BITS-1:0]current_timestamp,
    output [TIMESTAMP_BITS-1:0]out1, 
    output [POLARITY_BITS -1:0]out2, 
    output [TIMESTAMP_BITS-1:0]out3, 
    output [POLARITY_BITS -1:0]out4);

logic [TIMESTAMP_BITS-1:0]out1_reg;
logic [TIMESTAMP_BITS-1:0]out3_reg;
logic [TIMESTAMP_BITS-1:0]age1; 
logic [TIMESTAMP_BITS-1:0]age2; 
logic [TIMESTAMP_BITS-1:0]patch_ts1;
logic [TIMESTAMP_BITS-1:0]patch_ts2;
logic [TIMESTAMP_BITS-1:0]ts_diff1;
logic [TIMESTAMP_BITS-1:0]ts_diff2; 
assign out2 = read_data1[POLARITY_BITS -1:0];
assign out4 = read_data2[POLARITY_BITS -1:0];
//assign out1 = out1_reg;
//assign out3 = out3_reg;
assign patch_ts1 = read_data1[WORD_SIZE-1:POLARITY_BITS];
assign patch_ts2 = read_data2[WORD_SIZE-1:POLARITY_BITS];

assign out1 = age1;
assign out3 = age2;

always_comb begin
    if(patch_ts1 < ts_tau_diff) begin																			
        ts_diff1 = 0;
		  age1 = 0;
    end
    else begin
            ts_diff1 = current_timestamp - patch_ts1;
            if ( ts_diff1 > 6'b111111)   
                ts_diff1 =  {6'b000000,6'b111111,4'b0000};
            else    
                ts_diff1 =  {6'b000000,ts_diff1[5:0],4'b0000};
            age1 = 1 - (ts_diff1);
    end
end    

always_comb begin
    if(patch_ts2 < ts_tau_diff) begin																			
        ts_diff2 =0;
		  age2 = 0;
    end
    else begin
            ts_diff2 = current_timestamp - patch_ts2;
            if ( ts_diff2 > 6'b111111)   
                ts_diff2 =  {6'b000000,6'b111111,4'b0000};
            else    
                ts_diff2 =  {6'b000000,ts_diff2[5:0],4'b0000}; 
            age2 = 1 - (ts_diff2);
    end
end   


/*
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
*/

endmodule


