set clock_period 1
set io_delay 0.2
create_clock -name clk -period $clock_period [get_ports clk]
set_input_delay -clock [get_clocks clk] -add_delay -max $io_delay [all_inputs]
set_output_delay -clock [get_clocks clk] -add_delay -max $io_delay [all_outputs]

remove_input_delay {read_data_mem_vld1}
remove_input_delay {read_data_mem_vld2}
remove_input_delay {read_data1_mem}
remove_input_delay {read_data2_mem}
remove_output_delay {write_data_mem}
remove_output_delay {rw}
remove_output_delay {cen}
remove_output_delay {addr_port1}
remove_output_delay {addr_port2}
