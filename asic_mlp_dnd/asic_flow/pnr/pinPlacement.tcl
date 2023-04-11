# getPinAssignMode -pinEditInBatch -quiet
# setPinAssignMode -pinEditInBatch true
# editPin -pinWidth 0.09 -pinDepth 0.47 -fixOverlap 1 -unit MICRON -spreadDirection clockwise -side Left -layer 1 -spreadType center -spacing 1.5 -pin {clk {x[0]} {x[1]} {x[2]} {x[3]} {y[0]} {y[1]} {y[2]} {y[3]} {z[0]} {z[1]} {z[2]} {z[3]}}
# setPinAssignMode -pinEditInBatch false
# getPinAssignMode -pinEditInBatch -quiet
# setPinAssignMode -pinEditInBatch true
# editPin -fixOverlap 1 -unit MICRON -spreadDirection clockwise -side Right -layer 1 -spreadType center -spacing 1.6 -pin {{out[0]} {out[1]} {out[2]} {out[3]} {out[4]} {out[5]}}
# setPinAssignMode -pinEditInBatch false


getPinAssignMode -pinEditInBatch -quiet
setPinAssignMode -pinEditInBatch true


editPin -fixOverlap 1 -unit MICRON -spreadDirection clockwise -side Left -layer 3 -spreadType center -spacing 0.8 -pin {{addr_port1[0]} {addr_port1[1]} {addr_port1[2]} {addr_port1[3]} {addr_port1[4]} {addr_port1[5]} {addr_port1[6]} {addr_port1[7]} {addr_port1[8]} {addr_port1[9]} {addr_port1[10]} {addr_port1[11]} {addr_port1[12]} {addr_port1[13]} {addr_port1[14]} {addr_port1[15]} {addr_port1[16]} {addr_port2[0]} {addr_port2[1]} {addr_port2[2]} {addr_port2[3]} {addr_port2[4]} {addr_port2[5]} {addr_port2[6]} {addr_port2[7]} {addr_port2[8]} {addr_port2[9]} {addr_port2[10]} {addr_port2[11]} {addr_port2[12]} {addr_port2[13]} {addr_port2[14]} {addr_port2[15]} {addr_port2[16]} {cavier_in[0]} {cavier_in[1]} {cavier_in[2]} {cavier_in[3]} {cavier_in[4]} {cavier_in[5]} {cavier_in[6]} {cavier_in[7]} {cavier_in[8]} {cavier_in[9]} {cavier_in[10]} {cavier_in[11]} {cavier_in[12]} {cavier_in[13]} {cavier_in[14]} {cavier_in[15]} {cavier_in[16]} {cavier_in[17]} {cavier_in[18]} cavier_in_vld cen clk {current_timestamp[0]} {current_timestamp[1]} {current_timestamp[2]} {current_timestamp[3]} {current_timestamp[4]} {current_timestamp[5]} {current_timestamp[6]} {current_timestamp[7]} {current_timestamp[8]} {current_timestamp[9]} {current_timestamp[10]} {current_timestamp[11]} {current_timestamp[12]} {current_timestamp[13]} {current_timestamp[14]} {current_timestamp[15]} current_timestamp_vld {read_data1_mem[0]} {read_data1_mem[1]} {read_data1_mem[2]} {read_data1_mem[3]} {read_data1_mem[4]} {read_data1_mem[5]} {read_data1_mem[6]} {read_data1_mem[7]} {read_data1_mem[8]} {read_data1_mem[9]} {read_data1_mem[10]} {read_data1_mem[11]} {read_data1_mem[12]} {read_data1_mem[13]} {read_data1_mem[14]} {read_data1_mem[15]} {read_data1_mem[16]} {read_data1_mem[17]} {read_data2_mem[0]} {read_data2_mem[1]} {read_data2_mem[2]} {read_data2_mem[3]} {read_data2_mem[4]} {read_data2_mem[5]} {read_data2_mem[6]} {read_data2_mem[7]} {read_data2_mem[8]} {read_data2_mem[9]} {read_data2_mem[10]} {read_data2_mem[11]} {read_data2_mem[12]} {read_data2_mem[13]} {read_data2_mem[14]} {read_data2_mem[15]} {read_data2_mem[16]} {read_data2_mem[17]} read_data_mem_vld1 read_data_mem_vld2 rst_n rw {write_data_mem[0]} {write_data_mem[1]} {write_data_mem[2]} {write_data_mem[3]} {write_data_mem[4]} {write_data_mem[5]} {write_data_mem[6]} {write_data_mem[7]} {write_data_mem[8]} {write_data_mem[9]} {write_data_mem[10]} {write_data_mem[11]} {write_data_mem[12]} {write_data_mem[13]} {write_data_mem[14]} {write_data_mem[15]} {write_data_mem[16]} {write_data_mem[17]}}


setPinAssignMode -pinEditInBatch false
getPinAssignMode -pinEditInBatch -quiet
setPinAssignMode -pinEditInBatch true

editPin -fixOverlap 1 -unit MICRON -spreadDirection clockwise -side Right -layer 3 -spreadType center -spacing 0.8 -pin {{out[0]} {out[1]} {out[2]} {out[3]} {out[4]} {out[5]} {out[6]} {out[7]} {out[8]} {out[9]} {out[10]} {out[11]} {out[12]} {out[13]} {out[14]} {out[15]} out_vld}
setPinAssignMode -pinEditInBatch false
