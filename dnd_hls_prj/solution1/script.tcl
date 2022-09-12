############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project dnd_hls_prj
set_top writePixScale0
add_files dnd_hls_prj/src/dndAccel.cpp
add_files dnd_hls_prj/src/dndAccel.h
add_files -tb dnd_hls_prj/src/test.cpp
open_solution "solution1"
set_part {xc7z035-ffg676-2}
create_clock -period 10 -name default
set_clock_uncertainty 0.1
#source "./dnd_hls_prj/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
