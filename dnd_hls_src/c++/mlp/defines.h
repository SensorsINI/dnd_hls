// This file is part of https://github.com/SensorsINI/dnd_hls. 
// This intellectual property is licensed under the terms of the project license available at the root of the project.
#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 98
#define N_LAYER_3 10
#define N_LAYER_6 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<4,1> layer2_t;
typedef ap_fixed<18,8> qact0_table_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<4,1> weight3_t;
typedef ap_fixed<4,1> bias3_t;
typedef ap_uint<1> layer3_index;
typedef ap_ufixed<4,0> layer5_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<4,1> weight6_t;
typedef ap_fixed<4,1> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<16,6> output_default_t;
typedef ap_fixed<18,8> outputtable_t;
typedef ap_fixed<16,6> result_t;

#endif
