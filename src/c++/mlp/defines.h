#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 98
#define N_LAYER_2 20
#define N_LAYER_5 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<4,1> weight2_t;
typedef ap_fixed<4,1> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<4,0> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<4,1> weight5_t;
typedef ap_fixed<4,1> bias5_t;
typedef ap_uint<1> layer5_index;
typedef ap_ufixed<16,0> result_t;
typedef ap_fixed<18,8> output_act_table_t;

#endif
