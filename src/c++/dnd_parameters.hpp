/*
 * Denoising filter for DVS devices using MLP parameters.
 */

#include <ap_int.h>
#include <math.h>
#include <algorithm>

// DVS parameters
#define DVS_WIDTH 346
#define DVS_HEIGHT 260
#define CAVIAR_X_Y_BITS 9 //(int)std::max(ceil(log2(DVS_WIDTH)), ceil(log2(DVS_HEIGHT)))

typedef ap_uint<CAVIAR_X_Y_BITS * 2 + 1> caviar_data_t;


// Temporal image parameters
#define NNb_TI_PATCH_WIDTH 7
#define TIMESTAMP_BITS 16
#define POLARITY_BITS 2
#define TIMESTAMP_IMAGE_DATA_BITS TIMESTAMP_BITS+POLARITY_BITS
#define TAU_BITS 6
#define TAU 64		// 2^TAU_BITS --> 64ms
#define CURRENT_EVENT_POSITION NNb_TI_PATCH_WIDTH*NNb_TI_PATCH_WIDTH / 2
#define NNb_TI_PATCH_SIZE NNb_TI_PATCH_WIDTH*NNb_TI_PATCH_WIDTH

typedef ap_uint<TIMESTAMP_IMAGE_DATA_BITS> timestamp_image_data_t;
typedef ap_uint<TIMESTAMP_BITS> timestamp_t;
typedef ap_int<POLARITY_BITS> polarity_t;


// MLP parameters
#define MLP_INPUT_ACTIVATION_BITS 8
#define MLP_INPUT_ACTIVATION_INT_BITS 2
#define MLP_INPUT_NEURONS NNb_TI_PATCH_WIDTH*NNb_TI_PATCH_WIDTH*2  // NNb_TI_patch size and the polarity of each element
#define MLP_OUTPUT_PREDICTION_BITS 8
#define MLP_OUTPUT_PREDICTION_INT_BITS 4

typedef ap_ufixed<12, TAU_BITS> timestamp_diff_t;
typedef ap_fixed<MLP_INPUT_ACTIVATION_BITS, MLP_INPUT_ACTIVATION_INT_BITS> mlp_input_activation_t;
typedef ap_ufixed<MLP_OUTPUT_PREDICTION_BITS, MLP_OUTPUT_PREDICTION_INT_BITS> mlp_output_prediction_data_t;
//typedef ap_int<MLP_INPUT_NEURONS*MLP_INPUT_ACTIVATION_BITS> mlp_input_data_t;
