#include "dnd_top.hpp"

#define SIGNAL_NOISE_THRESHOLD 0.5
#define VERBOSE

caviar_data_t dnd_top(
		caviar_data_t spike_in,
		timestamp_polarity_image_data_t timestamp_polarity_image[DVS_WIDTH * DVS_HEIGHT],
		timestamp_t current_timestamp
		){
#pragma HLS INTERFACE ap_vld port=spike_in,return
#pragma HLS PIPELINE

	// Define the mlp_activations signal
	mlp_input_activation_t mlp_activations[MLP_INPUT_NEURONS];
#pragma HLS ARRAY_MAP variable=mlp_activations horizontal

	// Create the activations for the current incoming spike
	dnd_create_mlp_activation(spike_in, timestamp_polarity_image, current_timestamp, mlp_activations);

	// Run the MLP using the previous activations
	unsigned short size_in;
	unsigned short size_out;
	result_t mlp_out[N_LAYER_5];
	myproject(mlp_activations, mlp_out, size_in, size_out);

	// Decide if the incoming event is signal or noise
#ifdef VERBOSE
	printf("MLP Prediction: %f >= TH: %f", mlp_out[0].to_float(), SIGNAL_NOISE_THRESHOLD);
#endif
	if(mlp_out[0] >= SIGNAL_NOISE_THRESHOLD){
		return spike_in;
	}else{
		return 0;
	}

}
