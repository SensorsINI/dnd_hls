#include "dnd_create_mlp_activation.hpp"
//#define VERBOSE

void dnd_create_mlp_activation(
		caviar_data_t caviar_spike,
		timestamp_polarity_image_data_t timestamp_polarity_image[DVS_WIDTH * DVS_HEIGHT], 	//TODO change the memory access using @Min memory layout
		timestamp_t current_timestamp,
		mlp_input_activation_t mlp_activation[MLP_INPUT_NEURONS]
		) {
#pragma HLS PIPELINE

#ifdef VERBOSE
	printf("Get the spike column, row and pol from the caviar input\n");
#endif

	// Gets X,Y coordinate and polarity of the current event.
	ap_uint<CAVIAR_X_Y_BITS> spike_y = caviar_spike.range(caviar_spike.width-1 , CAVIAR_X_Y_BITS+1);
	ap_uint<CAVIAR_X_Y_BITS> spike_x = caviar_spike.range(CAVIAR_X_Y_BITS, 1);	//Remove the polarity to get the spike address
	polarity_t spike_pol = (caviar_spike.range(0, 0) == 1) ? 1 : -1; 			//MLP polarity: 0-> No or old spike; 1-> ON; -1-> OFF

	// Set some useful variables
	timestamp_polarity_image_data_t timestamp_polarity_patch[NNb_TI_PATCH_WIDTH*NNb_TI_PATCH_WIDTH];
	uint TI_addr, PATCH_addr;
	int patch_border = NNb_TI_PATCH_WIDTH >> 1;
	int spike_surround_y = spike_y - patch_border;
	int spike_surround_x = spike_x - patch_border;
	mlp_input_activation_t age;
	mlp_input_activation_t pol;
	timestamp_diff_t ts_diff;
	int ts_tau_diff = current_timestamp - TAU;
	int act_idx = 0;
	timestamp_polarity_image_data_t ts_pol;
	timestamp_t patch_ts;
	polarity_t patch_pol;

#ifdef VERBOSE
	printf("Read the NNb_TI_patch (Timestamp:Polarity) from the Timestamp Image.\n");
#endif

	read_col_NNb_TI_patch: for(int patch_x=0; patch_x<NNb_TI_PATCH_WIDTH; patch_x++){
#pragma HLS UNROLL
		read_row_NNb_TI_patch: for(int patch_y=0; patch_y<NNb_TI_PATCH_WIDTH; patch_y++){
#pragma HLS UNROLL

			TI_addr = (spike_surround_y+patch_y)*DVS_WIDTH + (spike_surround_x+patch_x);
//			PATCH_addr = patch_y*NNb_TI_PATCH_WIDTH+patch_x;	// Read row first
			PATCH_addr = patch_x*NNb_TI_PATCH_WIDTH+patch_y;  // Read column first
			timestamp_polarity_patch[PATCH_addr] = timestamp_polarity_image[TI_addr];   // Read row first (Shasha does like this)

#ifdef VERBOSE
			printf("\t NNb_TI_patch[%u] <-- TI[%u] TI_value: %u\n",	PATCH_addr, TI_addr, timestamp_polarity_image[TI_addr].to_uint());
#endif
		}
	}

#ifdef VERBOSE
	printf("Create activation array for the MLP.\n");
	printf("Age formula: Ax,y = if Tx,y < (Te - tau) then 0 else (1 - (Te-Tx,y) / tau) %% Age [0, 1]\n");
#endif

	concat_mlp_activation: for(int act_idx=0; act_idx<NNb_TI_PATCH_SIZE; act_idx++){
#pragma HLS UNROLL
		// Get the timestamp_pol data from the patch
		ts_pol = timestamp_polarity_patch[act_idx];
		patch_ts = ts_pol.range(TIMESTAMP_POLARITY_IMAGE_DATA_BITS-1, POLARITY_BITS);
		patch_pol = ts_pol.range(POLARITY_BITS-1, 0);

		if(patch_ts < ts_tau_diff){
			age = 0;
		}
		else{
			ts_diff = current_timestamp - patch_ts;
//			age = 1 - (ts_diff) / (float)TAU;
			age = 1 - (ts_diff >> TAU_BITS);
//			printf("ts_diff %f, ts_diff >> 6 %f, age %f", ts_diff.to_float(), (ts_diff>>6).to_float(), age.to_float());
		}
		mlp_activation[act_idx] = age;

#ifdef VERBOSE
		printf("\t Mapping the age of NNb_TI_patch_pixel #%u into Neuron #%u: Calculating the age... Tx,y: %u, Te: %u, Tau: %d --> Age: %f\n",
						act_idx, act_idx, ts_pol.range(TIMESTAMP_POLARITY_IMAGE_DATA_BITS-1, POLARITY_BITS).to_uint(), current_timestamp.to_uint(), TAU, age.to_float());
#endif

		// For the incoming event take its polarity, otherwise get the polarity from the TI
		if(act_idx == CURRENT_EVENT_POSITION){
			pol = spike_pol;
		}else{
			pol = patch_pol;
		}
		mlp_activation[act_idx+NNb_TI_PATCH_SIZE] = pol;

#ifdef VERBOSE
		printf("\t Mapping the pol of NNb_TI_patch_pixel #%u into Neuron #%u: Getting the polarity --> Pol: %.1f\n",
						act_idx, act_idx+NNb_TI_PATCH_SIZE, pol.to_float());
#endif
	}

	// Update the TI with the current event
	timestamp_polarity_image_data_t timestamp_polarity;
	timestamp_polarity.range(timestamp_polarity.width-1, POLARITY_BITS)= current_timestamp;
	timestamp_polarity.range(POLARITY_BITS-1, 0) = spike_pol;
	timestamp_polarity_image[spike_y * DVS_WIDTH + spike_x] = timestamp_polarity;

#ifdef VERBOSE
	printf("Updating the current event (y: %u, x: %u, addr: %u) information (Te: %u, Pol: %d, %d) in the Timestamp Image.\n", spike_y.to_uint(), spike_x.to_uint(),
			spike_y * DVS_WIDTH + spike_x, timestamp_polarity.range(timestamp_polarity.width-1, POLARITY_BITS).to_uint(), timestamp_polarity.range(POLARITY_BITS-1,0).to_int(),
			timestamp_polarity.to_uint());
#endif
}
