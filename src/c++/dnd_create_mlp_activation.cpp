#include "dnd_create_mlp_activation.hpp"
//#define VERBOSE

void dnd_create_mlp_activation(
		caviar_data_t caviar_spike,
		timestamp_polarity_image_data_t timestamp_polarity_image[DVS_WIDTH * DVS_HEIGHT], 	//TODO change the memory access using @Min memory layout
//		timestamp_polarity_image_data_t timestamp_polarity_image[DVS_WIDTH][DVS_HEIGHT/COMBINED_SPIKES],
		timestamp_t current_timestamp,
		mlp_input_activation_t mlp_activation[MLP_INPUT_NEURONS]
		) {
#pragma HLS DATAFLOW
#ifdef VERBOSE
	printf("Get the spike column, row and pol from the caviar input\n");
#endif

	ap_uint<CAVIAR_X_Y_BITS> spike_y = caviar_spike.range(caviar_spike.width-1 , CAVIAR_X_Y_BITS+1);	// Get y spike coordinate
	ap_uint<CAVIAR_X_Y_BITS> spike_x = caviar_spike.range(CAVIAR_X_Y_BITS, 1);							// Get x spike coordinate
	polarity_t spike_pol = (caviar_spike.range(0, 0) == 1) ? 1 : -1; 									// Get spike polarity and change it to MLP polarity: 0-> No or old spike; 1-> ON; -1-> OFF

	// Set some useful variables
	timestamp_polarity_data_t timestamp_polarity_patch[NNb_TI_PATCH_WIDTH*NNb_TI_PATCH_WIDTH];
	uint TI_addr, PATCH_addr;
	int patch_border = NNb_TI_PATCH_WIDTH >> 1;
	int spike_surround_y = spike_y - patch_border;														// Calculate the y coordinate of the first pixel of the patch
	int spike_surround_x = spike_x - patch_border;														// Calculate the x coordinate of the first pixel of the patch
	mlp_input_activation_t age;
	mlp_input_activation_t pol;
	timestamp_diff_t ts_diff;
	int ts_tau_diff = current_timestamp - TAU;															// Calculate the difference between TAU and the current timestamp for further use
//	int act_idx = 0;
	timestamp_polarity_data_t ts_pol;
	timestamp_t patch_ts;
	polarity_t patch_pol;

//	timestamp_polarity_image_data_t *whole_colum;
//	timestamp_polarity_image_data_t combined_spike, combined_spike_next;
//	int spike_combined_idx, t_p_idx, t_p_start_bit, t_p_end_bit;

#ifdef VERBOSE
	printf("Read the NNb_TI_patch (Timestamp:Polarity) from the Timestamp Image.\n");
#endif

	read_col_NNb_TI_patch: for(int patch_x=0; patch_x<NNb_TI_PATCH_WIDTH; patch_x++){
#pragma HLS UNROLL
		read_row_NNb_TI_patch: for(int patch_y=0; patch_y<NNb_TI_PATCH_WIDTH; patch_y++){
#pragma HLS UNROLL

			TI_addr 	= (spike_surround_y+patch_y)*DVS_WIDTH + (spike_surround_x+patch_x);			// Calculate the TPI memory address to get the information (ts and pol) of the next spike that belong to the patch
			PATCH_addr 	= patch_x*NNb_TI_PATCH_WIDTH+patch_y;  // Read column first						// Calculate the PATCH memory address where the information from the TPI memory will be stored

			timestamp_polarity_patch[PATCH_addr] = timestamp_polarity_image[TI_addr];					// Store the information from the TPI memory to PATCH

#ifdef VERBOSE
			printf("\t NNb_TI_patch[%u] <-- TI[%u] TI_value: %u\n",	PATCH_addr, TI_addr, timestamp_polarity_image[TI_addr].to_uint());
#endif
		}
	}
//	read_col_NNb_TI_patch: for(int patch_x=0; patch_x<NNb_TI_PATCH_WIDTH; patch_x++){
//#pragma HLS UNROLL
//		spike_combined_idx = spike_surround_y >> 5;														// Get the index of the combined spike where the first patch pixel belongs
//		combined_spike = timestamp_polarity_image[spike_surround_x+patch_x][spike_combined_idx];		// Get the combined spike data
//		combined_spike_next = timestamp_polarity_image[spike_surround_x+patch_x][spike_combined_idx+1];	// Get the next combined spike data
//		t_p_idx = spike_surround_y % COMBINED_SPIKES;													// Get index of the ts_pol data inside of the combined spike
//
//		read_row_NNb_TI_patch: for(int patch_y=0; patch_y<NNb_TI_PATCH_WIDTH; patch_y++){
//#pragma HLS UNROLL
//			PATCH_addr 		= patch_x * NNb_TI_PATCH_WIDTH + patch_y;  									// Calculate the address of the patch for the current ts_pol data (Read column first)
//			t_p_start_bit 	= t_p_idx * (TIMESTAMP_POLARITY_DATA_BITS);									// Get the LSB of the ts_pol data
//			t_p_end_bit 	= t_p_start_bit + TIMESTAMP_POLARITY_DATA_BITS - 1;							// Get the MSB of the ts_pol data
//			ts_pol 			= combined_spike.range(t_p_end_bit, t_p_start_bit);							// Get the ts_pol data from the combined spike
//
//			timestamp_polarity_patch[PATCH_addr] = ts_pol;												// Store the ts_pol data in the patch
//			if(t_p_idx + patch_y == COMBINED_SPIKES - 1){												// Check if the combined spike limit is reached
//				combined_spike = combined_spike_next;													// Update the combined spike with the next one
//				t_p_idx = 0;																			// Reset the index of the ts_pol data to zero
//			}else{
//				t_p_idx++;																				// Point to the next ts_pol data inside of the combined spike
//			}
//#ifdef VERBOSE
////			printf("\t NNb_TI_patch[%u] <-- TI[%u] TI_value: %u\n",	PATCH_addr, TI_addr, ts_pol.to_uint());
//			printf("\t NNb_TI_patch[%u] <-- TPI_value: %u\n",	PATCH_addr, ts_pol.to_uint());
//#endif
//		}
//	}

#ifdef VERBOSE
	printf("Create activation array for the MLP.\n");
	printf("Age formula: Ax,y = if Tx,y < (Te - tau) then 0 else (1 - (Te-Tx,y) / tau) %% Age [0, 1]\n");
#endif

	concat_mlp_activation: for(int act_idx=0; act_idx<NNb_TI_PATCH_SIZE; act_idx++){						// Loop over the PATCH to compute the activations [age and pol]
#pragma HLS UNROLL
		ts_pol = timestamp_polarity_patch[act_idx];															// Get next data from the PATCH
		patch_ts = ts_pol.range(TIMESTAMP_POLARITY_DATA_BITS-1, POLARITY_BITS);								// Get the timestamp
		patch_pol = ts_pol.range(POLARITY_BITS-1, 0);														// Get the polarity

		if(patch_ts < ts_tau_diff){																			// Compute the age according to the formula from the paper
			age = 0;
		}
		else{
			ts_diff = current_timestamp - patch_ts;
			age = 1 - (ts_diff >> TAU_BITS);
		}
		mlp_activation[act_idx] = age;																		// Put the age on the right position of the activation array

#ifdef VERBOSE
		printf("\t Mapping the age of NNb_TI_patch_pixel #%u into Neuron #%u: Calculating the age... Tx,y: %u, Te: %u, Tau: %d --> Age: %f\n",
						act_idx, act_idx, ts_pol.range(TIMESTAMP_POLARITY_DATA_BITS-1, POLARITY_BITS).to_uint(), current_timestamp.to_uint(), TAU, age.to_float());
#endif

		if(act_idx == CURRENT_EVENT_POSITION){																// For the incoming event take its polarity, otherwise get the polarity from the TI
			pol = spike_pol;
		}else{
			pol = patch_pol;
		}
		mlp_activation[act_idx+NNb_TI_PATCH_SIZE] = pol;													// Put the pol on the right position of the activation array

#ifdef VERBOSE
		printf("\t Mapping the pol of NNb_TI_patch_pixel #%u into Neuron #%u: Getting the polarity --> Pol: %.1f\n",
						act_idx, act_idx+NNb_TI_PATCH_SIZE, pol.to_float());
#endif
	}

	timestamp_polarity_image_data_t timestamp_polarity;
	timestamp_polarity.range(timestamp_polarity.width-1, POLARITY_BITS)= current_timestamp;				// Put the current timestamp in the ts_pol data
	timestamp_polarity.range(POLARITY_BITS-1, 0) = spike_pol;											// Put the pol in the ts_pol data
	timestamp_polarity_image[spike_y * DVS_WIDTH + spike_x] = timestamp_polarity;						// Update the TPI with the current spike information (ts and pol) in the address corresponding to the current spike

//	// Update the TPI with the current event
//	timestamp_polarity_data_t timestamp_polarity;
//	timestamp_polarity.range(timestamp_polarity.width-1, POLARITY_BITS)	= current_timestamp;
//	timestamp_polarity.range(POLARITY_BITS-1, 0) 						= spike_pol;
//
//	spike_combined_idx 		= spike_y >> 5;																// Get the index of the combined spike where the current spike pixel belongs
//	combined_spike 			= timestamp_polarity_image[spike_x][spike_combined_idx];					// Get the combined spike data
//	t_p_idx				   	= spike_y % COMBINED_SPIKES;												// Get index of the ts_pol data inside of the combined spike
//	t_p_start_bit 			= t_p_idx * (TIMESTAMP_POLARITY_DATA_BITS);									// Get the LSB of the timestamp_polarity data
//	t_p_end_bit 			= t_p_start_bit + TIMESTAMP_POLARITY_DATA_BITS -1;							// Get the MSB of the timestamp_polarity data
//
//	combined_spike.range(t_p_end_bit, t_p_start_bit)	= timestamp_polarity;							// Store the ts_pol data inside the combined spike
//	timestamp_polarity_image[spike_x][spike_y] 			= combined_spike;								// Update the TPI with the new data

#ifdef VERBOSE
	printf("Updating the current event (y: %u, x: %u, addr: %u) information (Te: %u, Pol: %d, %d) in the Timestamp Image.\n", spike_y.to_uint(), spike_x.to_uint(),
			spike_y * DVS_WIDTH + spike_x, timestamp_polarity.range(timestamp_polarity.width-1, POLARITY_BITS).to_uint(), timestamp_polarity.range(POLARITY_BITS-1,0).to_int(),
			timestamp_polarity.to_uint());
#endif
}
