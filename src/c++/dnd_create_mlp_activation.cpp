#include "dnd_create_mlp_activation.hpp"
#define VERBOSE

void dnd_create_mlp_activation(
		caviar_data_t caviar_data,
		timestamp_image_data_t timestamp_image[DVS_WIDTH * DVS_HEIGHT], 	//TODO change the memory access using @Min memory layout
		timestamp_t current_time,
		mlp_input_activation_t mlp_activation[MLP_INPUT_NEURONS]
		) {

#ifdef VERBOSE
	printf("Get the spike column, row and pol from the caviar input\n");
#endif
	// Gets X,Y coordinate and polarity of the current event.
	ap_uint<CAVIAR_X_Y_BITS> caviar_y = caviar_data.range(caviar_data.width-1 , CAVIAR_X_Y_BITS+1);
	ap_uint<CAVIAR_X_Y_BITS> caviar_x = caviar_data.range(CAVIAR_X_Y_BITS, 1);	//Remove the polarity to get the spike address
	polarity_t spike_pol = (caviar_data.range(0, 0) == 1) ? 1 : -1; 			//MLP polarity: 0-> No or old spike; 1-> ON; -1-> OFF

#ifdef VERBOSE
	printf("Read the NNb_TI_patch (Timestamp:Polarity) from the Timestamp Image.\n");
#endif
	timestamp_image_data_t timestamp_polarity_patch[NNb_TI_PATCH_WIDTH*NNb_TI_PATCH_WIDTH];

	uint TI_addr, PATCH_addr;
	read_col_NNb_TI_patch: for(int patch_x=0; patch_x<NNb_TI_PATCH_WIDTH; patch_x++){
		read_row_NNb_TI_patch: for(int patch_y=0; patch_y<NNb_TI_PATCH_WIDTH; patch_y++){

			TI_addr = ((caviar_y-floor(NNb_TI_PATCH_WIDTH/2)+patch_y)*DVS_WIDTH + (caviar_x-floor(NNb_TI_PATCH_WIDTH/2))+patch_x);
//			PATCH_addr = patch_y*NNb_TI_PATCH_WIDTH+patch_x;	// Read row first
			PATCH_addr = patch_x*NNb_TI_PATCH_WIDTH+patch_y;  // Read column first
			timestamp_polarity_patch[PATCH_addr] = timestamp_image[TI_addr];   // Read row first (Shasha does like this)
#ifdef VERBOSE
			printf("\t NNb_TI_patch[%u] <-- TI[%u] TI_value: %u\n",	PATCH_addr, TI_addr, timestamp_image[TI_addr].to_uint());
#endif
		}
	}

#ifdef VERBOSE
	printf("Create activation array for the MLP.\n");
#endif
	mlp_input_activation_t age;
	mlp_input_activation_t pol;
	int activation_pol;
	timestamp_diff_t ts_diff;
#ifdef VERBOSE
	printf("Age formula: Ax,y = if Tx,y < (Te - tau) then 0 else (1 - (Te-Tx,y) / tau) %% Age [0, 1]\n");
#endif
	concat_mlp_activation: for(int idx=0; idx<NNb_TI_PATCH_SIZE; idx++){

		// Get the timestamp_pol data from the patch
		timestamp_image_data_t ts_pol = timestamp_polarity_patch[idx];
		if(ts_pol.range(TIMESTAMP_IMAGE_DATA_BITS-1, POLARITY_BITS) < (current_time - TAU)){
			age = 0;
		}
		else{
			ts_diff = current_time - ts_pol.range(TIMESTAMP_IMAGE_DATA_BITS-1, POLARITY_BITS);
//			age = 1 - (ts_diff) / (float)TAU;
			age = 1 - (ts_diff >> TAU_BITS);
//			printf("ts_diff %f, ts_diff >> 6 %f, age %f", ts_diff.to_float(), (ts_diff>>6).to_float(), age.to_float());
		}
		mlp_activation[idx] = age;
#ifdef VERBOSE
		printf("\t Mapping the age of NNb_TI_patch_pixel #%u into Neuron #%u: Calculating the age... Tx,y: %u, Te: %u, Tau: %d --> Age: %f\n",
						idx, idx, ts_pol.range(TIMESTAMP_IMAGE_DATA_BITS-1, POLARITY_BITS).to_uint(), current_time.to_uint(), TAU, age.to_float());
#endif

		// For the incoming event take its polarity, otherwise get the polarity from the TI
		if(idx == CURRENT_EVENT_POSITION){
			activation_pol = spike_pol;
		}else{
			activation_pol = ts_pol.range(POLARITY_BITS-1, 0);
		}
		mlp_activation[idx+NNb_TI_PATCH_SIZE] = mlp_input_activation_t(activation_pol);
#ifdef VERBOSE
		printf("\t Mapping the pol of NNb_TI_patch_pixel #%u into Neuron #%u: Getting the polarity --> Pol: %d (%f)\n",
						idx, idx+NNb_TI_PATCH_SIZE, activation_pol, mlp_input_activation_t(activation_pol).to_float());
#endif
	}

	// Update the TI with the current event
	timestamp_image_data_t timestamp_polarity;
	timestamp_polarity.range(timestamp_polarity.width-1, POLARITY_BITS)= current_time;
	timestamp_polarity.range(POLARITY_BITS-1, 0) = spike_pol;
	timestamp_image[caviar_y * DVS_WIDTH + caviar_x] = timestamp_polarity;
	printf("Updating the current event (y: %u, x: %u, addr: %u) information (Te: %u, Pol: %d, %d) in the Timestamp Image.\n", caviar_y.to_uint(), caviar_x.to_uint(),
			caviar_y * DVS_WIDTH + caviar_x, timestamp_polarity.range(timestamp_polarity.width-1, POLARITY_BITS).to_uint(), timestamp_polarity.range(POLARITY_BITS-1,0).to_int(),
			timestamp_polarity.to_uint());
}
