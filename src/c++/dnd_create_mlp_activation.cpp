#include "dnd_create_mlp_activation.hpp"

void dnd_create_mlp_activation(
		caviar_data_t caviar_data,
		timestamp_image_data_t timestamp_image[DVS_WIDTH * DVS_HEIGHT],
		timestamp_t current_time,
		mlp_input_activation_t mlp_activation[MLP_INPUT_NEURONS]
		) {

	// Define some useful variables
	ap_uint<ROW_COL_BITS> caviar_col = caviar_data.range(caviar_data.width-1 , ROW_COL_BITS+1);
	ap_uint<ROW_COL_BITS> caviar_row = caviar_data.range(ROW_COL_BITS, 1);	//Remove the polarity to get the spike address
	timestamp_image_data_t timestamp_polarity_patch[NNb_TI_PATCH_WIDTH*NNb_TI_PATCH_WIDTH];

	// Save the timestamp and the polarity for the current spike into the timestamp image
//	caviar_data_t spike_addr = caviar_data >> 1;
	polarity_t spike_pol = (caviar_data.range(0, 0) == 1) ? 1 : 2;					//MLP polarity: 0-> No or old spike; 1-> ON; 2-> OFF
	timestamp_image_data_t timestamp_polarity = (current_time << 2) | spike_pol;
	timestamp_image[caviar_row * DVS_WIDTH + caviar_col] = timestamp_polarity;
	uint ts_pol = timestamp_polarity.to_uint();
	uint addr = caviar_row.to_uint() * DVS_WIDTH + caviar_col.to_uint();
	printf("Addr: %u, ts_pol: %u\n", addr, ts_pol);


	// Read the NNb_TI_patch (Timestamp:Polarity) from the timestamp image
	read_row_NNb_TI_patch: for(int patch_row=0; patch_row<NNb_TI_PATCH_WIDTH; patch_row++){
		read_col_NNb_TI_patch: for(int patch_col=0; patch_col<NNb_TI_PATCH_WIDTH; patch_col++){
			addr = ((caviar_row-floor(NNb_TI_PATCH_WIDTH/2)+patch_row)*DVS_WIDTH + (caviar_col-floor(NNb_TI_PATCH_WIDTH/2))+patch_col);
			timestamp_polarity_patch[patch_row*NNb_TI_PATCH_WIDTH+patch_col] = timestamp_image[addr];
		}
	}

	// Send activation to MLP
	//mlp_input_data_t mlp_activation_tmp;
	mlp_input_activation_t age;
	mlp_input_activation_t pol;
	concat_mlp_activation: for(int idx=0; idx<MLP_INPUT_NEURONS; idx+=2){
		if(timestamp_polarity_patch[idx].range(TIMESTAMP_IMAGE_DATA_BITS-1, 2) < (current_time - TAU)){
			age = 0;
		}
		else{
			age = 1 - (float)(current_time - (timestamp_polarity_patch[idx].range(TIMESTAMP_IMAGE_DATA_BITS-1, 2))) / (float)TAU;
		}
		pol = timestamp_polarity_patch[idx].range(1, 0);
		mlp_activation[2*idx+0] = age;
		mlp_activation[2*idx+1] = pol;
	}
}
