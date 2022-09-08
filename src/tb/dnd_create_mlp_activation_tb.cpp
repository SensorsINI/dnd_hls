#include <stdio.h>
#include <string.h>

#include "dnd_create_mlp_activation.hpp"

#define SPIKE_COUNT 1000

int main(){

	printf("Starting the test for read the Temporal Image and create the activations for the MLP \n");

	// Create a Timestamp image with 0 values
	timestamp_image_data_t timestamp_image[DVS_WIDTH * DVS_HEIGHT];
//	for(int idx=0; idx<DVS_WIDTH*DVS_HEIGHT; idx++){
//		timestamp_image[idx] = 0;
//	}

	// Open spike txt file
	FILE *spike_f = fopen("v2e-dvs-events.txt", "rb");
	// Avoid comment lines
	char line[LINE_MAX];
	std::fgets(line, LINE_MAX, spike_f);
	while(line[0] == '#'){
		std::fgets(line, LINE_MAX, spike_f);
	}

	// Create the output file where activation will be written per line
	FILE* activation_f = fopen("activations.txt", "wb");
	// Write header information
	for(int idx=0; idx<MLP_INPUT_NEURONS; idx++){
		if(idx/10 == 0){
			fprintf(activation_f, "Act._N%d     ", idx);
		}else{
			fprintf(activation_f, "Act._N%d    ", idx);
		}
	}
	fprintf(activation_f, "\n");

	// Send the spikes to the dnd_create_mlp_activation module
	caviar_data_t caviar_data;
	timestamp_t current_time;
	mlp_input_activation_t mlp_activations[MLP_INPUT_NEURONS];
	char* token;
	const char delimiter[2] = " ";
	for(int s_id=0; s_id<SPIKE_COUNT; s_id++){
		// Get timestamp
		token = strtok(line, delimiter);
		uint ts = (uint)(std::stof(token)*1000000);
		current_time = ts;
		// Get spike address+polarity
		token = std::strtok(NULL, delimiter);
		uint16_t row = std::stoi(token);
		token = std::strtok(NULL, delimiter);
		uint16_t col = std::stoi(token);
		token = std::strtok(NULL, delimiter);
		uint16_t pol = std::stoi(token);
		caviar_data = 0x00000000 | (row << ROW_COL_BITS+1);
		caviar_data = caviar_data | (col << 1);
		caviar_data += pol;
//		printf("Timestamp: %d us | Spike: row %d, col %d, pol %d; ROW_COL_BITS %d --> Caviar data: %d\n",
//								spike_timestamp.to_int(), row, col, pol, ROW_COL_BITS, caviar_data.to_int());

		dnd_create_mlp_activation(caviar_data, timestamp_image, current_time, mlp_activations);

		// Write the activation into the ouput file
		for(int idx=0; idx<MLP_INPUT_NEURONS; idx++){
			fprintf(activation_f, "%f\t", mlp_activations[idx].to_float());
		}
		fprintf(activation_f, "\n");

		// Get next spike
		std::fgets(line, LINE_MAX, spike_f);
	}

	fclose(activation_f);
	fclose(spike_f);
	return 0;
}
