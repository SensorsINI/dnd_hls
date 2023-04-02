#include <stdio.h>
#include <string.h>
#include "dnd_top.hpp"
#include "mlp/defines.h"

#define SPIKE_COUNT 1000
// Events captured from the DVS.
const char * input_events_file_path = "/home/arios/Projects/dvs_denoising/dnd_hls_repo/src/tb/data/v2e-dvs-events.txt";


int main() {

	printf("Starting the test to read the Temporal Image and create the activations for the MLP \n");

	ap_fixed<3,1,AP_RND_INF,AP_WRAP_SM> test = -1;
	printf("%f, ", test.to_float());
	test = 1;
	printf("%f\n", test.to_float());

	printf("Create the Timestamp Image (TI)\n");
//	timestamp_polarity_image_data_t timestamp_image[DVS_WIDTH * DVS_HEIGHT];

	printf("Open the input txt spikes file and create the output activation file.\n");
	FILE *spike_f = fopen(input_events_file_path, "rb");
	// Avoid comment lines
	char line[LINE_MAX];
	std::fgets(line, LINE_MAX, spike_f);
	while (line[0] == '#') {
		std::fgets(line, LINE_MAX, spike_f);
	}

	printf("Loop in the input events file and send the spikes (one-by-one) to the dnd_create_mlp_activation module.\n");
	caviar_data_t spike_in;
	timestamp_t current_time;
	mlp_input_activation_t mlp_activations[MLP_INPUT_NEURONS];
	char *token;
	const char delimiter[2] = " ";
	unsigned short size_in;
	unsigned short size_out;
	result_t mlp_out[N_LAYER_6];
	caviar_data_t *spike_out;
	for (int s_id = 0; s_id < SPIKE_COUNT; s_id++) {
		// Get timestamp
		token = strtok(line, delimiter);
		uint ts = (uint) (std::stof(token) * 1000);  // Convert to ms
		current_time = ts;
		// Get spike address+polarity
		token = std::strtok(NULL, delimiter);
		uint16_t x = std::stoi(token);
		token = std::strtok(NULL, delimiter);
		uint16_t y = std::stoi(token);
		token = std::strtok(NULL, delimiter);
		uint16_t pol = std::stoi(token);
		spike_in = 0x00000000 | (y << CAVIAR_X_Y_BITS + 1);
		spike_in = spike_in | (x << 1);
		spike_in += pol;
		printf("Spike #%u --> Ts: %u, X_col: %u, Y_row: %u, Pol: %u. Caviar spike in: %u \t ",
				s_id, current_time.to_uint(), x, y, pol, spike_in.to_uint());

//		caviar_data_t spike_out = dnd_top(spike_in, timestamp_image, current_time);
		caviar_data_t spike_out = dnd_top(spike_in, current_time);

		printf("\tCaviar spike out: %u\n", spike_out.to_uint());

		// Get next spike
		if (std::fgets(line, LINE_MAX, spike_f) == NULL) {
			break;
		}
	}

	fclose(spike_f);
}
