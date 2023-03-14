#include "dnd_parameters.hpp"

void dnd_create_mlp_activation(
		caviar_data_t caviar_spike,
		timestamp_polarity_image_data_t timestamp_polarity_image[DVS_WIDTH * DVS_HEIGHT],
		timestamp_t current_timestamp,
		mlp_input_activation_t mlp_activation[MLP_INPUT_NEURONS]
		);
