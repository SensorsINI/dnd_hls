// This file is part of https://github.com/SensorsINI/dnd_hls.
// This intellectual property is licensed under the terms of the project license available at the root of the project.

#include "dnd_parameters.hpp"
#include "dnd_create_mlp_activation.hpp"
#include "mlp/myproject.h"


caviar_data_t dnd_top(
		caviar_data_t spike_in,
//		timestamp_polarity_image_data_t timestamp_polarity_image[DVS_WIDTH * DVS_HEIGHT],
//		timestamp_polarity_image_data_t timestamp_polarity_image[DVS_WIDTH][DVS_HEIGHT/COMBINED_SPIKES], // DIFFERENT MEMORY INTERFACE.
		timestamp_t current_timestamp
		);
