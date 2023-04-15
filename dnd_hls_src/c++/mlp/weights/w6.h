// This file is part of https://github.com/SensorsINI/dnd_hls. 
// This intellectual property is licensed under the terms of the project license available at the root of the project.
//Numpy array shape [10, 1]
//Min -1.000000000000
//Max 0.875000000000
//Number of zeros 0

#ifndef W6_H_
#define W6_H_

#ifndef __SYNTHESIS__
weight6_t w6[10];
#else
weight6_t w6[10] = {0.875, -0.875, 0.625, -0.625, 0.625, -0.875, -1.000, 0.625, -1.000, 0.500};
#endif

#endif
