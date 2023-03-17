//Numpy array shape [10]
//Min -0.375000000000
//Max 0.500000000000
//Number of zeros 0

#ifndef B3_H_
#define B3_H_

#ifndef __SYNTHESIS__
bias3_t b3[10];
#else
bias3_t b3[10] = {-0.375, 0.375, -0.250, 0.500, -0.375, 0.500, 0.375, -0.375, 0.500, -0.250};
#endif

#endif
