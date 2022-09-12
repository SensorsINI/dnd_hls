#ifndef ABMOFACCEL
#define ABMOFACCEl

#include<stdint.h>
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <math.h>

#define DEBUG 0
#define CUST_DATA_MASK 0x3ff
#define POLARITY_SHIFT 11
#define POLARITY_MASK (1 << POLARITY_SHIFT)  // 1 bit at bit 11
#define POLARITY_Y_ADDR_SHIFT 22
#define POLARITY_Y_ADDR_MASK (511 << POLARITY_Y_ADDR_SHIFT) // 9 bits from bits 22 to 30
#define POLARITY_X_ADDR_SHIFT 12
#define POLARITY_X_ADDR_MASK (1023 << POLARITY_X_ADDR_SHIFT) // 10 bits from bits 12 to 21

#define AEDAT_POLARITY_SHIFT 11
#define AEDAT_POLARITY_MASK (1 << AEDAT_POLARITY_SHIFT)  // 1 bit at bit 11
#define AEDAT_POLARITY_Y_ADDR_SHIFT 22
#define AEDAT_POLARITY_Y_ADDR_MASK (511 << AEDAT_POLARITY_Y_ADDR_SHIFT) // 9 bits from bits 22 to 30
#define AEDAT_POLARITY_X_ADDR_SHIFT 12
#define AEDAT_POLARITY_X_ADDR_MASK (1023 << AEDAT_POLARITY_X_ADDR_SHIFT) // 10 bits from bits 12 to 21

#define SLICES_NUMBER 4
#define SLICE_WIDTH  512
#define SLICE_HEIGHT 512

#define DVS_WIDTH  346
#define DVS_HEIGHT 260

#define BITS_PER_PIXEL 4
#define COMBINED_PIXELS 32

// ABMOF parameters, hardcoded:
#define BLOCK_SIZE 11
#define BLOCK_SIZE_SCALE_2 7
#define BLOCK_SIZE_SCALE_1 13
#define BLOCK_SIZE_SCALE_0 25

#define COL_SUM_BITS 16
#define VALID_CNT_BITS 6

#define BLOCK_AREA (BLOCK_SIZE * BLOCK_SIZE)
#define BLOCK_AREA_SCALE_0 (BLOCK_SIZE_SCALE_0 * BLOCK_SIZE_SCALE_0)
#define BLOCK_AREA_SCALE_1 (BLOCK_SIZE_SCALE_1 * BLOCK_SIZE_SCALE_1)
#define BLOCK_AREA_SCALE_2 (BLOCK_SIZE_SCALE_2 * BLOCK_SIZE_SCALE_2)

#define SEARCH_DISTANCE 3
#define MAX_SEARCH_DIST_RADIUS ((4+2+1)*SEARCH_DISTANCE)
#define AREA_NUMBER 32
#define AREA_SIZE (SLICE_WIDTH/AREA_NUMBER)
#define INIT_AREA_THERSHOLD 700
#define MAX_SLICE_DURATION_US 300000

// Number of Parallel Computing units
#define NPC_SCALE_0 3
#define NPC_SCALE_1 1

const float glIterCntNPCScale0Float = float(BLOCK_SIZE_SCALE_0)/float(NPC_SCALE_0);
const int glConstIterCntNPCScale0 = ceil(glIterCntNPCScale0Float);
#define ITER_CNT_NPC_SCALE_0 ceil(float(BLOCK_SIZE_SCALE_0)/float(NPC_SCALE_0))

const float glIterCntNPCScale1Float = float(BLOCK_SIZE_SCALE_1)/float(NPC_SCALE_1);
const int glConstIterCntNPCScale1 = ceil(glIterCntNPCScale1Float);
#define ITER_CNT_NPC_SCALE_1 ceil(float(BLOCK_SIZE_SCALE_1)/float(NPC_SCALE_1))

// Number of Parallel Computing units
// Must make sure ITER_CNT_NPC_SCALE * NPC_SCALE <= BLOCK_SIZE_SCALE + 2 * SEARCH_DISTANCE
// when choosing NPC for a scale.
// For example, 8 is not suitable for scale 0 if BLOCK_SIZE_SCALE_0 = 25 and SEARCH_DISTANCE = 3
// because ITER_CNT_NPC_SCALE * NPC_SCALE for scale 0 is 32, but BLOCK_SIZE_SCALE + 2 * SEARCH_DISTANCE
// for scale 0 is 31. However, bigger value such as 9 is okay.
#define NPC_SCALE_0 3
#define NPC_SCALE_1 1

// Valid pixel occupancy parameter
const float glValidPixOccupancy = 0.01;
const int glMinValidPixNum = int(glValidPixOccupancy * BLOCK_AREA);
const int glMinValidPixNumScale0 = int(glValidPixOccupancy * BLOCK_AREA_SCALE_0);
const int glMinValidPixNumScale1 = int(glValidPixOccupancy * BLOCK_AREA_SCALE_1);
const int glMinValidPixNumScale2 = int(glValidPixOccupancy * BLOCK_AREA_SCALE_2);

// max allowed sad distance, normalized. Same as jaer.
// To remove float computation on hardware, convert it back to int and only support weightDistance = 1.0
const float maxAllowedSadDistance = 0.5;
const int maxAllowedSadValueScale0 = ap_uint<BITS_PER_PIXEL>(0xffff) * 2 * BLOCK_AREA_SCALE_0;
const int maxAllowedSadValueScale1 = ap_uint<BITS_PER_PIXEL>(0xffff) * 2 * BLOCK_AREA_SCALE_1;
const int maxAllowedSadValueScale2 = ap_uint<BITS_PER_PIXEL>(0xffff) * 2 * BLOCK_AREA_SCALE_2;

#define BLOCK_COL_PIXELS BITS_PER_PIXEL * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)
#define BLOCK_SCALE0_COL_PIXELS BITS_PER_PIXEL * (BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE)
#define BLOCK_SCALE1_COL_PIXELS BITS_PER_PIXEL * (BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE)
#define BLOCK_SCALE2_COL_PIXELS BITS_PER_PIXEL * (BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE)

#define COL_BITS BITS_PER_PIXEL * (BLOCK_SIZE)

typedef ap_axiu<64,1,1,1> inputDataElement;
typedef ap_axiu<32,1,1,1> outputDataElement_t;

typedef ap_uint<BITS_PER_PIXEL> pix_t;
typedef ap_int<COMBINED_PIXELS * BITS_PER_PIXEL> col_pix_t;
typedef ap_int<COMBINED_PIXELS * BITS_PER_PIXEL * 2> two_cols_pix_t;
typedef ap_uint<2> sliceIdx_t;

typedef ap_int<BLOCK_COL_PIXELS> apIntBlockCol_t;
typedef ap_int<BLOCK_SCALE0_COL_PIXELS> apIntBlockScale0Col_t;
typedef ap_int<BLOCK_SCALE1_COL_PIXELS> apIntBlockScale1Col_t;
typedef ap_int<BLOCK_SCALE2_COL_PIXELS> apIntBlockScale2Col_t;

typedef ap_uint<COL_SUM_BITS * (2 * SEARCH_DISTANCE + 1)> apUintColSum_t;
typedef ap_uint<VALID_CNT_BITS * (2 * SEARCH_DISTANCE + 1)> apUintValidCnt_t;

// For scale 0
typedef ap_uint<COL_SUM_BITS * (2 * SEARCH_DISTANCE + 1) * NPC_SCALE_0> apUintColSumNPC_t;
typedef ap_uint<VALID_CNT_BITS * (2 * SEARCH_DISTANCE + 1) * NPC_SCALE_0> apUintValidCntNPC_t;
typedef ap_uint<6 * NPC_SCALE_0> apUintRefZeroCntNPC_t;
typedef ap_int<BLOCK_SCALE0_COL_PIXELS * NPC_SCALE_0> apIntBlockScale0ColNPC_t;

// For scale 1
typedef ap_uint<COL_SUM_BITS * (2 * SEARCH_DISTANCE + 1) * NPC_SCALE_1> apUintColSumScale1NPC_t;
typedef ap_uint<VALID_CNT_BITS * (2 * SEARCH_DISTANCE + 1) * NPC_SCALE_1> apUintValidCntScale1NPC_t;
typedef ap_uint<6 * NPC_SCALE_1> apUintRefZeroCntScale1NPC_t;
typedef ap_int<BLOCK_SCALE1_COL_PIXELS * NPC_SCALE_1> apIntBlockScale1ColNPC_t;

//typedef ap_int<COL_BITS> apIntColBits_t;
typedef ap_uint<17> apUint17_t;
typedef ap_uint<16> apUint16_t;
typedef ap_uint<15> apUint15_t;
typedef ap_uint<6> apUint6_t;
typedef ap_uint<1> apUint1_t;
typedef ap_uint<16 * (2 * SEARCH_DISTANCE + 1)> apUint112_t;
typedef ap_uint<6 * (2 * SEARCH_DISTANCE + 1)> apUint42_t;
typedef ap_uint<10> apUint10_t;

#define BLOCK_COL_PIXELS BITS_PER_PIXEL * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)
#define PIXS_PER_COL (SLICE_HEIGHT/COMBINED_PIXELS)

typedef struct
{
	uint32_t currentDeltaTSHW;
	uint32_t currentAreaCntThr;
} status_t;

void writePix(ap_uint<8> x, ap_uint<8> y, sliceIdx_t sliceIdx);
pix_t readPix(ap_uint<8> x, ap_uint<8> y, sliceIdx_t sliceIdx);

void topHW(ap_uint<8> x, ap_uint<8> y, sliceIdx_t idx, ap_int<16> *miniSumRet);

ap_int<16> min(ap_int<16> inArr[2*SEARCH_DISTANCE + 1], int8_t *index);

void testTemp(uint64_t * data, sliceIdx_t idx, int16_t eventCnt,
		int32_t *eventSlice);

void feedback(apUint15_t miniSumRet, apUint6_t OFRet, apUint1_t rotateFlg, uint16_t *thrRet);

void parseEvents(uint64_t * dataStream, int32_t eventsArraySize, int32_t *eventSlice);

void EVABMOFStream(ap_uint<16> xIn, ap_uint<16> yIn, ap_uint<64> tsIn, ap_uint<1> polIn,
		ap_uint<16> *xOut, ap_uint<16> *yOut, ap_uint<64> *tsOut, ap_uint<1> *polOut,
		apUint17_t *pixelDataOut,
		ap_uint<32> config, ap_uint<32> *status);

void EVABMOFScalar(ap_uint<16> xIn, ap_uint<16> yIn, ap_uint<64> tsIn, ap_uint<1> polIn,
		ap_uint<16> *xOut, ap_uint<16> *yOut, ap_uint<64> *tsOut, ap_uint<1> *polOut,
		apUint17_t *pixelOut,
		ap_uint<32> config, ap_uint<32> *status);

void EVABMOFStreamNoConfigNoStaus(hls::stream< ap_uint<16> > &xStreamIn, hls::stream< ap_uint<16> > &yStreamIn, hls::stream< ap_uint<64> > &tsStreamIn, hls::stream< ap_uint<1> > &polStreamIn,
		hls::stream< ap_uint<16> > &xStreamOut, hls::stream< ap_uint<16> > &yStreamOut, hls::stream< ap_uint<64> > &tsStreamOut, hls::stream< ap_uint<1> > &polStreamOut,
		hls::stream< ap_uint<17> > &pixelDataStream);

void EVABMOFStreamWithControl(hls::stream< ap_uint<16> > &xStreamIn, hls::stream< ap_uint<16> > &yStreamIn,
		hls::stream< ap_uint<64> > &tsStreamIn, hls::stream< ap_uint<1> > &polStreamIn,
		hls::stream< ap_uint<1> > &controlStreamIn,
		hls::stream< ap_uint<16> > &xStreamOut, hls::stream< ap_uint<16> > &yStreamOut, hls::stream< ap_uint<64> > &tsStreamOut, hls::stream< ap_uint<1> > &polStreamOut,
		hls::stream< apUint17_t > &pixelDataStream,
		ap_uint<32> config, ap_uint<32> *status);
#endif
