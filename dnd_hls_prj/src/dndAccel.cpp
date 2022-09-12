#include <iostream>
#include "ap_int.h"
#include "dndAccel.h"
//#include "hls_math.h"
#include "utils/x_hls_utils.h"

static col_pix_t glPLSlices[SLICES_NUMBER][SLICE_WIDTH][SLICE_HEIGHT/COMBINED_PIXELS];
static col_pix_t glPLSlicesScale1[SLICES_NUMBER][SLICE_WIDTH/2][SLICE_HEIGHT/COMBINED_PIXELS/2];
static col_pix_t glPLSlicesScale2[SLICES_NUMBER][SLICE_WIDTH/4][SLICE_HEIGHT/COMBINED_PIXELS/4];
static sliceIdx_t glPLActiveSliceIdx = 0, glPLTminus1SliceIdx, glPLTminus2SliceIdx;
sliceIdx_t oldIdx = glPLActiveSliceIdx;
sliceIdx_t oldIdxScale1 = glPLActiveSliceIdx;
sliceIdx_t oldIdxScale2 = glPLActiveSliceIdx;

// This flag is only for ABMOF processing
apUint1_t glRotateFlg = 0;
// This flag is only for forward bypass directly.
apUint1_t glRotateFlgBypass = 0;

static uint16_t eventIterSize = 100;

static hls::stream<uint16_t> glThrStream("glThresholdStream");

static ap_uint<13> resetCnt, resetCntScale0, resetCntScale1, resetCntScale2;

static ap_uint<1> areaCountExceeded = false;

static ap_uint<32> glConfig;
static status_t glStatus;
uint16_t glSFASTAreaCntThr = INIT_AREA_THERSHOLD, glSFASTAreaCntThrBak = glSFASTAreaCntThr; // Init value

#define INPUT_COLS 4

// Function Description: return the minimum value of an array.
ap_int<16> min(ap_int<16> inArr[2*SEARCH_DISTANCE + 1], int8_t *index)
{
#pragma HLS PIPELINE
#pragma HLS ARRAY_RESHAPE variable=inArr complete dim=1
#pragma HLS INLINE off
	ap_int<16> tmp = inArr[0];
	int8_t tmpIdx = 0;
	minLoop: for(int8_t i = 0; i < 2*SEARCH_DISTANCE + 1; i++)
	{
		// Here is a bug. Use the if-else statement,
		// cannot use the question mark statement.
		// Otherwise a lot of muxs will be generated,
		// DON'T KNOW WHY. SHOULD BE A BUG.
		if(inArr[i] < tmp) tmpIdx = i;
		if(inArr[i] < tmp) tmp = inArr[i];
//		tmp = (inArr[i] < tmp) ? inArr[i] : tmp;
	}
	*index = tmpIdx;
	return tmp;
}

ap_int<16> minWide(apUintColSum_t inData, int8_t *index)
{
#pragma HLS PIPELINE
#pragma HLS INLINE off
	ap_int<16> tmp = inData.range(COL_SUM_BITS - 1, 0);
	inData = inData >> COL_SUM_BITS;
	int8_t tmpIdx = 0;
	minLoop: for(int8_t i = 1; i < 2*SEARCH_DISTANCE + 1; i++)
	{
		ap_int<16> currentData = inData.range(COL_SUM_BITS - 1, 0);
		inData = inData >> COL_SUM_BITS;
		ap_uint<1> cond = (currentData < tmp);
		tmpIdx = (cond) ? i : tmpIdx;
		tmp = (cond) ? currentData : tmp;
	}
	*index = tmpIdx;
	return tmp;
}

pix_t readPixFromCol(col_pix_t colData, ap_uint<8> idx)
{
#pragma HLS INLINE
	pix_t retData;
	// Use bit selection plus for-loop to read multi-bits from a wider bit width value
	// rather than use range selection directly. The reason is that the latter will use
	// a lot of shift-register which will increase a lot of LUTs consumed.
	readWiderBitsLoop: for(int8_t yIndex = 0; yIndex < BITS_PER_PIXEL; yIndex++)
	{
#pragma HLS UNROLL
		const int bitOffset = BITS_PER_PIXEL >> 1;
		ap_uint<8 + bitOffset> colIdx;
		// Concatenate and bit shift rather than multiple and accumulation (MAC) can save area.
		colIdx.range(8 + bitOffset - 1, bitOffset) = ap_uint<10>(idx * BITS_PER_PIXEL).range(8 + bitOffset - 1, bitOffset);
		colIdx.range(bitOffset - 1, 0) = ap_uint<2>(yIndex);

		retData[yIndex] = colData[colIdx];
//		retData[yIndex] = colData[BITS_PER_PIXEL*idx + yIndex];
	}
	return retData;
}

pix_t readPixFromTwoCols(two_cols_pix_t colData, ap_uint<8> idx)
{
#pragma HLS INLINE
	pix_t retData;
	// Use bit selection plus for-loop to read multi-bits from a wider bit width value
	// rather than use range selection directly. The reason is that the latter will use
	// a lot of shift-register which will increase a lot of LUTs consumed.
//	ap_uint<256> colIdxHi, colIdxLo;
//	colIdxHi = (ap_uint<8>(idx * BITS_PER_PIXEL)(8,2), ap_uint<2>(0));
//	colIdxLo = (ap_uint<8>(idx * BITS_PER_PIXEL)(8,2), ap_uint<2>(BITS_PER_PIXEL - 1));
//	retData = colData(colIdxHi, colIdxLo);
	readTwoColsWiderBitsLoop: for(int8_t yIndex = 0; yIndex < BITS_PER_PIXEL; yIndex++)
	{
#pragma HLS UNROLL
		const int bitOffset = BITS_PER_PIXEL >> 1;
		ap_uint<8 + bitOffset> colIdx;
		// Concatenate and bit shift rather than multiple and accumulation (MAC) can save area.
		colIdx.range(8 + bitOffset - 1, bitOffset) = ap_uint<10>(idx * BITS_PER_PIXEL).range(8 + bitOffset - 1, bitOffset);
		colIdx.range(bitOffset - 1, 0) = ap_uint<2>(yIndex);

		retData[yIndex] = colData[colIdx];
//		retData[yIndex] = colData[BITS_PER_PIXEL*idx + yIndex];
	}
	return retData;
}

void writePixToCol(col_pix_t *colData, ap_uint<8> idx, pix_t pixData)
{
#pragma HLS INLINE
	writeWiderBitsLoop: for(int8_t yIndex = 0; yIndex < BITS_PER_PIXEL; yIndex++)
	{
#pragma HLS UNROLL
		const int bitOffset = BITS_PER_PIXEL >> 1;
		ap_uint<8 + bitOffset> colIdx;
		// Concatenate and bit shift rather than multiple and accumulation (MAC) can save area.
		colIdx.range(8 + bitOffset - 1, bitOffset) = ap_uint<10>(idx * BITS_PER_PIXEL).range(8 + bitOffset - 1, bitOffset);
		colIdx.range(bitOffset - 1, 0) = ap_uint<2>(yIndex);

		(*colData)[colIdx] = pixData[yIndex];
	}
}

void resetPix(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS INLINE
	glPLSlices[sliceIdx][x][y/COMBINED_PIXELS] = 0;
	glPLSlicesScale1[sliceIdx][x/2][y/COMBINED_PIXELS/2] = 0;
	glPLSlicesScale2[sliceIdx][x/4][y/COMBINED_PIXELS/4] = 0;
}

void resetPixScale0(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS INLINE
	glPLSlices[sliceIdx][x][y/COMBINED_PIXELS] = 0;
}

void resetPixScale1(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS INLINE
	glPLSlicesScale1[sliceIdx][x][y/COMBINED_PIXELS] = 0;
}

void resetPixScale2(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS INLINE
	glPLSlicesScale2[sliceIdx][x][y/COMBINED_PIXELS] = 0;
}

void writePix(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS DEPENDENCE variable=glPLSlicesScale2 inter false
#pragma HLS DEPENDENCE variable=glPLSlicesScale1 inter false
#pragma HLS RESOURCE variable=glPLSlicesScale2 core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale2 cyclic factor=1 dim=3
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale2 complete dim=1
#pragma HLS RESOURCE variable=glPLSlicesScale1 core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale1 cyclic factor=1 dim=3
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale1 complete dim=1
#pragma HLS PIPELINE
#pragma HLS RESOURCE variable=glPLSlices core=RAM_T2P_BRAM
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=glPLSlices complete dim=1
#pragma HLS ARRAY_PARTITION variable=glPLSlices cyclic factor=1 dim=3
#pragma HLS DEPENDENCE variable=glPLSlices inter false
	col_pix_t tmpData;
	pix_t tmpTmpData;

	ap_uint<8> yNewIdx = y%COMBINED_PIXELS;

	tmpData = glPLSlices[sliceIdx][x][y/COMBINED_PIXELS];

	tmpTmpData = readPixFromCol(tmpData, yNewIdx);

	ap_uint<1> cmpFlg = ap_uint<1>(tmpTmpData < (ap_uint< BITS_PER_PIXEL >(0xff)));
	tmpTmpData +=  cmpFlg;

	writePixToCol(&tmpData, yNewIdx, tmpTmpData);

	glPLSlices[sliceIdx][x][y/COMBINED_PIXELS] = tmpData;

    // write scale 1
	apUint10_t xScale1 = x/2;
	apUint10_t yScale1 = y/2;
    ap_uint<8> yNewIdxScale1 = yScale1%COMBINED_PIXELS;

	col_pix_t tmpDataScale1;
	pix_t tmpTmpDataScale1;

	tmpDataScale1 = glPLSlicesScale1[sliceIdx][xScale1][yScale1/COMBINED_PIXELS];
	tmpTmpDataScale1 = readPixFromCol(tmpDataScale1, yNewIdxScale1);
	ap_uint<1> cmpFlgScale1 = ap_uint<1>(tmpTmpDataScale1 < (ap_uint< BITS_PER_PIXEL >(0xff)));
	tmpTmpDataScale1 +=  cmpFlgScale1;
	writePixToCol(&tmpDataScale1, yNewIdxScale1, tmpTmpDataScale1);
    glPLSlicesScale1[sliceIdx][xScale1][yScale1/COMBINED_PIXELS] = tmpDataScale1;

    // write scale 2
    // write scale 1
    apUint10_t xScale2 = x/4;
    apUint10_t yScale2 = y/4;
    ap_uint<8> yNewIdxScale2 = yScale2%COMBINED_PIXELS;

	col_pix_t tmpDataScale2;
	pix_t tmpTmpDataScale2;

	tmpDataScale2 = glPLSlicesScale2[sliceIdx][xScale2][yScale2/COMBINED_PIXELS];
	tmpTmpDataScale2 = readPixFromCol(tmpDataScale2, yNewIdxScale2);
	ap_uint<1> cmpFlgScale2 = ap_uint<1>(tmpTmpDataScale2 < (ap_uint< BITS_PER_PIXEL >(0xff)));
	tmpTmpDataScale2 +=  cmpFlgScale2;
	writePixToCol(&tmpDataScale2, yNewIdxScale2, tmpTmpDataScale2);
    glPLSlicesScale2[sliceIdx][xScale2][yScale2/COMBINED_PIXELS] = tmpDataScale2;
}

void writePixScale0(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS PIPELINE
#pragma HLS RESOURCE variable=glPLSlices core=RAM_T2P_BRAM
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=glPLSlices complete dim=1
#pragma HLS ARRAY_PARTITION variable=glPLSlices cyclic factor=1 dim=3
#pragma HLS DEPENDENCE variable=glPLSlices inter false
	col_pix_t tmpData;
	pix_t tmpTmpData;

	ap_uint<8> yNewIdx = y%COMBINED_PIXELS;

	tmpData = glPLSlices[sliceIdx][x][y/COMBINED_PIXELS];

	tmpTmpData = readPixFromCol(tmpData, yNewIdx);

	ap_uint<1> cmpFlg = ap_uint<1>(tmpTmpData < (ap_uint< BITS_PER_PIXEL >(0xff)));
	tmpTmpData +=  cmpFlg;

	writePixToCol(&tmpData, yNewIdx, tmpTmpData);

	glPLSlices[sliceIdx][x][y/COMBINED_PIXELS] = tmpData;
}

void writePixScale1(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS DEPENDENCE variable=glPLSlicesScale1 inter false
#pragma HLS RESOURCE variable=glPLSlicesScale1 core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale1 cyclic factor=1 dim=3
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale1 complete dim=1
#pragma HLS PIPELINE
#pragma HLS INLINE
    // write scale 1
	apUint10_t xScale1 = x/2;
	apUint10_t yScale1 = y/2;
    ap_uint<8> yNewIdxScale1 = yScale1%COMBINED_PIXELS;

	col_pix_t tmpDataScale1;
	pix_t tmpTmpDataScale1;

	tmpDataScale1 = glPLSlicesScale1[sliceIdx][xScale1][yScale1/COMBINED_PIXELS];
	tmpTmpDataScale1 = readPixFromCol(tmpDataScale1, yNewIdxScale1);
	ap_uint<1> cmpFlgScale1 = ap_uint<1>(tmpTmpDataScale1 < (ap_uint< BITS_PER_PIXEL >(0xff)));
	tmpTmpDataScale1 +=  cmpFlgScale1;
	writePixToCol(&tmpDataScale1, yNewIdxScale1, tmpTmpDataScale1);
    glPLSlicesScale1[sliceIdx][xScale1][yScale1/COMBINED_PIXELS] = tmpDataScale1;
}

void writePixScale2(apUint10_t x, apUint10_t y, sliceIdx_t sliceIdx)
{
#pragma HLS DEPENDENCE variable=glPLSlicesScale2 inter false
#pragma HLS RESOURCE variable=glPLSlicesScale2 core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale2 cyclic factor=1 dim=3
#pragma HLS ARRAY_PARTITION variable=glPLSlicesScale2 complete dim=1
#pragma HLS PIPELINE
#pragma HLS INLINE
    // write scale 2
    apUint10_t xScale2 = x/4;
    apUint10_t yScale2 = y/4;
    ap_uint<8> yNewIdxScale2 = yScale2%COMBINED_PIXELS;

	col_pix_t tmpDataScale2;
	pix_t tmpTmpDataScale2;

	tmpDataScale2 = glPLSlicesScale2[sliceIdx][xScale2][yScale2/COMBINED_PIXELS];
	tmpTmpDataScale2 = readPixFromCol(tmpDataScale2, yNewIdxScale2);
	ap_uint<1> cmpFlgScale2 = ap_uint<1>(tmpTmpDataScale2 < (ap_uint< BITS_PER_PIXEL >(0xff)));
	tmpTmpDataScale2 +=  cmpFlgScale2;
	writePixToCol(&tmpDataScale2, yNewIdxScale2, tmpTmpDataScale2);
    glPLSlicesScale2[sliceIdx][xScale2][yScale2/COMBINED_PIXELS] = tmpDataScale2;
}


// for scale 0
void readBlockCols(apUint10_t x, apUint10_t y, ap_int<8> xInitOffset, ap_int<8> yInitOffset,
		sliceIdx_t sliceIdxRef, sliceIdx_t sliceIdxTag,
		pix_t refCol[BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE],
		pix_t tagCol[BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE])
{
#pragma HLS INLINE
#pragma HLS PIPELINE
#pragma HLS ARRAY_RESHAPE variable=refCol complete dim=1
#pragma HLS ARRAY_RESHAPE variable=tagCol complete dim=1

	two_cols_pix_t refColData;
    two_cols_pix_t tagColData;
    ap_uint<10> xWithInitOffset = x + xInitOffset;
    ap_uint<10> yWithInitOffset = y + yInitOffset;

    ap_uint<8> neighboryOffset;
    if ( y%COMBINED_PIXELS < BLOCK_SIZE_SCALE_0/2 + SEARCH_DISTANCE )
    {
        neighboryOffset = y/COMBINED_PIXELS - 1;
    }
    else
    {
        neighboryOffset = y/COMBINED_PIXELS + 1;
    }

    ap_uint<8> neighboryOffsetWithInitOffset;
    if ( yWithInitOffset%COMBINED_PIXELS < BLOCK_SIZE_SCALE_0/2 + SEARCH_DISTANCE )
    {
    	neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS - 1;
    }
    else
    {
    	neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 1;
    }

    // concatenate two columns together, permanent offset for ref colomun is SEARCH_DISTANCE
    refColData = (glPLSlices[sliceIdxRef][x + SEARCH_DISTANCE][y/COMBINED_PIXELS], glPLSlices[sliceIdxRef][x + SEARCH_DISTANCE][neighboryOffset]);
    //	cout << "refColData: " << refColData << endl;
    // concatenate two columns together
    // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
    tagColData = (glPLSlices[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], glPLSlices[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);

	// find the bottom pixel of the column that centered on y.
	ap_uint<6> yColOffsetIdx = y%COMBINED_PIXELS - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoop: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; i++)
	{
		refCol[i] = readPixFromTwoCols(refColData,  yColOffsetIdx);
		yColOffsetIdx++;
	}

	// find the bottom pixel of the column that centered on yInitOffset.
	ap_uint<6> yColOffsetWithInitOffsetIdx = yWithInitOffset%COMBINED_PIXELS - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoopWithInitOffset: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; i++)
	{
		tagCol[i] = readPixFromTwoCols(tagColData,  yColOffsetWithInitOffsetIdx);
		yColOffsetWithInitOffsetIdx++;
	}
}


void readBlockColsScale1(apUint10_t x, apUint10_t y, ap_int<8> xInitOffset, ap_int<8> yInitOffset,
		sliceIdx_t sliceIdxRef, sliceIdx_t sliceIdxTag,
		pix_t refColScale1[BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE],
		pix_t tagColScale1[BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE])
{
#pragma HLS ARRAY_RESHAPE variable=tagColScale1 complete dim=1
#pragma HLS ARRAY_RESHAPE variable=refColScale1 complete dim=1
#pragma HLS PIPELINE
#pragma HLS INLINE

	two_cols_pix_t refColData;
    two_cols_pix_t tagColData;
    ap_uint<10> xWithInitOffset = x + xInitOffset;
    ap_uint<10> yWithInitOffset = y + yInitOffset;

    ap_uint<8> neighboryOffset;
    if ( y%COMBINED_PIXELS < BLOCK_SIZE_SCALE_1/2 + SEARCH_DISTANCE )
    {
        neighboryOffset = y/COMBINED_PIXELS - 1;
    }
    else
    {
        neighboryOffset = y/COMBINED_PIXELS + 1;
    }

    ap_uint<8> neighboryOffsetWithInitOffset;
    if ( yWithInitOffset%COMBINED_PIXELS < BLOCK_SIZE_SCALE_1/2 + SEARCH_DISTANCE )
    {
    	neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS - 1;
    }
    else
    {
    	neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 1;
    }

    // concatenate two columns together, permanent offset for ref colomun is SEARCH_DISTANCE
    refColData = (glPLSlicesScale1[sliceIdxRef][x + SEARCH_DISTANCE][y/COMBINED_PIXELS], glPLSlicesScale1[sliceIdxRef][x + SEARCH_DISTANCE][neighboryOffset]);
    //	cout << "refColData: " << refColData << endl;
    // concatenate two columns together
    // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
    tagColData = (glPLSlicesScale1[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], glPLSlicesScale1[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);

	// find the bottom pixel of the column that centered on y.
	ap_uint<6> yColOffsetIdx = y%COMBINED_PIXELS - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoop: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; i++)
	{
		refColScale1[i] = readPixFromTwoCols(refColData,  yColOffsetIdx);
		yColOffsetIdx++;
	}

	// find the bottom pixel of the column that centered on yInitOffset.
	ap_uint<6> yColOffsetWithInitOffsetIdx = yWithInitOffset%COMBINED_PIXELS - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoopWithInitOffset: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; i++)
	{
		tagColScale1[i] = readPixFromTwoCols(tagColData,  yColOffsetWithInitOffsetIdx);
		yColOffsetWithInitOffsetIdx++;
	}
}

void readBlockColsScale2(apUint10_t x, apUint10_t y, ap_int<8> xInitOffset, ap_int<8> yInitOffset,
		sliceIdx_t sliceIdxRef, sliceIdx_t sliceIdxTag,
		pix_t refColScale2[BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE],
		pix_t tagColScale2[BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE])
{
#pragma HLS ARRAY_RESHAPE variable=tagColScale2 complete dim=1
#pragma HLS ARRAY_RESHAPE variable=refColScale2 complete dim=1
#pragma HLS PIPELINE
#pragma HLS INLINE

	two_cols_pix_t refColData;
    two_cols_pix_t tagColData;
    ap_uint<10> xWithInitOffset = x + xInitOffset;
    ap_uint<10> yWithInitOffset = y + yInitOffset;

    ap_uint<8> neighboryOffset;
    if ( y%COMBINED_PIXELS < BLOCK_SIZE_SCALE_2/2 + SEARCH_DISTANCE )
    {
        neighboryOffset = y/COMBINED_PIXELS - 1;
    }
    else
    {
        neighboryOffset = y/COMBINED_PIXELS + 1;
    }

    ap_uint<8> neighboryOffsetWithInitOffset;
    if ( yWithInitOffset%COMBINED_PIXELS < BLOCK_SIZE_SCALE_2/2 + SEARCH_DISTANCE )
    {
    	neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS - 1;
    }
    else
    {
    	neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 1;
    }
    // concatenate two columns together
    refColData = (glPLSlicesScale2[sliceIdxRef][x][y/COMBINED_PIXELS], glPLSlicesScale2[sliceIdxRef][x][neighboryOffset]);
    //	cout << "refColData: " << refColData << endl;
    // concatenate two columns together
    // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
    tagColData = (glPLSlicesScale2[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], glPLSlicesScale2[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);

	// find the bottom pixel of the column that centered on y.
	ap_uint<6> yColOffsetIdx = y%COMBINED_PIXELS - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoop: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; i++)
	{
		refColScale2[i] = readPixFromTwoCols(refColData,  yColOffsetIdx);
		yColOffsetIdx++;
	}

	// find the bottom pixel of the column that centered on yInitOffset.
	ap_uint<6> yColOffsetWithInitOffsetIdx = yWithInitOffset%COMBINED_PIXELS - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoopWithInitOffset: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; i++)
	{
		tagColScale2[i] = readPixFromTwoCols(tagColData,  yColOffsetWithInitOffsetIdx);
		yColOffsetWithInitOffsetIdx++;
	}
}

void getXandY(const uint64_t * data, hls::stream<apUint10_t>  &xStream, hls::stream<apUint10_t> &yStream, hls::stream<uint32_t> &tsStream, hls::stream<apUint17_t> &packetEventDataStream)
//void getXandY(const uint64_t * data, int32_t eventsArraySize, ap_uint<8> *xStream, ap_uint<8> *yStream)
{
#pragma HLS INLINE off
#pragma HLS INLINE off

	// Every event always consists of 2 int32_t which is 8bytes.
//	getXandYLoop:for(int32_t i = 0; i < eventIterSize; i++)
//	{
		uint64_t tmp = *data;
		apUint10_t xWr, yWr;
		xWr = ((tmp) >> POLARITY_X_ADDR_SHIFT) & POLARITY_X_ADDR_MASK;
		yWr = ((tmp) >> POLARITY_Y_ADDR_SHIFT) & POLARITY_Y_ADDR_MASK;
		bool pol  = ((tmp) >> POLARITY_SHIFT) & POLARITY_MASK;
		uint32_t ts = tmp >> 32;

//		writePix(xWr, yWr, glPLActiveSliceIdx);
//		resetPix(xWr, yWr, glPLActiveSliceIdx + 3);

//		shiftCnt = 0;
//		miniRetVal = 0x7fff;
//		for(int8_t i = 0; i <= 2*SEARCH_DISTANCE; i++)
//		{
//				miniSumTmp[i] = 0;
//		}
//		for(int8_t i = 0; i <= 2*SEARCH_DISTANCE; i++)
//		{
//			for(int8_t j = 0; j <= 2*SEARCH_DISTANCE; j++)
//			{
//				localSumReg[i][j] = 0;
//			}
//		}

		xStream << xWr;
		yStream << yWr;
		tsStream << ts;
		packetEventDataStream << apUint17_t(xWr.to_int() + (yWr.to_int() << 8) + (pol << 16));
//		*xStream++ = xWr;
//		*yStream++ = yWr;
//	}
}

