#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
using namespace std;
#include "ap_fixed.h"
#include "dndAccel.h"
#include "time.h"
#include<stdio.h>

#define TEST_TIMES 20

static col_pix_t slicesSW[SLICES_NUMBER][SLICE_WIDTH][SLICE_HEIGHT/COMBINED_PIXELS];
static col_pix_t slicesScale1SW[SLICES_NUMBER][SLICE_WIDTH/2][SLICE_HEIGHT/COMBINED_PIXELS/2];
static col_pix_t slicesScale2SW[SLICES_NUMBER][SLICE_WIDTH/4][SLICE_HEIGHT/COMBINED_PIXELS/4];
static sliceIdx_t glPLActiveSliceIdxSW = 0;

void resetPixSW(ap_uint<10> x, ap_uint<10> y, sliceIdx_t sliceIdx)
{
	slicesSW[sliceIdx][x][y/COMBINED_PIXELS] = 0;
	slicesScale1SW[sliceIdx][x/2][y/COMBINED_PIXELS/2] = 0;
	slicesScale2SW[sliceIdx][x/4][y/COMBINED_PIXELS/4] = 0;
}

void writePixSW(ap_uint<10> x, ap_uint<10> y, sliceIdx_t sliceIdx)
{
    // write scale 0
	int8_t yNewIdx = y%COMBINED_PIXELS;
//	cout << "Data before write : " << slicesSW[sliceIdx][x][y/COMBINED_PIXELS].range(4 * yNewIdx + 3, 4 * yNewIdx) << endl;
	pix_t tmp = slicesSW[sliceIdx][x][y/COMBINED_PIXELS].range(4 * yNewIdx + 3, 4 * yNewIdx);
    if (tmp >= (ap_uint< BITS_PER_PIXEL >(0xff))) tmp = (ap_uint< BITS_PER_PIXEL >(0xff));
    else tmp += 1;
	slicesSW[sliceIdx][x][y/COMBINED_PIXELS].range(4 * yNewIdx + 3, 4 * yNewIdx) = tmp;
//	cout << "Data after write : " << slicesSW[sliceIdx][x][y/COMBINED_PIXELS].range(4 * yNewIdx + 3, 4 * yNewIdx) << endl;

    // write scale 1
    ap_uint<8> xScale1 = x/2;
    ap_uint<8> yScale1 = y/2;
	int8_t yNewIdxScale1 = yScale1%COMBINED_PIXELS;
//	cout << "Data before write : " << slicesScale1SW[sliceIdx][xScale1][yScale1/COMBINED_PIXELS].range(4 * yNewIdxScale1 + 3, 4 * yNewIdxScale1) << endl;
	pix_t tmpScale1 = slicesScale1SW[sliceIdx][xScale1][yScale1/COMBINED_PIXELS].range(4 * yNewIdxScale1 + 3, 4 * yNewIdxScale1);
    if (tmpScale1 >= (ap_uint< BITS_PER_PIXEL >(0xff))) tmpScale1 = (ap_uint< BITS_PER_PIXEL >(0xff));
    else tmpScale1 += 1;
	slicesScale1SW[sliceIdx][xScale1][yScale1/COMBINED_PIXELS].range(4 * yNewIdxScale1 + 3, 4 * yNewIdxScale1) = tmpScale1;
//	cout << "Data after write : " << slicesScale1SW[sliceIdx][xScale1][yScale1/COMBINED_PIXELS].range(4 * yNewIdxScale1 + 3, 4 * yNewIdxScale1) << endl;

    // write scale 2
    ap_uint<8> xScale2 = x/4;
    ap_uint<8> yScale2 = y/4;
	int8_t yNewIdxScale2 = yScale2%COMBINED_PIXELS;
//	cout << "Data before write : " << slicesScale2SW[sliceIdx][xScale2][yScale2/COMBINED_PIXELS].range(4 * yNewIdxScale2 + 3, 4 * yNewIdxScale2) << endl;
	pix_t tmpScale2 = slicesScale2SW[sliceIdx][xScale2][yScale2/COMBINED_PIXELS].range(4 * yNewIdxScale2 + 3, 4 * yNewIdxScale2);
    if (tmpScale2 >= (ap_uint< BITS_PER_PIXEL >(0xff))) tmpScale2 = (ap_uint< BITS_PER_PIXEL >(0xff));
    else tmpScale2 += 1;
	slicesScale2SW[sliceIdx][xScale2][yScale2/COMBINED_PIXELS].range(4 * yNewIdxScale2 + 3, 4 * yNewIdxScale2) = tmpScale2;
//	cout << "Data after write : " << slicesScale2SW[sliceIdx][xScale2][yScale2/COMBINED_PIXELS].range(4 * yNewIdxScale2 + 3, 4 * yNewIdxScale2) << endl;
}

void readBlockColsSWScale0(ap_uint<10> x, ap_uint<10> y, ap_int<8> xInitOffset, ap_int<8> yInitOffset,
		sliceIdx_t sliceIdxRef, sliceIdx_t sliceIdxTag,
		pix_t refCol[BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE], pix_t tagCol[BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE])
{

	two_cols_pix_t refColData;
    two_cols_pix_t tagColData;
    ap_uint<8> neighboryOffset;
    ap_uint<8> neighboryOffsetWithInitOffset;
    ap_uint<10> xWithInitOffset = x + xInitOffset;
    ap_uint<10> yWithInitOffset = y + yInitOffset;

    if ( y%COMBINED_PIXELS < BLOCK_SIZE_SCALE_0/2 + SEARCH_DISTANCE )
    {
        neighboryOffset = y/COMBINED_PIXELS - 1;
        // concatenate two columns together
        refColData = (slicesSW[sliceIdxRef][x][y/COMBINED_PIXELS], slicesSW[sliceIdxRef][x][neighboryOffset]); 
    }
    else if ( y%COMBINED_PIXELS >  COMBINED_PIXELS - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE - 1 )
    {
        neighboryOffset = y/COMBINED_PIXELS + 1;
        // concatenate two columns together
        refColData = (slicesSW[sliceIdxRef][x][y/COMBINED_PIXELS], slicesSW[sliceIdxRef][x][neighboryOffset]); 
   }
    else
    {
        neighboryOffset = y/COMBINED_PIXELS + 0;
        // concatenate two columns together
        refColData = (slicesSW[sliceIdxRef][x][y/COMBINED_PIXELS], slicesSW[sliceIdxRef][x][neighboryOffset]); 
    }

    if ( yWithInitOffset%COMBINED_PIXELS < BLOCK_SIZE_SCALE_0/2 + SEARCH_DISTANCE )
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS - 1;
        // concatenate two columns together
        //	cout << "refColData: " << refColData << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColData = (slicesSW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesSW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }
    else if ( yWithInitOffset%COMBINED_PIXELS >  COMBINED_PIXELS - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE - 1 )
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 1;
        // concatenate two columns together
        //	cout << "refColData: " << refColData << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColData = (slicesSW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesSW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }
    else
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 0;
        // concatenate two columns together
        //	cout << "refColData: " << refColData << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColData = (slicesSW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesSW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }

	// find the bottom pixel of the column that centered on y.
	ap_uint<6> yColOffsetIdx = y%COMBINED_PIXELS - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE + COMBINED_PIXELS;

	readRefLoop: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; i++)
	{
		refCol[i] = refColData.range(yColOffsetIdx * 4 + 3, yColOffsetIdx * 4);
		yColOffsetIdx++;
	}

	// find the bottom pixel of the column that centered on yInitOffset.
	ap_uint<6> yColOffsetWithInitOffsetIdx = yWithInitOffset%COMBINED_PIXELS - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoopWithInitOffset: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; i++)
	{
		tagCol[i] = tagColData.range(yColOffsetWithInitOffsetIdx * 4 + 3, yColOffsetWithInitOffsetIdx * 4);
		yColOffsetWithInitOffsetIdx++;
	}
}

void readBlockColsSWScale1(ap_uint<10> x, ap_uint<10> y, ap_int<8> xInitOffset, ap_int<8> yInitOffset,
		sliceIdx_t sliceIdxRef, sliceIdx_t sliceIdxTag,
		pix_t refColScale1[BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE], pix_t tagColScale1[BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE])
{
	two_cols_pix_t refColDataScale1;
    two_cols_pix_t tagColDataScale1;
    ap_uint<10> xScale1 = x;
    ap_uint<10> yScale1 = y;
    ap_uint<8> neighboryOffsetScale1;
    ap_uint<8> neighboryOffsetWithInitOffset;
    ap_uint<10> xWithInitOffset = x + xInitOffset;
    ap_uint<10> yWithInitOffset = y + yInitOffset;

    if ( yScale1%COMBINED_PIXELS < BLOCK_SIZE_SCALE_1/2 + SEARCH_DISTANCE )
    {
        neighboryOffsetScale1 = yScale1/COMBINED_PIXELS - 1;
        // concatenate two columns together
        refColDataScale1 = (slicesScale1SW[sliceIdxRef][xScale1][yScale1/COMBINED_PIXELS], slicesScale1SW[sliceIdxRef][xScale1][neighboryOffsetScale1]); 
    }
    else if ( yScale1%COMBINED_PIXELS >  COMBINED_PIXELS - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE - 1 )
    {
        neighboryOffsetScale1 = yScale1/COMBINED_PIXELS + 1;
        // concatenate two columns together
        refColDataScale1 = (slicesScale1SW[sliceIdxRef][xScale1][yScale1/COMBINED_PIXELS], slicesScale1SW[sliceIdxRef][xScale1][neighboryOffsetScale1]); 
    }
    else
    {
        neighboryOffsetScale1 = yScale1/COMBINED_PIXELS + 0;
        refColDataScale1 = (slicesScale1SW[sliceIdxRef][xScale1][yScale1/COMBINED_PIXELS], slicesScale1SW[sliceIdxRef][xScale1][neighboryOffsetScale1]); 
    }

    if ( yWithInitOffset%COMBINED_PIXELS < BLOCK_SIZE_SCALE_1/2 + SEARCH_DISTANCE )
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS - 1;
        //	cout << "refColDataScale1: " << refColDataScale1 << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColDataScale1 = (slicesScale1SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesScale1SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }
    else if ( yWithInitOffset%COMBINED_PIXELS >  COMBINED_PIXELS - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE - 1 )
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 1;
        //	cout << "refColDataScale1: " << refColDataScale1 << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColDataScale1 = (slicesScale1SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesScale1SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }
    else
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 0;
        //	cout << "refColDataScale1: " << refColDataScale1 << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColDataScale1 = (slicesScale1SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesScale1SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }

	// find the bottom pixel of the column that centered on y.
	ap_uint<6> yColOffsetIdxScale1 = yScale1%COMBINED_PIXELS - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE + COMBINED_PIXELS;

	readRefLoop: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; i++)
	{
		refColScale1[i] = refColDataScale1.range(yColOffsetIdxScale1 * 4 + 3, yColOffsetIdxScale1 * 4);
		yColOffsetIdxScale1++;
	}

	// find the bottom pixel of the column that centered on yInitOffset.
	ap_uint<6> yColOffsetWithInitOffsetIdx = yWithInitOffset%COMBINED_PIXELS - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoopWithInitOffset: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; i++)
	{
		tagColScale1[i] = tagColDataScale1.range(yColOffsetWithInitOffsetIdx * 4 + 3, yColOffsetWithInitOffsetIdx * 4);
		yColOffsetWithInitOffsetIdx++;
	}
}

void readBlockColsSWScale2(ap_uint<10> x, ap_uint<10> y, ap_int<8> xInitOffset, ap_int<8> yInitOffset,
		sliceIdx_t sliceIdxRef, sliceIdx_t sliceIdxTag,
		pix_t refColScale2[BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE], pix_t tagColScale2[BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE])
{
	two_cols_pix_t refColDataScale2;
    two_cols_pix_t tagColDataScale2;
    ap_uint<10> xScale2 = x;
    ap_uint<10> yScale2 = y;
    ap_uint<8> neighboryOffsetScale2;
    ap_uint<8> neighboryOffsetWithInitOffset;
    ap_uint<10> xWithInitOffset = x + xInitOffset;
    ap_uint<10> yWithInitOffset = y + yInitOffset;

    if ( yScale2%COMBINED_PIXELS < BLOCK_SIZE_SCALE_2/2 + SEARCH_DISTANCE )
    {
        neighboryOffsetScale2 = yScale2/COMBINED_PIXELS - 1;
        // concatenate two columns together
        refColDataScale2 = (slicesScale2SW[sliceIdxRef][xScale2][yScale2/COMBINED_PIXELS], slicesScale2SW[sliceIdxRef][xScale2][neighboryOffsetScale2]);
    }
    else if ( yScale2%COMBINED_PIXELS >  COMBINED_PIXELS - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE - 1 )
    {
        neighboryOffsetScale2 = yScale2/COMBINED_PIXELS + 1;
        // concatenate two columns together
        refColDataScale2 = (slicesScale2SW[sliceIdxRef][xScale2][yScale2/COMBINED_PIXELS], slicesScale2SW[sliceIdxRef][xScale2][neighboryOffsetScale2]); 
    }
    else
    {
        neighboryOffsetScale2 = yScale2/COMBINED_PIXELS + 0;
        refColDataScale2 = (slicesScale2SW[sliceIdxRef][xScale2][yScale2/COMBINED_PIXELS], slicesScale2SW[sliceIdxRef][xScale2][neighboryOffsetScale2]); 
    }

    if ( yWithInitOffset%COMBINED_PIXELS < BLOCK_SIZE_SCALE_2/2 + SEARCH_DISTANCE )
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS - 1;
        //	cout << "refColDataScale2: " << refColDataScale2 << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColDataScale2 = (slicesScale2SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesScale2SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }
    else if ( yWithInitOffset%COMBINED_PIXELS >  COMBINED_PIXELS - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE - 1 )
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 1;
        //	cout << "refColDataScale2: " << refColDataScale2 << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColDataScale2 = (slicesScale2SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesScale2SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }
    else
    {
        neighboryOffsetWithInitOffset = yWithInitOffset/COMBINED_PIXELS + 0;
        //	cout << "refColDataScale2: " << refColDataScale2 << endl;
        // concatenate two columns together
        // Use explicit cast here, otherwise it will generate a lot of select operations which consumes more LUTs than MUXs.
        tagColDataScale2 = (slicesScale2SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][yWithInitOffset/COMBINED_PIXELS], slicesScale2SW[(sliceIdx_t)(sliceIdxTag + 0)][xWithInitOffset][neighboryOffsetWithInitOffset]);
    }

	// find the bottom pixel of the column that centered on y.
	ap_uint<6> yColOffsetIdxScale2 = yScale2%COMBINED_PIXELS - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE + COMBINED_PIXELS;

	readRefLoop: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; i++)
	{
		refColScale2[i] = refColDataScale2.range(yColOffsetIdxScale2 * 4 + 3, yColOffsetIdxScale2 * 4);
		yColOffsetIdxScale2++;
	}

	// find the bottom pixel of the column that centered on yInitOffset.
	ap_uint<6> yColOffsetWithInitOffsetIdx = yWithInitOffset%COMBINED_PIXELS - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE + COMBINED_PIXELS;
	readRefLoopWithInitOffset: for(ap_uint<8> i = 0; i < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; i++)
	{
		tagColScale2[i] = tagColDataScale2.range(yColOffsetWithInitOffsetIdx * 4 + 3, yColOffsetWithInitOffsetIdx * 4);
		yColOffsetWithInitOffsetIdx++;
	}
}

void colSADSumSW(pix_t in1[BLOCK_SIZE + 2 * SEARCH_DISTANCE],
		pix_t in2[BLOCK_SIZE + 2 * SEARCH_DISTANCE],
		int16_t out[2 * SEARCH_DISTANCE + 1])
{
//	cout << "in1 is: " << endl;
//	for (int m = 0; m < BLOCK_SIZE + 2 * SEARCH_DISTANCE; m++)
//	{
//		cout << in1[m] << " ";
//	}
//	cout << endl;
//
//	cout << "in2 is: " << endl;
//	for (int m = 0; m < BLOCK_SIZE + 2 * SEARCH_DISTANCE; m++)
//	{
//		cout << in2[m] << " ";
//	}
//	cout << endl;

	for(int i = 0; i <= 2 * SEARCH_DISTANCE; i++)
	{
		int16_t tmpOut = 0;
		for(int j = 0; j < BLOCK_SIZE; j++)
		{
			tmpOut += abs(in1[j + SEARCH_DISTANCE] - in2[i+j]);  // in1 should get the col data centered on current event.
		}
//		cout << "tmpOut is " << tmpOut << endl;
		out[i] = tmpOut;
	}
}

// Set the initial value as the max integer, cannot be 0x7fff, DON'T KNOW WHY.
static ap_int<16> miniRetVal = 0x7fff;
static ap_uint<6> minOFRet = ap_uint<6>(0xff);

static ap_int<16> miniSumTmp[2*SEARCH_DISTANCE + 1] = {0, 0, 0, 0, 0, 0, 0};
static ap_int<16> localSumReg[BLOCK_SIZE][2*SEARCH_DISTANCE + 1];

static int16_t testTmpSum;
void miniSADSumSW(pix_t in1[BLOCK_SIZE + 2 * SEARCH_DISTANCE],
		pix_t in2[BLOCK_SIZE + 2 * SEARCH_DISTANCE],
		int16_t shiftCnt,
		ap_int<16> *miniSumRet,
		ap_uint<6> *OFRet)
{
	int16_t out[2*SEARCH_DISTANCE + 1];

	colSADSumSW(in1, in2, out);

	ap_uint<1> cond1 = (shiftCnt >= BLOCK_SIZE - 1) ? 1 : 0;
	for(int8_t i = 0; i < BLOCK_SIZE - 1; i++)
	{
		shiftInnerLoop: for(int8_t j = 0; j <= 2*SEARCH_DISTANCE; j++)
		{
			localSumReg[i][j] = localSumReg[i + 1][j];
		}
	}

	for(int8_t j = 0; j <= 2*SEARCH_DISTANCE; j++)
	{
		localSumReg[BLOCK_SIZE - 1][j] = out[j];
	}


	ap_uint<3> OFRet_x = minOFRet.range(2, 0);
	ap_uint<3> OFRet_y = minOFRet.range(5, 3);

	for(int8_t i = 0; i <= 2*SEARCH_DISTANCE; i++)
	{
		miniSumTmp[i] = 0;
		for(int8_t j = 0; j <= BLOCK_SIZE - 1; j++)
		{
			miniSumTmp[i] += localSumReg[j][i];
		}
	}

//	cout << "miniSumTmp is: " << endl;
//	for (int m = 0; m <= 2 * SEARCH_DISTANCE; m++)
//	{
//		cout << miniSumTmp[m] << " ";
//	}
//	cout << endl;

	// Find the minimal of current column.
	ap_int<16> miniRetValTmpIter = ap_int<16>(*min_element(miniSumTmp, miniSumTmp + 2*SEARCH_DISTANCE + 1));
	int miniIdx = distance(miniSumTmp, min_element(miniSumTmp, miniSumTmp + 2*SEARCH_DISTANCE + 1));

	// Compare with current global minimum value.
	if (miniRetValTmpIter < miniRetVal)
	{
		if((shiftCnt >= BLOCK_SIZE - 1))
		{
			miniRetVal = miniRetValTmpIter;     // Update the global value
			OFRet_x = ap_uint<3>(shiftCnt - BLOCK_SIZE + 1);   // Record the shift value and store it in OFRet_x
			OFRet_y = ap_uint<3>(miniIdx);
			minOFRet = OFRet_y.concat(OFRet_x);     // Update the OF value.
//			cout << "OF and global minimum updated at index shiftCnt: " << shiftCnt << endl;
		}
	}

//	cout << "OF_x is: " << OFRet_x << "\t OF_y is: " << OFRet_y << endl;

	*miniSumRet = miniRetVal;
	*OFRet = minOFRet;

//	std::cout << "miniSumRetSW is: " << *miniSumRet << "\t OFRetSW is: " << std::hex << *OFRet << std::endl;
//	std::cout << std::dec;    // Restore dec mode
}

void blockSADSW(pix_t blockIn1[BLOCK_SIZE][BLOCK_SIZE], pix_t blockIn2[BLOCK_SIZE][BLOCK_SIZE], uint16_t *sumRet)
{
    uint16_t tmpSum = 0;
    uint16_t validPixRefBlockCnt = 0, validPixTagBlockCnt = 0, nonZeroMatchCnt = 0;
    for(uint8_t i = 0; i < BLOCK_SIZE; i++)
    {
        for(uint8_t j = 0; j < BLOCK_SIZE; j++)
        {
            tmpSum += abs(blockIn1[i][j] - blockIn2[i][j]);

            if (blockIn1[i][j] != 0)
            {
                validPixRefBlockCnt++;
            }
            if (blockIn2[i][j] != 0)
            {
                validPixTagBlockCnt++;
            }
            if (blockIn1[i][j] != 0 && blockIn2[i][j] != 0)
            {
                nonZeroMatchCnt++;
            }
        }
    }

    // Remove outliers
    if (validPixRefBlockCnt < glMinValidPixNum || validPixTagBlockCnt < glMinValidPixNum || nonZeroMatchCnt < glMinValidPixNum)
    {
        tmpSum = 0x7fff;
    }
    *sumRet = tmpSum;
}

void blockSADSWScale0(pix_t blockIn1[BLOCK_SIZE_SCALE_0][BLOCK_SIZE_SCALE_0], pix_t blockIn2[BLOCK_SIZE_SCALE_0][BLOCK_SIZE_SCALE_0], uint16_t *sumRet)
{
    uint16_t tmpSum = 0;
    uint16_t validPixRefBlockCnt = 0, validPixTagBlockCnt = 0, nonZeroMatchCnt = 0;
    for(uint8_t i = 0; i < BLOCK_SIZE_SCALE_0; i++)
    {
        for(uint8_t j = 0; j < BLOCK_SIZE_SCALE_0; j++)
        {
            tmpSum += abs(blockIn1[i][j] - blockIn2[i][j]);

            if (blockIn1[i][j] != 0)
            {
                validPixRefBlockCnt++;
            }
            if (blockIn2[i][j] != 0)
            {
                validPixTagBlockCnt++;
            }
            if (blockIn1[i][j] != 0 && blockIn2[i][j] != 0)
            {
                nonZeroMatchCnt++;
            }
        }
    }

    // Remove outliers
    if (validPixRefBlockCnt < glMinValidPixNumScale0 || validPixTagBlockCnt < glMinValidPixNumScale0 || nonZeroMatchCnt < glMinValidPixNumScale0)
    {
        tmpSum = 0x7fff;
    }
    *sumRet = tmpSum;
}

void blockSADSWScale1(pix_t blockIn1[BLOCK_SIZE_SCALE_1][BLOCK_SIZE_SCALE_1], pix_t blockIn2[BLOCK_SIZE_SCALE_1][BLOCK_SIZE_SCALE_1], uint16_t *sumRet)
{
    uint16_t tmpSum = 0;
    uint16_t validPixRefBlockCnt = 0, validPixTagBlockCnt = 0, nonZeroMatchCnt = 0;
    for(uint8_t i = 0; i < BLOCK_SIZE_SCALE_1; i++)
    {
        for(uint8_t j = 0; j < BLOCK_SIZE_SCALE_1; j++)
        {
            tmpSum += abs(blockIn1[i][j] - blockIn2[i][j]);

            if (blockIn1[i][j] != 0)
            {
                validPixRefBlockCnt++;
            }
            if (blockIn2[i][j] != 0)
            {
                validPixTagBlockCnt++;
            }
            if (blockIn1[i][j] != 0 && blockIn2[i][j] != 0)
            {
                nonZeroMatchCnt++;
            }
        }
    }

    // Remove outliers
    if (validPixRefBlockCnt < glMinValidPixNumScale1 || validPixTagBlockCnt < glMinValidPixNumScale1 || nonZeroMatchCnt < glMinValidPixNumScale1)
    {
        tmpSum = 0x7fff;
    }
    *sumRet = tmpSum;
}

void blockSADSWScale2(pix_t blockIn1[BLOCK_SIZE_SCALE_2][BLOCK_SIZE_SCALE_2], pix_t blockIn2[BLOCK_SIZE_SCALE_2][BLOCK_SIZE_SCALE_2], uint16_t *sumRet)
{
    uint16_t tmpSum = 0;
    uint16_t validPixRefBlockCnt = 0, validPixTagBlockCnt = 0, nonZeroMatchCnt = 0;
    for(uint8_t i = 0; i < BLOCK_SIZE_SCALE_2; i++)
    {
        for(uint8_t j = 0; j < BLOCK_SIZE_SCALE_2; j++)
        {
            tmpSum += abs(blockIn1[i][j] - blockIn2[i][j]);

            if (blockIn1[i][j] != 0)
            {
                validPixRefBlockCnt++;
            }
            if (blockIn2[i][j] != 0)
            {
                validPixTagBlockCnt++;
            }
            if (blockIn1[i][j] != 0 && blockIn2[i][j] != 0)
            {
                nonZeroMatchCnt++;
            }
        }
    }

    // Remove outliers
    if (validPixRefBlockCnt < glMinValidPixNumScale2 || validPixTagBlockCnt < glMinValidPixNumScale2 || nonZeroMatchCnt < glMinValidPixNumScale2)
    {
        tmpSum = 0x7fff;
    }
    *sumRet = tmpSum;
}

void miniBlockSADSW(pix_t refBlock[BLOCK_SIZE][BLOCK_SIZE],
        pix_t tagBlock[BLOCK_SIZE + 2 * SEARCH_DISTANCE][BLOCK_SIZE + 2 * SEARCH_DISTANCE], bool printBlocksEnable,
        ap_int<16> *miniRet, ap_uint<6> *OFRet)
{
    uint16_t tmpSum = 0x7fff;
    ap_uint<3> tmpOF_x = ap_uint<3>(7);
    ap_uint<3> tmpOF_y = ap_uint<3>(7);

    if(printBlocksEnable == true)
    {
        cout << "Reference block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE; blockY++)
            {
                cout << refBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;

        cout << "target block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE + 2 * SEARCH_DISTANCE; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE + 2 * SEARCH_DISTANCE; blockY++)
            {
                cout << tagBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

    for(uint8_t xOffset = 0; xOffset < 2 * SEARCH_DISTANCE + 1; xOffset++)
    {
        for(uint8_t yOffset = 0; yOffset < 2 * SEARCH_DISTANCE + 1; yOffset++)
        {
            pix_t tagBlockIn[BLOCK_SIZE][BLOCK_SIZE];
            uint16_t tmpBlockSum;
            for(uint8_t i = 0; i < BLOCK_SIZE; i++)
            {
                for(uint8_t j = 0; j < BLOCK_SIZE; j++)
                {
                    tagBlockIn[i][j] = tagBlock[i + xOffset][j + yOffset];
                }
            }

            blockSADSW(refBlock, tagBlockIn, &tmpBlockSum);

            if(tmpBlockSum < tmpSum)
            {
                tmpSum = tmpBlockSum;
                tmpOF_x = ap_uint<3>(xOffset);
                tmpOF_y = ap_uint<3>(yOffset);
            }
        }
    }

    if(tmpSum == 0x7fff)
    {
        tmpOF_x = 7;
        tmpOF_y = 7;
    }
    
    *miniRet = tmpSum;
    *OFRet = tmpOF_y.concat(tmpOF_x);
    if(printBlocksEnable == true)
    {
    	std::cout << "miniSumRetSW is: " << *miniRet << "\t OFRetSW is: " << std::hex << *OFRet << std::endl;
    	std::cout << std::dec;    // Restore dec mode
    }

}

void miniBlockSADSWScale0(pix_t refBlock[BLOCK_SIZE_SCALE_0][BLOCK_SIZE_SCALE_0],
        pix_t tagBlock[BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE][BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE], bool printBlocksEnable,
        ap_int<16> *miniRet, ap_uint<6> *OFRet)
{
    uint16_t tmpSum = 0x7fff;
    ap_uint<3> tmpOF_x = ap_uint<3>(7);
    ap_uint<3> tmpOF_y = ap_uint<3>(7);

    if(printBlocksEnable == true)
    {
        cout << "Reference block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE_SCALE_0; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE_SCALE_0; blockY++)
            {
                cout << refBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;

        cout << "target block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; blockY++)
            {
                cout << tagBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

    for(uint8_t xOffset = 0; xOffset < 2 * SEARCH_DISTANCE + 1; xOffset++)
    {
        for(uint8_t yOffset = 0; yOffset < 2 * SEARCH_DISTANCE + 1; yOffset++)
        {
            pix_t tagBlockIn[BLOCK_SIZE_SCALE_0][BLOCK_SIZE_SCALE_0];
            uint16_t tmpBlockSum;
            for(uint8_t i = 0; i < BLOCK_SIZE_SCALE_0; i++)
            {
                for(uint8_t j = 0; j < BLOCK_SIZE_SCALE_0; j++)
                {
                    tagBlockIn[i][j] = tagBlock[i + xOffset][j + yOffset];
                }
            }

            blockSADSWScale0(refBlock, tagBlockIn, &tmpBlockSum);

            if(tmpBlockSum < tmpSum)
            {
                tmpSum = tmpBlockSum;
                tmpOF_x = ap_uint<3>(xOffset);
                tmpOF_y = ap_uint<3>(yOffset);
            }
        }
    }

    if(tmpSum == 0x7fff)
    {
        tmpOF_x = 7;
        tmpOF_y = 7;
    }

    *miniRet = tmpSum;
    *OFRet = tmpOF_y.concat(tmpOF_x);
    if(printBlocksEnable == true)
    {
    	std::cout << "miniSumRetSW is: " << *miniRet << "\t OFRetSW is: " << std::hex << *OFRet << std::endl;
    	std::cout << std::dec;    // Restore dec mode
    }

}

void miniBlockSADSWScale1(pix_t refBlock[BLOCK_SIZE_SCALE_1][BLOCK_SIZE_SCALE_1],
        pix_t tagBlock[BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE][BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE], bool printBlocksEnable,
        ap_int<16> *miniRet, ap_uint<6> *OFRet)
{
    uint16_t tmpSum = 0x7fff;
    ap_uint<3> tmpOF_x = ap_uint<3>(7);
    ap_uint<3> tmpOF_y = ap_uint<3>(7);

    if(printBlocksEnable == true)
    {
        cout << "Reference block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE_SCALE_1; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE_SCALE_1; blockY++)
            {
                cout << refBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;

        cout << "target block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; blockY++)
            {
                cout << tagBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

    for(uint8_t xOffset = 0; xOffset < 2 * SEARCH_DISTANCE + 1; xOffset++)
    {
        for(uint8_t yOffset = 0; yOffset < 2 * SEARCH_DISTANCE + 1; yOffset++)
        {
            pix_t tagBlockIn[BLOCK_SIZE_SCALE_1][BLOCK_SIZE_SCALE_1];
            uint16_t tmpBlockSum;
            for(uint8_t i = 0; i < BLOCK_SIZE_SCALE_1; i++)
            {
                for(uint8_t j = 0; j < BLOCK_SIZE_SCALE_1; j++)
                {
                    tagBlockIn[i][j] = tagBlock[i + xOffset][j + yOffset];
                }
            }

            blockSADSWScale1(refBlock, tagBlockIn, &tmpBlockSum);

            if(tmpBlockSum < tmpSum)
            {
                tmpSum = tmpBlockSum;
                tmpOF_x = ap_uint<3>(xOffset);
                tmpOF_y = ap_uint<3>(yOffset);
            }
        }
    }

    if(tmpSum == 0x7fff)
    {
        tmpOF_x = 7;
        tmpOF_y = 7;
    }

    *miniRet = tmpSum;
    *OFRet = tmpOF_y.concat(tmpOF_x);
    if(printBlocksEnable == true)
    {
    	std::cout << "miniSumRetSW is: " << *miniRet << "\t OFRetSW is: " << std::hex << *OFRet << std::endl;
    	std::cout << std::dec;    // Restore dec mode
    }

}

void miniBlockSADSWScale2(pix_t refBlock[BLOCK_SIZE_SCALE_2][BLOCK_SIZE_SCALE_2],
        pix_t tagBlock[BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE][BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE], bool printBlocksEnable,
        ap_int<16> *miniRet, ap_uint<6> *OFRet)
{
    uint16_t tmpSum = 0x7fff;
    ap_uint<3> tmpOF_x = ap_uint<3>(7);
    ap_uint<3> tmpOF_y = ap_uint<3>(7);

    if(printBlocksEnable == true)
    {
        cout << "Reference block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE_SCALE_2; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE_SCALE_2; blockY++)
            {
                cout << refBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;

        cout << "target block is: " << endl;
        for(uint8_t blockX = 0; blockX < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; blockX++)
        {
            for(uint8_t blockY = 0; blockY < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; blockY++)
            {
                cout << tagBlock[blockX][blockY] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

    for(uint8_t xOffset = 0; xOffset < 2 * SEARCH_DISTANCE + 1; xOffset++)
    {
        for(uint8_t yOffset = 0; yOffset < 2 * SEARCH_DISTANCE + 1; yOffset++)
        {
            pix_t tagBlockIn[BLOCK_SIZE_SCALE_2][BLOCK_SIZE_SCALE_2];
            uint16_t tmpBlockSum;
            for(uint8_t i = 0; i < BLOCK_SIZE_SCALE_2; i++)
            {
                for(uint8_t j = 0; j < BLOCK_SIZE_SCALE_2; j++)
                {
                    tagBlockIn[i][j] = tagBlock[i + xOffset][j + yOffset];
                }
            }

            blockSADSWScale2(refBlock, tagBlockIn, &tmpBlockSum);

            if(tmpBlockSum < tmpSum)
            {
                tmpSum = tmpBlockSum;
                tmpOF_x = ap_uint<3>(xOffset);
                tmpOF_y = ap_uint<3>(yOffset);
            }
        }
    }

    if(tmpSum == 0x7fff)
    {
        tmpOF_x = 7;
        tmpOF_y = 7;
    }

    *miniRet = tmpSum;
    *OFRet = tmpOF_y.concat(tmpOF_x);
    if(printBlocksEnable == true)
    {
    	std::cout << "miniSumRetSW is: " << *miniRet << "\t OFRetSW is: " << std::hex << *OFRet << std::endl;
    	std::cout << std::dec;    // Restore dec mode
    }

}

void testMiniSADSumWrapperSW(apIntBlockCol_t *input1, apIntBlockCol_t *input2, int16_t eventCnt, apUint15_t *miniSum, apUint6_t *OF)
{
	pix_t ref[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tag[BLOCK_SIZE + 2 * SEARCH_DISTANCE];

	apIntBlockCol_t inData1, inData2;
	ap_int<16> miniSumSWRet;
	ap_uint<6> OFRetSWRet;

	for(int32_t i = 0; i < eventCnt; i++)
	{
		// Initialize the localSumReg
		for(int idx1 = 0; idx1 < BLOCK_SIZE; idx1++)
		{
			for(int idx2 = 0; idx2 < BLOCK_SIZE; idx2++)
			{
				localSumReg[idx1][idx2] = 0;
			}
		}
		miniRetVal = 0x7fff;
		minOFRet = ap_uint<6>(0xff);

		for(int shiftOffset = 0; shiftOffset < BLOCK_SIZE + 2 * SEARCH_DISTANCE; shiftOffset++)
		{
			cout << "current iteration index is: " << i * (BLOCK_SIZE + 2 * SEARCH_DISTANCE) + shiftOffset << endl;
			inData1 = *input1++;
			inData2 = *input2++;

			for(int j = 0; j < BLOCK_SIZE + 2 * SEARCH_DISTANCE; j++)
			{
				ref[j] = pix_t(inData1.range(4*j + 3, 4*j));
				tag[j] = pix_t(inData2.range(4*j + 3, 4*j));
			}

			miniSADSumSW(ref, tag, shiftOffset, &miniSumSWRet, &OFRetSWRet);
		}

		std::cout << "miniSumRetSW is: " << apUint15_t(miniSumSWRet) << "\t OFRetSW is: " << std::hex << OFRetSWRet << std::endl;
		cout << dec;

		miniSum[i] = apUint15_t(miniSumSWRet);
		OF[i] = OFRetSWRet;
	}
}

void testSingleRwslicesSW(ap_uint<8> x, ap_uint<8> y, sliceIdx_t idx, pix_t refCol[BLOCK_SIZE + 2 * SEARCH_DISTANCE], pix_t tagCol[BLOCK_SIZE + 2 * SEARCH_DISTANCE])
{
	writePixSW(x, y, idx);
	readBlockColsSWScale0(x, y, 0, 0, idx + 1, idx + 2, refCol, tagCol);
	resetPixSW(x, y, idx + 3);
}

void testRwslicesSW(uint64_t * data, sliceIdx_t idx, int16_t eventCnt, apIntBlockCol_t *refData, apIntBlockCol_t *tagData)
{
	// Check the accumulation slice is clear or not
	for(int32_t xAddr = 0; xAddr < SLICE_WIDTH; xAddr++)
	{
		for(int32_t yAddr = 0; yAddr < SLICE_HEIGHT; yAddr = yAddr + COMBINED_PIXELS)
		{
			if (slicesSW[idx][xAddr][yAddr/COMBINED_PIXELS] != 0)
			{
				cout << "Ha! I caught you, the pixel which is not clear!" << endl;
				cout << "x is: " << xAddr << "\t y is: " << yAddr << "\t idx is: " << idx << endl;
			}
		}
	}

	for(int32_t i = 0; i < eventCnt; i++)
	{
		uint64_t tmp = *data++;
		ap_uint<8> xWr, yWr;
		xWr = ((tmp) >> POLARITY_X_ADDR_SHIFT) & POLARITY_X_ADDR_MASK;
		yWr = ((tmp) >> POLARITY_Y_ADDR_SHIFT) & POLARITY_Y_ADDR_MASK;
		bool pol  = ((tmp) >> POLARITY_SHIFT) & POLARITY_MASK;
		int64_t ts = tmp >> 32;

		writePixSW(xWr, yWr, idx);

		resetPixSW(i/(PIXS_PER_COL), (i % (PIXS_PER_COL)) * COMBINED_PIXELS, (sliceIdx_t)(idx + 3));
//		cout << "tmp is: " << hex << tmp << endl;
//		cout << "x is: " << xWr << "\t y is: " << yWr << "\t idx is: " << idx << endl;

		for(int8_t xOffSet = 0; xOffSet < BLOCK_SIZE + 2 * SEARCH_DISTANCE; xOffSet++)
		{
            pix_t out1[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];
            pix_t out2[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];

            pix_t out1Scale1[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];
            pix_t out2Scale1[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];

            pix_t out1Scale2[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];
            pix_t out2Scale2[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];

//				resetPix(xRd + xOffSet, yRd , (sliceIdx_t)(idx + 3));

//			resetPix(xRd + xOffSet, 1 , (sliceIdx_t)(idx + 3));

			readBlockColsSWScale0(xWr + xOffSet, yWr ,0, 0, idx + 1, idx + 2, out1, out2);


			apIntBlockCol_t refBlockCol;
			apIntBlockCol_t tagBlockCol;

			for (int8_t l = 0; l < BLOCK_SIZE + 2 * SEARCH_DISTANCE; l++)
			{
				refBlockCol.range(BITS_PER_PIXEL * l + BITS_PER_PIXEL - 1, BITS_PER_PIXEL * l) = out1[l];
				tagBlockCol.range(BITS_PER_PIXEL * l + BITS_PER_PIXEL - 1, BITS_PER_PIXEL * l) = out2[l];
			}

			*refData++ = refBlockCol;
			*tagData++ = tagBlockCol;
		}
	}


	for (int16_t resetCnt = 0; resetCnt < 2048; resetCnt = resetCnt + 2)
	{
		resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL) * COMBINED_PIXELS, (sliceIdx_t)(idx + 3));
		resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL + 1) * COMBINED_PIXELS, (sliceIdx_t)(idx + 3));
	}
}

void testTempSW(uint64_t * data, sliceIdx_t idx, int16_t eventCnt, int32_t *eventSlice)
{
	// Check the accumulation slice is clear or not
	for(int32_t xAddr = 0; xAddr < SLICE_WIDTH; xAddr++)
	{
		for(int32_t yAddr = 0; yAddr < SLICE_HEIGHT; yAddr = yAddr + COMBINED_PIXELS)
		{
			if (slicesSW[idx][xAddr][yAddr/COMBINED_PIXELS] != 0)
			{
				for(int r = 0; r < 1000; r++)
				{
					cout << "Ha! I caught you, the pixel which is not clear!" << endl;
					cout << "x is: " << xAddr << "\t y is: " << yAddr << "\t idx is: " << idx << endl;
				}
			}
		}
	}

	ap_int<16> miniSumSWRet;
	ap_uint<6> OFRetSWRet;

	for(int32_t i = 0; i < eventCnt; i++)
	{
		uint64_t tmp = *data++;
		ap_uint<8> xWr, yWr;
		xWr = ((tmp) >> POLARITY_X_ADDR_SHIFT) & POLARITY_X_ADDR_MASK;
		yWr = ((tmp) >> POLARITY_Y_ADDR_SHIFT) & POLARITY_Y_ADDR_MASK;
		bool pol  = ((tmp) >> POLARITY_SHIFT) & POLARITY_MASK;
		int64_t ts = tmp >> 32;

		writePixSW(xWr, yWr, idx);

		resetPixSW(i/(PIXS_PER_COL), (i % (PIXS_PER_COL)) * COMBINED_PIXELS, (sliceIdx_t)(idx + 3));
//		cout << "tmp is: " << hex << tmp << endl;
//		cout << "x is: " << xWr << "\t y is: " << yWr << "\t idx is: " << idx << endl;

		// Initialize the localSumReg
		for(int idx1 = 0; idx1 < BLOCK_SIZE; idx1++)
		{
			for(int idx2 = 0; idx2 < BLOCK_SIZE; idx2++)
			{
				localSumReg[idx1][idx2] = 0;
			}
		}
		miniRetVal = 0x7fff;
		minOFRet = ap_uint<6>(0xff);

		for(int8_t xOffSet = 0; xOffSet < BLOCK_SIZE + 2 * SEARCH_DISTANCE; xOffSet++)
		{
            pix_t out1[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];
            pix_t out2[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];

            pix_t out1Scale1[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];
            pix_t out2Scale1[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];

            pix_t out1Scale2[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];
            pix_t out2Scale2[BLOCK_SIZE+ 2 * SEARCH_DISTANCE];

//				resetPix(xRd + xOffSet, yRd , (sliceIdx_t)(idx + 3));

//			resetPix(xRd + xOffSet, 1 , (sliceIdx_t)(idx + 3));

			readBlockColsSWScale0(xWr + xOffSet, yWr, 0, 0, idx + 1, idx + 2, out1, out2);


			apIntBlockCol_t refBlockCol;
			apIntBlockCol_t tagBlockCol;

			for (int8_t l = 0; l < BLOCK_SIZE + 2 * SEARCH_DISTANCE; l++)
			{
				refBlockCol.range(BITS_PER_PIXEL * l + BITS_PER_PIXEL - 1, BITS_PER_PIXEL * l) = out1[l];
				tagBlockCol.range(BITS_PER_PIXEL * l + BITS_PER_PIXEL - 1, BITS_PER_PIXEL * l) = out2[l];
			}

			miniSADSumSW(out1, out2, xOffSet, &miniSumSWRet, &OFRetSWRet);
			//		testMiniSADSumWrapperSW(refBlockColData, tagBlockColData, eventCnt, miniSumSW, OFRetSW);

			if (refBlockCol != 0 && tagBlockCol == 0)
			{
				cout << "Should not stop here, it is only a debug breakpoint." << endl;
			}
			cout  << "refBlockColSW is: " << hex <<  refBlockCol << endl;
			cout  << "tagBlockColSW is: " << hex <<  tagBlockCol << endl;
//			*refData++ = refBlockCol;
//			*tagData++ = tagBlockCol;
		}

		apUint17_t tmp1 = apUint17_t(xWr.to_int() + (yWr.to_int() << 8) + (1 << 16));
		ap_int<9> tmp2 = miniSumSWRet.range(8, 0);
		apUint6_t tmpOF = OFRetSWRet;
		ap_uint<32> output = (tmp2, (tmpOF, tmp1));
//		std :: cout << "tmp1 is "  << std::hex << tmp1 << std :: endl;
//		std :: cout << "tmp2 is "  << std::hex << tmp2 << std :: endl;
//		std :: cout << "output is "  << std::hex << output << std :: endl;
//		std :: cout << "eventSlice is "  << std::hex << output.to_int() << std :: endl;
		*eventSlice++ = output.to_int();
	}


	for (int16_t resetCnt = 0; resetCnt < 2048; resetCnt = resetCnt + 2)
	{
		resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL) * COMBINED_PIXELS, (sliceIdx_t)(idx + 3));
		resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL + 1) * COMBINED_PIXELS, (sliceIdx_t)(idx + 3));
	}
}

static uint16_t areaEventRegsSW[AREA_NUMBER][AREA_NUMBER];
static uint16_t areaEventThrSW = INIT_AREA_THERSHOLD;
static ap_uint<1> areaCountExceeded = false;
static uint16_t OFRetRegsSW[2 * SEARCH_DISTANCE + 1][2 * SEARCH_DISTANCE + 1];


static void feedbackSW(apUint15_t miniSumRet, apUint6_t OFRet, apUint1_t rotateFlg, uint16_t *thrRet)
{
    if(miniSumRet <= 0x1ff && miniSumRet > 0 && OFRet != 0x3f)
    {
        uint16_t OFRetHistCnt = OFRetRegsSW[OFRet.range(2, 0)][OFRet.range(5, 3)];
        OFRetHistCnt = OFRetHistCnt + 1;
        OFRetRegsSW[OFRet.range(2, 0)][OFRet.range(5, 3)] = OFRetHistCnt;
    }

	if(rotateFlg)
	{
		uint16_t countSum = 0;
		uint16_t histCountSum = 0;
		uint16_t radiusSum =  0;
		uint16_t radiusCountSum =  0;

		for(int8_t OFRetHistX = -SEARCH_DISTANCE; OFRetHistX <= SEARCH_DISTANCE; OFRetHistX++)
		{
			for(int8_t OFRetHistY = -SEARCH_DISTANCE; OFRetHistY <= SEARCH_DISTANCE; OFRetHistY++)
			{
				uint16_t count = OFRetRegsSW[OFRetHistX+SEARCH_DISTANCE][OFRetHistY+SEARCH_DISTANCE];
				float radius = pow(OFRetHistX,  2) + pow(OFRetHistY,  2);
				countSum += count;
				radiusCountSum += radius * count;

				histCountSum += 1;
				radiusSum += radius;

				OFRetRegsSW[OFRetHistX+SEARCH_DISTANCE][OFRetHistY+SEARCH_DISTANCE] = 0;
			}
		}

		if (countSum >= 10)
		{
			float avgMatchDistance = (float)radiusCountSum / countSum;
			float avgTargetDistance = (float)radiusSum / histCountSum;

			if(avgMatchDistance > avgTargetDistance )
			{
//				areaEventThrSW -= areaEventThrSW * 3/64;
				if (areaEventThrSW <= 100)
				{
					areaEventThrSW = 100;
				}
				std::cout << "AreaEventThr is decreased. New areaEventThr from SW is: " << areaEventThrSW << std::endl;
			}
			else if (avgMatchDistance < avgTargetDistance)
			{

//				areaEventThrSW += areaEventThrSW *3/64;
				if (areaEventThrSW >= 1500)
				{
					areaEventThrSW = 1500;
				}
				std::cout << "AreaEventThr is increased. New areaEventThr from SW is: " << areaEventThrSW << std::endl;
			}
		}
	}


    *thrRet = areaEventThrSW;

}


uint32_t currentTs = 0, lastTs = 0;
void parseEventsSW(uint64_t * dataStream, int32_t eventsArraySize, int32_t *eventSlice, ap_uint<32> *custDataOut)
{
//	glPLActiveSliceIdxSW--;
//	sliceIdx_t idx = glPLActiveSliceIdxSW;

//	cout << "Current Event packet's event number is: " << eventsArraySize << endl;
	for(int32_t i = 0; i < eventsArraySize; i++)
	{
		uint64_t tmp = *dataStream++;
		ap_uint<10> xWr, yWr;
		xWr = ((tmp) & POLARITY_X_ADDR_MASK) >> POLARITY_X_ADDR_SHIFT;
		yWr = ((tmp) & POLARITY_Y_ADDR_MASK) >> POLARITY_Y_ADDR_SHIFT;
		bool pol  = ((tmp) & POLARITY_MASK) >> POLARITY_SHIFT;
		uint64_t ts = tmp >> 32;
		ap_uint<1> SFASTCorner = tmp & 1; // LSB

        /* These two values are only for debug and test */
        ap_uint<2> OFGT_scale = (tmp >> 14);
        ap_uint<3> OFGT_x = (tmp >> 26);
        ap_uint<3> OFGT_y = (tmp >> 29);
        ap_uint<6> OFGT = OFGT_y.concat(OFGT_x);

		ap_int<16> miniRet= 0x7fff;
		ap_uint<16> OFRet = 0x7f7f;
		ap_uint<2> scaleRet = 3;
		ap_int<16> miniRetScale0 = 0x7fff;
		ap_uint<6> OFRetScale0 = 0x7f7f;
		ap_int<16> miniRetScale1 = 0x7fff;
		ap_uint<6> OFRetScale1= 0x7f7f;
		ap_int<16> miniRetScale2 = 0x7fff;
		ap_uint<6> OFRetScale2= 0x7f7f;

        apUint1_t rotateFlg = 0;
        // The area threshold reached, rotate the slice index and clear the areaEventRegs.
        if( areaCountExceeded || (ts - currentTs) >= MAX_SLICE_DURATION_US )
        {
            glPLActiveSliceIdxSW--;
//            idx = glPLActiveSliceIdxSW;
            rotateFlg = 1;

            lastTs = currentTs;
            currentTs = ts;

            for(int r = 0; r < 1; r++)
            {
                cout << "Rotated successfully from SW!!!!" << endl;
                cout << "x is: " << xWr << "\t y is: " << yWr << "\t i is: " << i << "\t idx is: " << glPLActiveSliceIdxSW << endl;
                cout << "delataTs is: " << ((currentTs - lastTs) >> 9) << endl;
            }

            // Check the accumulation slice is clear or not
            for(int32_t xAddr = 0; xAddr < SLICE_WIDTH; xAddr++)
            {
                for(int32_t yAddr = 0; yAddr < SLICE_HEIGHT; yAddr = yAddr + COMBINED_PIXELS)
                {
                    if (slicesSW[glPLActiveSliceIdxSW][xAddr][yAddr/COMBINED_PIXELS] != 0)
                    {
                        for(int r = 0; r < 10; r++)
                        {
                            cout << "Ha! I caught you, the pixel which is not clear!" << endl;
                            cout << "x is: " << xAddr << "\t y is: " << yAddr << "\t idx is: " << glPLActiveSliceIdxSW << endl;
                        }
                    }
                }
            }

            for(int areaX = 0; areaX < AREA_NUMBER; areaX++)
            {
                for(int areaY = 0; areaY < AREA_NUMBER; areaY++)
                {
                    areaEventRegsSW[areaX][areaY] = 0;
                }
            }

           for (int16_t resetCnt = 0; resetCnt < SLICE_HEIGHT * SLICE_WIDTH / COMBINED_PIXELS; resetCnt = resetCnt + 2)
           {
               resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL) * COMBINED_PIXELS, (sliceIdx_t)(glPLActiveSliceIdxSW + 3));
               resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL + 1) * COMBINED_PIXELS, (sliceIdx_t)(glPLActiveSliceIdxSW + 3));
           }

        }

    	// update sliceSW
        uint16_t c = areaEventRegsSW[xWr/AREA_SIZE][yWr/AREA_SIZE];
    	c = c + 1;
        areaEventRegsSW[xWr/AREA_SIZE][yWr/AREA_SIZE] = c;
        areaCountExceeded = (c >= areaEventThrSW);

		writePixSW(xWr, yWr, glPLActiveSliceIdxSW);

		resetPixSW(i/(PIXS_PER_COL), (i % (PIXS_PER_COL)) * COMBINED_PIXELS, (sliceIdx_t)(glPLActiveSliceIdxSW + 3));

		// Only calculate the OF when SFASTCorner is 1
		if(SFASTCorner)
		{
	        pix_t block1[BLOCK_SIZE_SCALE_0][BLOCK_SIZE_SCALE_0];
	        pix_t block2[BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE][BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE];

	        pix_t block1Scale1[BLOCK_SIZE_SCALE_1][BLOCK_SIZE_SCALE_1];
	        pix_t block2Scale1[BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE][BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE];

	        pix_t block1Scale2[BLOCK_SIZE_SCALE_2][BLOCK_SIZE_SCALE_2];
	        pix_t block2Scale2[BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE][BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE];

	        bool printBlocksEnable = false;
	        if(ts == 148768831)
			{
	        	printBlocksEnable = true;
			}

	        // Scale 2 computation
			for(int8_t xOffset = 0; xOffset < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; xOffset++)
	        {
	            pix_t out1Scale2[BLOCK_SIZE_SCALE_2+ 2 * SEARCH_DISTANCE];
	            pix_t out2Scale2[BLOCK_SIZE_SCALE_2+ 2 * SEARCH_DISTANCE];

				readBlockColsSWScale2(xWr/4 - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE + xOffset, yWr/4, 0, 0, (glPLActiveSliceIdxSW + 1), (glPLActiveSliceIdxSW + 2), out1Scale2, out2Scale2);

	            for(int8_t yCopyOffset = 0; yCopyOffset < BLOCK_SIZE_SCALE_2; yCopyOffset++)
	            {
	                if (xOffset >= SEARCH_DISTANCE && xOffset < BLOCK_SIZE_SCALE_2 + SEARCH_DISTANCE)
	                {
	                    block1Scale2[xOffset - SEARCH_DISTANCE][yCopyOffset] = out1Scale2[yCopyOffset + SEARCH_DISTANCE];
	                }
	            }

	            for(int8_t yCopyOffset = 0; yCopyOffset < BLOCK_SIZE_SCALE_2 + 2 * SEARCH_DISTANCE; yCopyOffset++)
	            {
	                block2Scale2[xOffset][yCopyOffset] = out2Scale2[yCopyOffset];
	            }
			}
	        miniBlockSADSWScale2(block1Scale2, block2Scale2, printBlocksEnable, &miniRetScale2, &OFRetScale2);

	        // Scale 1 computation
	        ap_int<8> xInitOffsetScale1 = ap_int<8>(OFRetScale2.range(2,0) - 3) << 1;
	        ap_int<8> yInitOffsetScale1 = ap_int<8>(OFRetScale2.range(5,3) - 3) << 1;
	        for(int8_t xOffset = 0; xOffset < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; xOffset++)
	        {
	            pix_t out1Scale1[BLOCK_SIZE_SCALE_1+ 2 * SEARCH_DISTANCE];
	            pix_t out2Scale1[BLOCK_SIZE_SCALE_1+ 2 * SEARCH_DISTANCE];

				readBlockColsSWScale1(xWr/2 - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE + xOffset, yWr/2, xInitOffsetScale1,  yInitOffsetScale1, (glPLActiveSliceIdxSW + 1), (glPLActiveSliceIdxSW + 2), out1Scale1, out2Scale1);

	            for(int8_t yCopyOffset = 0; yCopyOffset < BLOCK_SIZE_SCALE_1; yCopyOffset++)
	            {
	                if (xOffset >= SEARCH_DISTANCE && xOffset < BLOCK_SIZE_SCALE_1 + SEARCH_DISTANCE)
	                {
	                    block1Scale1[xOffset - SEARCH_DISTANCE][yCopyOffset] = out1Scale1[yCopyOffset + SEARCH_DISTANCE];
	                }
	            }

	            for(int8_t yCopyOffset = 0; yCopyOffset < BLOCK_SIZE_SCALE_1 + 2 * SEARCH_DISTANCE; yCopyOffset++)
	            {
	                block2Scale1[xOffset][yCopyOffset] = out2Scale1[yCopyOffset];
	            }
			}
	        miniBlockSADSWScale1(block1Scale1, block2Scale1, printBlocksEnable, &miniRetScale1, &OFRetScale1);

	        // Scale 0 computation
	        ap_int<8> xInitOffsetScale0 = (ap_int<8>(OFRetScale1.range(2,0) - 3) << 1) + (xInitOffsetScale1 << 1);
	        ap_int<8> yInitOffsetScale0 = (ap_int<8>(OFRetScale1.range(5,3) - 3) << 1) + (yInitOffsetScale1 << 1);
	        for(int8_t xOffset = 0; xOffset < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; xOffset++)
	        {
	            pix_t out1[BLOCK_SIZE_SCALE_0+ 2 * SEARCH_DISTANCE];
	            pix_t out2[BLOCK_SIZE_SCALE_0+ 2 * SEARCH_DISTANCE];

				readBlockColsSWScale0(xWr - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE + xOffset, yWr, xInitOffsetScale0, yInitOffsetScale0 , (glPLActiveSliceIdxSW + 1), (glPLActiveSliceIdxSW + 2), out1, out2);

	            for(int8_t yCopyOffset = 0; yCopyOffset < BLOCK_SIZE_SCALE_0; yCopyOffset++)
	            {
	                if (xOffset >= SEARCH_DISTANCE && xOffset < BLOCK_SIZE_SCALE_0 + SEARCH_DISTANCE)
	                {
	                    block1[xOffset - SEARCH_DISTANCE][yCopyOffset] = out1[yCopyOffset + SEARCH_DISTANCE];
	                }
	            }

	            for(int8_t yCopyOffset = 0; yCopyOffset < BLOCK_SIZE_SCALE_0 + 2 * SEARCH_DISTANCE; yCopyOffset++)
	            {
	                block2[xOffset][yCopyOffset] = out2[yCopyOffset];
	            }
			}
			miniBlockSADSWScale0(block1, block2, printBlocksEnable, &miniRetScale0, &OFRetScale0);

	        if (xWr/4 - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE < 0 || xWr/4 + BLOCK_SIZE_SCALE_2/2 + SEARCH_DISTANCE >= DVS_WIDTH/4
	                || yWr/4 - BLOCK_SIZE_SCALE_2/2 - SEARCH_DISTANCE < 0 || yWr/4 + BLOCK_SIZE_SCALE_2/2 + SEARCH_DISTANCE >= DVS_HEIGHT/4) {
	        	miniRetScale2 = 0x7fff;
	        	OFRetScale2 = 0x3f;
	        }
	        if ((xWr/2 + xInitOffsetScale1) - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE < 0 || (xWr/2 + xInitOffsetScale1) + BLOCK_SIZE_SCALE_1/2 + SEARCH_DISTANCE >= DVS_WIDTH/2
	                ||(yWr/2 + yInitOffsetScale1) - BLOCK_SIZE_SCALE_1/2 - SEARCH_DISTANCE < 0 || (yWr/2 + yInitOffsetScale1) + BLOCK_SIZE_SCALE_1/2 + SEARCH_DISTANCE >= DVS_HEIGHT/2) {
	        	miniRetScale1 = 0x7fff;
	        	OFRetScale1 = 0x3f;
	        }
	        if ((xWr/1 + xInitOffsetScale0) - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE < 0 || (xWr/1 + xInitOffsetScale0) + BLOCK_SIZE_SCALE_0/2 + SEARCH_DISTANCE >= DVS_WIDTH/1
	                || (yWr/1 + yInitOffsetScale0) - BLOCK_SIZE_SCALE_0/2 - SEARCH_DISTANCE < 0 || (yWr/1 + yInitOffsetScale0) + BLOCK_SIZE_SCALE_0/2 + SEARCH_DISTANCE >= DVS_HEIGHT/1) {
	        	miniRetScale0 = 0x7fff;
	        	OFRetScale0 = 0x3f;
	        }
		}


        // If the result is valid, then result scale is 0. Otherwise, set scaleRet 3.
        if( miniRetScale2 >= maxAllowedSadValueScale2
        		|| miniRetScale1 >= maxAllowedSadValueScale1
				|| miniRetScale0 >= maxAllowedSadValueScale0 )
        {
        	// invalid result
        	miniRet = 0x7fff;
        	OFRet = 0x7f7f;
        	scaleRet = 3;
        }
        else
        {
            ap_int<8> xOFRetScale2 = ap_int<8>(OFRetScale2.range(2,0)) - 3;
            ap_int<8> yOFRetScale2 = ap_int<8>(OFRetScale2.range(5,3)) - 3;
            ap_int<8> xOFRetScale1 = ap_int<8>(OFRetScale1.range(2,0)) - 3;
            ap_int<8> yOFRetScale1 = ap_int<8>(OFRetScale1.range(5,3)) - 3;
            ap_int<8> xOFRetScale0 = ap_int<8>(OFRetScale0.range(2,0)) - 3;
            ap_int<8> yOFRetScale0 = ap_int<8>(OFRetScale0.range(5,3)) - 3;

            ap_int<8> xOFRet = (xOFRetScale2 * 4) + (xOFRetScale1 * 2) + xOFRetScale0;
            ap_int<8> yOFRet = (yOFRetScale2 * 4) + (yOFRetScale1 * 2) + yOFRetScale0;

            OFRet.range(7, 0) = ap_uint<8>(xOFRet);
            OFRet.range(15, 8) = ap_uint<8>(yOFRet);

            miniRet = miniRetScale0;
            scaleRet = 0;
        }

//        // Remove outliers
//        int block1ZeroCnt = 0;
//        for(int8_t block1IdxX = 0; block1IdxX < BLOCK_SIZE; block1IdxX++)
//        {
//            for(int8_t block1IdxY = 0; block1IdxY < BLOCK_SIZE; block1IdxY++)
//            {
//                if(block1[block1IdxX][block1IdxY] == 0)
//                {
//                    block1ZeroCnt++;
//                }
//            }
//        }
//
//        if(block1ZeroCnt > BLOCK_SIZE * (BLOCK_SIZE - 1))
//        {
//            miniRet = 0x7fff;
//            OFRet = 0x3f;
//        }

		apUint17_t tmp1 = apUint17_t(xWr.to_int() + (yWr.to_int() << 8) + (pol << 16));
		ap_int<9> tmp2 = miniRet.range(8, 0);
        ap_uint<9> delataTs = ((currentTs - lastTs) >> 9); 
		apUint6_t tmpOF = OFRet;
		if(tmpOF == 0x3f) tmpOF = 0x1b; // 0x1b means both OF_x and OF_y are 3. Minus 3 to restore the real value.
		ap_uint<32> output = (ap_uint<32>(yWr) << 16) + (xWr << 1) + pol;
		ap_uint<32> custData = OFRet;
		custData.range(17, 16) = scaleRet;
		custData[23] = rotateFlg;
		*custDataOut++ = custData;
		*eventSlice++ = output.to_int();

        /* -----------------Feedback part------------------------ */
		feedbackSW(miniRet, OFRet, rotateFlg, &areaEventThrSW);
	}

	resetLoop: for (int16_t resetCnt = 0; resetCnt < 2048; resetCnt = resetCnt + 2)
	{
		resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL) * COMBINED_PIXELS, (sliceIdx_t)(glPLActiveSliceIdxSW + 3));
		resetPixSW(resetCnt/PIXS_PER_COL, (resetCnt % PIXS_PER_COL + 1) * COMBINED_PIXELS, (sliceIdx_t)(glPLActiveSliceIdxSW + 3));
	}
}



int main(int argc, char *argv[])
{
	int testTimes = TEST_TIMES;
    if (argc == 2) testTimes = atoi(argv[1]);

    int total_err_cnt = 0;
	int retval=0;

	/******************* Test EVABMOFStream module from aedat file with GT**************************/
//	srand(0);
//	srand((unsigned)time(NULL));
	FILE * fp;

	if((fp = fopen("E://xfOpenCV//hls_2018_1//topParseEvents//DDD17_BackFromAirport_areaThr_700_SFAST-OFResult_GT.bin","rb"))== NULL)
	{
		printf("can not open the file\n");
		exit(0);
	}

	int32_t eventCnt = 4000;
	uint64_t data[eventCnt];
	int32_t eventSlice[eventCnt], eventSliceSW[eventCnt];
	ap_uint<32> custDataOutSW[eventCnt];

	ap_int<16> miniSumRet;
//	pix_t refColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//	pix_t refColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];

	sliceIdx_t idx;

    uint64_t lastMaxTs = 0;  // Record last maximum ts to make all the ts are monotonic.

    ap_uint<32> config, status;

	uint64_t x_in[eventCnt], y_in[eventCnt];
	ap_uint<16> x_out[eventCnt], y_out[eventCnt];
	ap_uint<64> ts_in[eventCnt], ts_out[eventCnt];
	ap_uint<1> pol_in[eventCnt], pol_out[eventCnt];
	ap_uint<17> retData[eventCnt];
	ap_uint<32> GTData[eventCnt];

	hls::stream< ap_uint<16> > xStreamIn("xStreamInSW"), yStreamIn("yStreamInSW"), xStreamOut("xStreamOut"), yStreamOut("yStreamOut");
	hls::stream< ap_uint<64> > tsStreamIn("tsStreamInSW"), tsStreamOut("tsStreamOut");
	hls::stream< ap_uint<1> > polStreamIn("polStreamInSW"), polStreamOut("polStreamOut");
	hls::stream< ap_uint<1> > cornerStreamIn("cornerStream");
	hls::stream< ap_uint<17> > miscDataStream("miscDataStream");

	testTimes = 30;
	for(int k = 0; k < testTimes; k++)
	{
		cout << "Test " << k << ":" << endl;

	    int err_cnt = 0;

		for (int i = 0; i < eventCnt; i++)
		{
			ap_uint<32> buf[3];
			fread(buf, 4, 3, fp);
	       	uint32_t data1 = ((uint32_t)(buf[0].range(7, 0)) << 24) + ((uint32_t)(buf[0].range(15, 8)) << 16) + ((uint32_t)(buf[0].range(23, 16)) << 8) + buf[0].range(31, 24);
	       	uint32_t data2 = ((uint32_t)(buf[1].range(7, 0)) << 24) + ((uint32_t)(buf[1].range(15, 8)) << 16) + ((uint32_t)(buf[1].range(23, 16)) << 8) + buf[1].range(31, 24);
	       	uint32_t data3 = ((uint32_t)(buf[2].range(7, 0)) << 24) + ((uint32_t)(buf[2].range(15, 8)) << 16) + ((uint32_t)(buf[2].range(23, 16)) << 8) + buf[2].range(31, 24);

	        x_in[i] = ((data1) & AEDAT_POLARITY_X_ADDR_MASK) >> AEDAT_POLARITY_X_ADDR_SHIFT;
			y_in[i] = ((data1) & AEDAT_POLARITY_Y_ADDR_MASK) >> AEDAT_POLARITY_Y_ADDR_SHIFT;
			pol_in[i]  = ((data1) & AEDAT_POLARITY_MASK) >> AEDAT_POLARITY_SHIFT;
			ts_in[i] = data2;
			GTData[i] = data3;

			ap_uint<1> SFASTCorner = GTData[i].bit(24);
			if(k == 8 && i == 3457)
			{
				int tmp = 0;
			}
//			idx = rand()%3;
	//		x = 255;
	//		y = 240;
//			cout << "x : " << x << endl;
//			cout << "y : " << y << endl;
//			cout << "idx : " << idx << endl;

			xStreamIn << x_in[i];
			yStreamIn << y_in[i];
			tsStreamIn << ts_in[i];
			polStreamIn << pol_in[i];
			cornerStreamIn << SFASTCorner;

			EVABMOFStreamWithControl(xStreamIn, yStreamIn, tsStreamIn, polStreamIn,
					cornerStreamIn,
					xStreamOut, yStreamOut, tsStreamOut,polStreamOut, miscDataStream,
					config, &status);

			x_out[i] = xStreamOut.read().to_uint();
			y_out[i] = yStreamOut.read().to_uint();
			ts_out[i] = tsStreamOut.read().to_uint();
			pol_out[i] = polStreamOut.read().to_bool();
			retData[i] = miscDataStream.read();

//			EVABMOFScalar(x_in[i], y_in[i], ts_in[i], pol_in[i],
//					&x_out[i], &y_out[i], &ts_out[i], &pol_out[i],
//					&retData[i],
//					config, &status);

			// GTData bit 24 is SFAST corner flag bit.
			data[i] = (uint64_t)(ts_in[i] << 32) + (uint64_t)(x_in[i] << POLARITY_X_ADDR_SHIFT) + (uint64_t)(y_in[i] << POLARITY_Y_ADDR_SHIFT) + (pol_in[i] << POLARITY_SHIFT) + SFASTCorner;
//			cout << "data[" << i << "] is: "<< hex << data[i]  << endl;
		}

		parseEventsSW(data, eventCnt, eventSliceSW, custDataOutSW);

		for (int j = 0; j < eventCnt; j++)
		{
			ap_uint<32> tmpData = eventSliceSW[j];
			ap_uint<10> y = tmpData.range(25, 16);
			ap_uint<10> x = tmpData.range(10, 1);

			ap_uint<10> xHW = x_out[j];
			ap_uint<10> yHW = y_out[j];

			ap_uint<1> rotateFlgGT = GTData[j].bit(23);
			ap_uint<1> rotateFlgSW = custDataOutSW[j].bit(23);
//			ap_uint<1> rotateFlgHW = retData[j].bit(10);

			ap_uint<2> OFRetValidGT = GTData[j].bit(16);
			ap_uint<1> SFASTCornerGT = GTData[j].bit(24);
			ap_uint<2> scaleRetSW = custDataOutSW[j].range(17, 16);

			ap_int<8> xOFRetGT = GTData[j].range(7, 0) - 127;
			ap_int<8> xOFRetSW = custDataOutSW[j].range(7, 0);
//			ap_int<8> xOFRetHW = retData[j].range(4, 0);

			ap_int<8> yOFRetGT = GTData[j].range(15, 8) - 127;
			ap_int<8> yOFRetSW = custDataOutSW[j].range(15, 8);
//			ap_int<8> yOFRetHW = retData[j].range(9, 5);

			ap_uint<11> compressedRetDataHW = retData[j].range(10, 0);
			ap_uint<1> rotateFlgHW;
			ap_int<8> xOFRetHW;
			ap_int<8> yOFRetHW;
			if(compressedRetDataHW == ap_uint<11>(0xffff))      // invalid data and not rotation flag
			{
				rotateFlgHW = 0;
				xOFRetHW = 0x7f;
				yOFRetHW = 0x7f;
			}
			else if(compressedRetDataHW == ap_uint<11>(0xfffe))  // RotateFlag data doesn't contain OF information
			{
				rotateFlgHW = 1;
				xOFRetHW = xOFRetSW;
				yOFRetHW = yOFRetSW;
			}
			else                                 // normal data, decompress it.
			{
				rotateFlgHW = 0;
				ap_uint<8> xOFRetNoSignHW = compressedRetDataHW/(2 * MAX_SEARCH_DIST_RADIUS + 1);
				ap_uint<8> yOFRetNoSignHW = compressedRetDataHW%(2 * MAX_SEARCH_DIST_RADIUS + 1);
				xOFRetHW = xOFRetNoSignHW - MAX_SEARCH_DIST_RADIUS;
				yOFRetHW = yOFRetNoSignHW - MAX_SEARCH_DIST_RADIUS;
			}

			// Compare HW and SW
			if( (rotateFlgHW != rotateFlgSW) || (x != xHW) || (y != yHW)
					|| (xOFRetHW.range(4, 0) != xOFRetSW.range(4, 0))
					|| (yOFRetHW.range(4, 0) != yOFRetSW.range(4, 0)) )
			{
				cout << "HW and SW is different." << endl;
                cout << "x for SW is: " << x << "\t y for SW is: " << y << endl;
                cout << "x for HW is: " << xHW << "\t y for HW is: " << yHW << endl;
				std::cout << "OF for eventSlice SW is: " << hex << custDataOutSW[j] << std::endl;
				std::cout << "OF for eventSlice HW is: " << hex << retData[j] << std::endl;
				std::cout << "OF for GT is: " << hex << GTData[j] << std::endl;
				cout << dec;
				err_cnt++;
				cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
			}

			// Compare SW and GT
			if( (rotateFlgGT != rotateFlgSW) )    // Rotate Flag is not correct
			{
				cout << "Rotation Flag is different." << endl;
                cout << "x is: " << x << "\t y is: " << y << endl;
				std::cout << "OF for eventSlice SW is: " << hex << custDataOutSW[j] << std::endl;
				std::cout << "OF for eventSlice HW is: " << hex << retData[j] << std::endl;
				std::cout << "OF for GT is: " << hex << GTData[j] << std::endl;
				cout << dec;
				err_cnt++;
				cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
			}
			else
			if( OFRetValidGT == 1 && SFASTCornerGT)    // scaleGT = 0: a valid OF result from GT; SFASTCornerGT = 1 means it is a corner; further checking
			{
				if( (scaleRetSW != 0) || (xOFRetGT != -xOFRetSW) || (yOFRetGT != -yOFRetSW) ) // java OF_GT has different sign with this C++ testbench.
				{
					cout << "Valid GT OF check failed." << endl;
	                cout << "x is: " << x << "\t y is: " << y << endl;
					std::cout << "OF for eventSlice SW is: " << hex << custDataOutSW[j] << std::endl;
					std::cout << "OF for eventSlice HW is: " << hex << retData[j] << std::endl;
					std::cout << "OF for GT is: " << hex << GTData[j] << std::endl;
					cout << dec;
					err_cnt++;
					cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
				}
			}
			else                                 // scaleGT != 0: an invalid OF result from GT
			{
				if( scaleRetSW != 3 )
				{
					cout << "GT OF is invalid while SW OF is valid." << endl;
	                cout << "x is: " << x << "\t y is: " << y << endl;
					std::cout << "OF for eventSlice SW is: " << hex << custDataOutSW[j] << std::endl;
					std::cout << "OF for eventSlice HW is: " << hex << retData[j] << std::endl;
					std::cout << "OF for GT is: " << hex << GTData[j] << std::endl;
					cout << dec;
					err_cnt++;
					cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
				}
			}
		}

		if(err_cnt == 0)
		{
			cout << "Test " << k << " passed." << endl;
		}
		total_err_cnt += err_cnt;
		cout << endl;
	}

//	/******************* Test EVABMOFStream module from random value**************************/
//	srand(0);
////	srand((unsigned)time(NULL));
//
//	int32_t eventCnt = 4000;
//	uint64_t data[eventCnt];
//	int32_t eventSlice[eventCnt], eventSliceSW[eventCnt];
//
//	ap_int<16> miniSumRet;
//	pix_t refColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//	pix_t refColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//
//	ap_uint<64> x, y;
//	ap_uint<64> ts[eventCnt];
//	ap_uint<1> pol;
//	sliceIdx_t idx;
//
//    uint64_t lastMaxTs = 0;  // Record last maximum ts to make all the ts are monotonic.
//
//	ap_uint<16> x_out[eventCnt], y_out[eventCnt];
//	ap_uint<64> ts_out[eventCnt];
//	ap_uint<1>  pol_out[eventCnt];
//	ap_uint<10> retData[eventCnt];
//
//	hls::stream< ap_uint<16> > xStreamIn("xStreamIn"), yStreamIn("yStreamIn"), xStreamOut("xStreamOut"), yStreamOut("yStreamOut");
//	hls::stream< ap_uint<64> > tsStreamIn("tsStreamIn"), tsStreamOut("tsStreamOut");
//	hls::stream< ap_uint<1> > polStreamIn("polStreamIn"), polStreamOut("polStreamOut");
//	hls::stream< ap_uint<10> > miscDataStream("miscDataStream");
//
//	for(int k = 0; k < TEST_TIMES; k++)
//	{
//		cout << "Test " << k << ":" << endl;
//
//	    int err_cnt = 0;
//
//		idx = sliceIdx_t(idx - 1);
//
//		for (int m = 0; m < eventCnt; m++)
//		{
//			ts[m]  = rand() + lastMaxTs;
//		}
//		sort(ts, ts+eventCnt);
//		lastMaxTs = ts[eventCnt -1];
//
//		for (int i = 0; i < eventCnt; i++)
//		{
//			x = rand()%50 + 40;
//			y = rand()%50 + 40;
//			pol = rand()%2;
////			idx = rand()%3;
//	//		x = 255;
//	//		y = 240;
////			cout << "x : " << x << endl;
////			cout << "y : " << y << endl;
////			cout << "idx : " << idx << endl;
//
//			xStreamIn << x;
//			yStreamIn << y;
//			tsStreamIn << ts[i];
//			polStreamIn << pol;
//
//			EVABMOFStreamNoConfigNoStaus(xStreamIn, yStreamIn, tsStreamIn, polStreamIn,
//					xStreamOut, yStreamOut, tsStreamOut, polStreamOut, miscDataStream);
//
//			xStreamOut >> x_out[i];
//			yStreamOut >> y_out[i];
//			tsStreamOut >> ts_out[i];
//			polStreamOut >> pol_out[i];
//			miscDataStream >> retData[i];
//
//			data[i] = (uint64_t)(ts[i] << 32) + (uint64_t)(x << 17) + (uint64_t)(y << 2) + (pol << 1);
////			cout << "data[" << i << "] is: "<< hex << data[i]  << endl;
//		}
//
//		parseEventsSW(data, eventCnt, eventSliceSW);
//
//		for (int j = 0; j < eventCnt; j++)
//		{
//			ap_uint<32> tmpData = eventSliceSW[j];
//			ap_uint<6> tmpOF = tmpData.range(22, 17);
//			ap_uint<9> deltaTs = tmpData.range(31, 23);
//
//			if(j == 0)
//			{
//				std::cout << "deltaTs from SW at index j = 0 is : " << deltaTs << std::endl;
//			}
//			if (((retData[j].bit(9) == 0) && (retData[j].range(6, 0) != tmpOF))
//					|| ((retData[j].bit(9) == 1) && (retData[j].range(8, 0) != deltaTs)))
//			{
//				std::cout << "OF for eventSliceSW is: " << tmpOF << std::endl;
//				std::cout << "OF for eventSlice is: " << retData[j] << std::endl;
//
//				err_cnt++;
//				cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
//			}
//		}
//
//		if(err_cnt == 0)
//		{
//			cout << "Test " << k << " passed." << endl;
//		}
//		total_err_cnt += err_cnt;
//		cout << endl;
//	}

//	/******************* Test parseEvents module from random value**************************/
//	int32_t eventCnt = 500;
//	uint64_t data[eventCnt];
//	int32_t eventSlice[eventCnt], eventSliceSW[eventCnt];
//
//	ap_int<16> miniSumRet;
//	pix_t refColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//	pix_t refColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//
//	ap_uint<64> x, y;
//	ap_uint<1> pol;
//	sliceIdx_t idx;
//
//	for(int k = 0; k < TEST_TIMES; k++)
//	{
//		cout << "Test " << k << ":" << endl;
//
//	    int err_cnt = 0;
//
//		idx = sliceIdx_t(idx - 1);
//
//		for (int i = 0; i < eventCnt; i++)
//		{
//			x = rand()%50 + 40;
//			y = rand()%50 + 40;
//			pol = rand()%2;
////			idx = rand()%3;
//	//		x = 255;
//	//		y = 240;
////			cout << "x : " << x << endl;
////			cout << "y : " << y << endl;
////			cout << "idx : " << idx << endl;
//
//			data[i] = (uint64_t)(x << 17) + (uint64_t)(y << 2) + (pol << 1);
////			cout << "data[" << i << "] is: "<< hex << data[i]  << endl;
//		}
//
//
//		parseEventsSW(data, eventCnt, eventSliceSW);
//		parseEvents(data, eventCnt, eventSlice);
//
//		for (int j = 0; j < eventCnt; j++)
//		{
//			if (eventSlice[j] != eventSliceSW[j])
//			{
//				std::cout << "eventSliceSW is: " << eventSliceSW[j] << std::endl;
//				std::cout << "eventSlice is: " << eventSlice[j] << std::endl;
//
//				err_cnt++;
//				cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
//			}
//		}
//
//		if(err_cnt == 0)
//		{
//			cout << "Test " << k << " passed." << endl;
//		}
//		total_err_cnt += err_cnt;
//		cout << endl;
//	}

	/******************* Test parseEvents module from specific file**************************/
//	int32_t eventCnt = 500;
//	uint64_t data[eventCnt];
//	int32_t eventSlice[eventCnt], eventSliceSW[eventCnt];
//
//    int startLine = 0;
//
//    ofstream resultfile;
//    resultfile.open ("testResult.txt");
//
//    for(int k = 0; k < TEST_TIMES; k++)
//    {
//    	cout << "Test " << k << ":" << endl;
//
//		int err_cnt = 0;
//
//	    ifstream file("box_trans-withOFResult_areaThr_1000_hardware_order_mutiscale-OFResult.txt");
//	    string str;
//
//	    // Nothing is executed until we arrived the desired line.
//	    for (int lineno = 0; getline (file,str) && lineno < startLine; lineno++);
//
//    	int lineCnt = 0;
//        std::cout << "Start reading line: " << startLine << std::endl;
//        while (getline(file, str))
//        {
//    	   stringstream stream(str);
//    	   uint64_t ts;
//			int x;
//			int y;
//			int polarity;
//			int OF_x;
//			int OF_y;
//			int OF_scale;
//			stream >> ts;
//			stream >> x;
//			stream >> y;
//			stream >> polarity;
//			stream >> OF_x;
//			stream >> OF_y;
//			stream >> OF_scale;
//			OF_x = (OF_x >> OF_scale);
//			OF_y = (OF_y >> OF_scale);
//
//			// y = DVS_HEIGHT -1 - y;   // OpenCV and jaer has inverse y coordinate.
//
//			if( y >= DVS_HEIGHT || y < 0 )  std::cout << "ts is :" << ts << "\t x is: " << x << "\t y is :" << y << "\t pol is:" << polarity << std::endl;
//			if( x >= DVS_WIDTH || x < 0 )  std::cout << "ts is :" << ts << "\t x is: " << x << "\t y is :" << y << "\t pol is:" << polarity << std::endl;
//
//			uint64_t temp = 0;
//			temp = (ts << 32) + ((3 - OF_y) << 29) + ((3 - OF_x) << 26) + (x << 17) + (OF_scale << 14) + (y << 2) + (polarity << 1) + 1;
//			data[lineCnt] = temp;
//
//			if(lineCnt >= eventCnt)
//			{
//				break;
//			}
//			lineCnt++;
//        }
//
//        startLine += eventCnt;
//
//		parseEventsSW(data, eventCnt, eventSliceSW);
//		parseEvents(data, eventCnt, eventSlice);
//
//		for (int j = 0; j < eventCnt; j++)
//		{
//			resultfile << (startLine - eventCnt + j) << (eventSlice[j] & 0xff) << " " << ((eventSlice[j] >> 8) & 0xff) << " "
//					<< ((eventSlice[j] >> 16) & 0x1) << " " <<  ((eventSlice[j] >> 17) & 0x7) << " "
//					<<  ((eventSlice[j] >> 20) & 0x7) << std::endl;
//
//			if (eventSlice[j] != eventSliceSW[j])
//			{
//				std::cout << "eventSliceSW is: " << eventSliceSW[j] << std::endl;
//				std::cout << "eventSlice is: " << eventSlice[j] << std::endl;
//
//				err_cnt++;
//				cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
//			}
//		}
//
//		if(err_cnt == 0)
//		{
//			cout << "Test " << k << " passed." << endl;
//		}
//		total_err_cnt += err_cnt;
//		cout << endl;
//	}
//
//	resultfile.close();


//		/******************* Test feedback module **************************/
//	int32_t eventCnt = 500;
//	apUint15_t miniSumRet;
//	apUint6_t OFRet;
//	apUint1_t rotateFlg;
//	uint16_t thrRetSW[eventCnt], thrRet[eventCnt];
//
//	for(int k = 0; k < TEST_TIMES; k++)
//	{
//		cout << "Test " << k << ":" << endl;
//		int err_cnt = 0;
//
//		for (int i = 0; i < eventCnt; i++)
//		{
//			miniSumRet = apUint15_t(rand()%0xffff);
//			OFRet = apUint6_t(rand()%7 + (rand()%7 << 3));
//			rotateFlg = rand()%2;
//
//			cout << "miniSumRet is: "  << hex << miniSumRet << endl;
//			cout << "OFRet is: "  << hex << OFRet << endl;
//			cout << "rotateFlg is: "  << hex << rotateFlg << endl;
//			cout << "areaEventThrSW is: "<< dec << areaEventThrSW << endl;
//
//			feedbackSW(miniSumRet, OFRet, rotateFlg, &thrRetSW[i]);
//			feedback(miniSumRet, OFRet, rotateFlg, &thrRet[i]);
//		}
//
//		for (int j = 0; j < eventCnt; j++)
//		{
//			if (thrRet[j] != thrRetSW[j])
//			{
//				std::cout << "thrRetSW is: " << thrRetSW[j] << std::endl;
//				std::cout << "thrRet is: " << thrRet[j] << std::endl;
//
//				err_cnt++;
//				cout << "Mismatch detected on TEST " << k << " and the mismatch index is: " << j << endl;
//			}
//		}
//
//		if(err_cnt == 0)
//		{
//			cout << "Test " << k << " passed." << endl;
//		}
//		total_err_cnt += err_cnt;
//		cout << endl;
//	}

	/******************* Test testTemp module **************************/
//	srand((unsigned)time(NULL));
//	int16_t eventCnt = 1000;
//
//	uint64_t data[eventCnt];
//	apIntBlockCol_t refData[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)], tagData[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)];
//	apIntBlockCol_t refDataSW[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)], tagDataSW[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)];
//
//	apUint15_t miniSum[eventCnt], miniSumSW[eventCnt];
//	apUint6_t OFRet[eventCnt], OFRetSW[eventCnt];
//
//	int32_t eventSlice[eventCnt], eventSliceSW[eventCnt];
//
//	ap_uint<64> x, y;
//	sliceIdx_t idx;
//
//	for(int k = 0; k < TEST_TIMES; k++)
//	{
//		cout << "Test " << k << ":" << endl;
//
//		idx = sliceIdx_t(idx - 1);
//
//		for (int i = 0; i < eventCnt; i++)
//		{
//			x = rand()%50;
//			y = rand()%50 + COMBINED_PIXELS;
////			idx = rand()%3;
//	//		x = 255;
//	//		y = 240;
////			cout << "x : " << x << endl;
////			cout << "y : " << y << endl;
////			cout << "idx : " << idx << endl;
//
//			data[i] = (uint64_t)(x << 17) + (uint64_t)(y << 2) + (1 << 1);
////			cout << "data[" << i << "] is: "<< hex << data[i]  << endl;
//		}
//
//
//		testTempSW(data, idx, eventCnt, eventSliceSW);
//		testTemp(data, idx, eventCnt, eventSlice);
//
//		for (int m = 0; m < eventCnt; m++)
//		{
//			if(eventSliceSW[m] != eventSlice[m])
//			{
//				std::cout << "eventSliceSW is: " << eventSliceSW[m] << std::endl;
//				std::cout << "eventSlice is: " << eventSlice[m] << std::endl;
//
//				err_cnt++;
//				cout<<"!!! ERROR: Mismatch detected at index" << m << "!!!" << endl;
//			}
//		}
//
//		if(err_cnt == 0)
//		{
//			cout << "Test " << k << " passed." << endl;
//		}
//		cout << endl;
//	}




	/******************* Test rwSlices module **************************/
//	srand((unsigned)time(NULL));
//	int16_t eventCnt = 500;
//
//	uint64_t data[eventCnt];
//	apIntBlockCol_t refData[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)], tagData[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)];
//	apIntBlockCol_t refDataSW[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)], tagDataSW[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)];
//
//	ap_uint<64> x, y;
//	sliceIdx_t idx;
//
//	for(int k = 0; k < TEST_TIMES; k++)
//	{
//		cout << "Test " << k << ":" << endl;
//
//		idx = sliceIdx_t(idx - 1);
//
//		for (int i = 0; i < eventCnt; i++)
//		{
//			x = rand()%20;
//			y = rand()%20 + COMBINED_PIXELS;
////			idx = rand()%3;
//	//		x = 255;
//	//		y = 240;
////			cout << "x : " << x << endl;
////			cout << "y : " << y << endl;
////			cout << "idx : " << idx << endl;
//
//			data[i] = (uint64_t)(x << 17) + (uint64_t)(y << 2) + (1 << 1);
////			cout << "data[" << i << "] is: "<< hex << data[i]  << endl;
//		}
//
//
//		testRwslicesSW(data, idx, eventCnt, refDataSW, tagDataSW);
//		testRwslices(data, idx, eventCnt, refData, tagData);
//
//		for (int m = 0; m < eventCnt; m++)
//		{
//			cout  << "refDataSW is: " << hex <<  refDataSW[m] << endl;
//			cout  << "tagDataSW is: " << hex <<  tagDataSW[m] << endl;
//			cout  << "refDataHW is: " << hex <<  refData[m] << endl;
//			cout  << "tagDataHW is: " << hex <<  tagData[m] << endl;
//			cout << dec;
//
//			if (refDataSW[m] != 0 && tagDataSW[m] == 0)
//			{
//				cout << "Should not stop here, it is only a debug breakpoint." << endl;
//			}
//
//			if(refDataSW[m] != refData[m] || tagDataSW[m] != tagData[m])
//			{
//				pix_t outSW1[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//				pix_t outSW2[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//				pix_t outHW1[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//				pix_t outHW2[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//
//				for(int i = 0; i < BLOCK_SIZE + 2 * SEARCH_DISTANCE; i++)
//				{
//					outSW1[i] = refDataSW[m].range(4 * i + 3, 4 * i);
//					outSW2[i] = tagDataSW[m].range(4 * i + 3, 4 * i);
//					outHW1[i] = refData[m].range(4 * i + 3, 4 * i);
//					outHW2[i] = tagData[m].range(4 * i + 3, 4 * i);
//				}
//
//				cout << "refDataSW is:  ";
//				for (int n = 0; n < BLOCK_SIZE + 2 * SEARCH_DISTANCE; n++)
//				{
//					cout << outSW1[n] << " ";
//				}
//				cout << "\t" ;
//
//				cout << "tagDataSW is:  ";
//				for (int n = 0; n < BLOCK_SIZE + 2 * SEARCH_DISTANCE; n++)
//				{
//					cout << outSW2[n] << " ";
//				}
//				cout << endl;
//
//				cout << "refData is:  ";
//				for (int n = 0; n < BLOCK_SIZE + 2 * SEARCH_DISTANCE; n++)
//				{
//					cout << outHW1[n] << " ";
//				}
//				cout << "\t" ;
//
//				cout << "tagData is:  ";
//				for (int n = 0; n < BLOCK_SIZE + 2 * SEARCH_DISTANCE; n++)
//				{
//					cout << outHW2[n] << " ";
//				}
//				cout << endl;
//
////				std::cout << "refDataSW is: " << refDataSW[m].to_ulong() << "\t tagDataSW is: " << std::hex << tagDataSW[m].to_ulong() << std::endl;
////				std::cout << "refData is: " << refData[m].to_ulong() << "\t tagData is: " << std::hex << tagData[m].to_ulong() << std::endl;
//
//				err_cnt++;
//				cout<<"!!! ERROR: Mismatch detected at index" << m << "!!!" << endl;
//			}
//		}
//
//		if(err_cnt == 0)
//		{
//			cout << "Test " << k << " passed." << endl;
//		}
//		cout << endl;
//	}

	/******************* Test miniSADSumWrapper module **************************/
//	srand((unsigned)time(NULL));
//	int16_t eventCnt = 20;
//
//	apIntBlockCol_t refBlockColData[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)];
//	apIntBlockCol_t tagBlockColData[eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE)];
//
//	apUint15_t miniSum[eventCnt], miniSumSW[eventCnt];
//	apUint6_t OFRet[eventCnt], OFRetSW[eventCnt];
//
//	for(int k = 0; k < TEST_TIMES; k++)
//	{
//		cout << "Test " << k << ":" << endl;
////		cout << "Input data is: " << endl;
//		for (int i = 0; i < eventCnt * (BLOCK_SIZE + 2 * SEARCH_DISTANCE); i++)
//		{
////			cout << "#" << i << ": " <<  endl;
//
//			for(int j = 0; j < BLOCK_SIZE + 2 * SEARCH_DISTANCE; j++)
//			{
//				pix_t tmp1 = pix_t(rand() % 16);
//				pix_t tmp2 = pix_t(rand() % 16);
//				refBlockColData[i].range(4 * j + 3, 4 * j) = tmp1;
//				tagBlockColData[i].range(4 * j + 3, 4 * j) = tmp2;
//
////				cout << tmp1 << "  " << tmp2 << "  ";
//			}
////			cout << endl;
////			cout << "refBlockColData[eventCnt] is: " << refBlockColData[eventCnt] << "\t tagBlockColData[eventCnt] is: " << tagBlockColData[eventCnt] << endl;
//		}
//		testMiniSADSumWrapperSW(refBlockColData, tagBlockColData, eventCnt, miniSumSW, OFRetSW);
//		testMiniSADSumWrapper(refBlockColData, tagBlockColData, eventCnt, miniSum, OFRet);
//
//		for (int m = 0; m < eventCnt; m++)
//		{
//			if(miniSumSW[m] != miniSum[m] || OFRetSW[m] != OFRet[m])
//			{
//				std::cout << "miniSumRetSW is: " << miniSumSW[m] << "\t OFRetSW is: " << std::hex << OFRetSW[m] << std::endl;
//				std::cout << "miniSumRetHW is: " << miniSum[m] << "\t OFRetHW is: " << std::hex << OFRet[m] << std::endl;
//
//				err_cnt++;
//				cout<<"!!! ERROR: Mismatch detected at index" << m << "!!!" << endl;
//			}
//		}
//
//		cout << endl;
//		cout << endl;
//		cout << endl;
//		cout << endl;
//	}


//	/******************* Test miniSADSum module **************************/
//	srand((unsigned)time(NULL));
//
//	pix_t input1[BLOCK_SIZE + 2 * SEARCH_DISTANCE],
//			input2[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//	ap_int<16> miniSum, miniSumSW;
//	ap_uint<6> OFRet, OFRetSW;
//
//	for(int k = 0; k < TEST_TIMES; k++)
//	{
//		cout << "Test " << k << ":" << endl;
//
//		for(int j = 0; j < BLOCK_SIZE + 2 * SEARCH_DISTANCE; j++)
//		{
//			input1[j] = rand() % 16;
//			input2[j] = rand() % 16;
//		}
//		miniSADSumSW(input1, input2, k, &miniSumSW, &OFRetSW);
//		miniSADSum(input1, input2, k + 1, &miniSum, &OFRet);
//
//		// Compare the results file with the golden results
//		cout << "miniSumSW is: " << miniSumSW << "\t OFRetSW is: " << hex << OFRetSW << endl;
//		cout << dec;    // Restore dec mode
//
//		cout << "miniSumHW is: " << miniSum << "\t OFRetHW is: " << hex << OFRet << endl;
//		cout << dec;    // Restore dec mode
//
//		for(int i = 0; i < 2 * SEARCH_DISTANCE + 1; i++)
//		{
//			if(miniSum != miniSumSW || OFRet != OFRetSW)
//			{
//				err_cnt++;
//				cout<<"!!! ERROR: Mismatch detected at index" << i << "!!!" << endl;
//			}
//		}
//
//		cout << endl;
//	}

//	/******************* Test singleRwslices module **************************/
//	pix_t refColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColSW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//	pix_t refColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE], tagColHW[BLOCK_SIZE + 2 * SEARCH_DISTANCE];
//
//	for(int k = 0; k < testTimes; k++)
//	{
//		ap_uint<8> x, y;
//		ap_uint<2> idx;
//
//		cout << "Test " << k << ":" << endl;
//		x = rand()%20;
//		y = rand()%20;
//		idx = rand()%3;
////		x = 255;
////		y = 240;
////		idx++;
//		cout << "x : " << x << endl;
//		cout << "y : " << y << endl;
//		cout << "idx : " << idx << endl;
//
//		testSingleRwslicesSW(x, y, idx, refColSW, tagColSW);
//		testSingleRwslicesHW(x, y, idx, refColHW, tagColHW);
//
//		cout << "refColSW is: " << endl;
//		for (int m = 0; m < BLOCK_SIZE + 2 * SEARCH_DISTANCE; m++)
//		{
//			cout << refColSW[m] << " ";
//		}
//		cout << endl;
//
//		cout << "tagColSW is: " << endl;
//		for (int m = 0; m < BLOCK_SIZE + 2 * SEARCH_DISTANCE; m++)
//		{
//			cout << tagColSW[m] << " ";
//		}
//		cout << endl;
//
//		cout << "refColHW is: " << endl;
//		for (int m = 0; m < BLOCK_SIZE + 2 * SEARCH_DISTANCE; m++)
//		{
//			cout << refColHW[m] << " ";
//		}
//		cout << endl;
//
//		cout << "tagColHW is: " << endl;
//		for (int m = 0; m < BLOCK_SIZE + 2 * SEARCH_DISTANCE; m++)
//		{
//			cout << tagColHW[m] << " ";
//		}
//		cout << endl;
//
//		for (int i = 0; i < BLOCK_SIZE + 2 * SEARCH_DISTANCE; i++)
//		{
//			{
//				if((refColHW[i] != refColSW[i]) || (tagColHW[i] != tagColSW[i]))
//				{
//					err_cnt++;
//					cout<<"!!! ERROR: Mismatch detected at index" << i << " !!!" << endl;
//				}
//			}
//		}
//
//		cout << endl;
//	}


//		/******************* Test min module **************************/
//		ap_int<16> testData[2*SEARCH_DISTANCE + 1];
//		ap_int<16> minSW, minHW;
//		int8_t indexSW, indexHW;
//
//		cout << "Start testing min module...... " << endl;
//
//		for(int k = 0; k < TEST_TIMES; k++)
//		{
//			cout << "Test " << k << ":" << endl;
//
//			for(int j = 0; j < 2*SEARCH_DISTANCE + 1; j++)
//			{
//				testData[j] = ap_int<16>(rand());
//			}
//
//			cout << "Test data is: " << endl;
//			for (int m = 0; m <= 2 * SEARCH_DISTANCE; m++)
//			{
//				cout << testData[m].to_short() << " ";
//			}
//			cout << endl;
//
//			minSW = *min_element(testData, testData + 2*SEARCH_DISTANCE + 1);
//			indexSW = distance(testData, min_element(testData, testData + 2*SEARCH_DISTANCE + 1));
//			minHW = min(testData, &indexHW);
//
//			cout << "minSW is: " << minSW.to_short() << "\t indexSW is: " << (short)indexSW << endl;
//			cout << "minHW is: " << minHW.to_short() << "\t indexHW is: " << (short)indexHW << endl;
//
//			if((minSW != minHW) || (indexSW != indexHW))
//			{
//				err_cnt++;
//				cout<<"!!! ERROR: Mismatch detected at index" << k << "!!!" << endl;
//			}
//			cout << endl;
//		}

	if (total_err_cnt == 0)
	{
			cout<<"*** TEST PASSED ***" << endl;
			retval = 0;
	} else
	{
			cout<<"!!! TEST FAILED - " << total_err_cnt << " mismatches detected !!!";
			cout<< endl;
			retval = -1;
	}

	// Return 0 if the test passes
	return retval;
}
