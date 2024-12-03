// Kernel that blends warpedFrame1 to warpedFrame2
__kernel void blendFrameKernel(__global const unsigned short* warpedFrame1,
							   __global const unsigned short* warpedFrame2,
							   __global unsigned short* outputFrame,
                               const float frame1Scalar,
							   const float frame2Scalar,
							   const int dimY,
                               const int dimX,
							   const int channelIdxOffset,
							   const float blackLevel,
							   const float whiteLevel,
							   const float maxVal) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);
	float pixelValue;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		pixelValue = (float)warpedFrame1[cy * dimX + cx] * frame1Scalar + 
					 (float)warpedFrame2[cy * dimX + cx] * frame2Scalar;
        outputFrame[cy * dimX + cx] = (unsigned short)fmax(fmin((pixelValue - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		pixelValue = (float)warpedFrame1[channelIdxOffset + cy * dimX + cx] * frame1Scalar + 
					 (float)warpedFrame2[channelIdxOffset + cy * dimX + cx] * frame2Scalar;
		outputFrame[channelIdxOffset + cy * dimX + cx] = (unsigned short)pixelValue;
	}
}