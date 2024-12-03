// Kernel that places half of frame 1 over the outputFrame
__kernel void insertFrameKernel(__global const unsigned short* frame1,
								__global unsigned short* outputFrame,
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

	// Y Channel
	if (cz == 0 && cy < dimY && cx < (dimX >> 1)) {
        outputFrame[cy * dimX + cx] = fmax(fmin(((float)frame1[cy * dimX + cx] - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < (dimX >> 1)) {
		outputFrame[channelIdxOffset + cy * dimX + cx] = frame1[channelIdxOffset + cy * dimX + cx];
	}
}