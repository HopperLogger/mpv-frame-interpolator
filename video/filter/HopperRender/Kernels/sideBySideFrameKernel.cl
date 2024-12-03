// Kernel that places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
__kernel void sideBySideFrameKernel(__global const unsigned short* frame1,
									__global const unsigned short* warpedFrame1,
									__global const unsigned short* warpedFrame2,
									__global unsigned short* outputFrame,
									const float frame1Scalar,
									const float frame2Scalar,
									const int dimY,
                                    const int dimX,
									const int halfDimY, 
									const int halfDimX,
									const int channelIdxOffset,
									const float blackLevel,
									const float whiteLevel,
									const float maxVal,
									const int middleValue) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);
	const int verticalOffset = dimY >> 2;
	const bool isYChannel = cz == 0 && cy < dimY && cx < dimX;
	const bool isUVChannel = cz == 1 && cy < halfDimY && cx < dimX;
	const bool isVChannel = (cx & 1) == 1;
	const bool isInLeftSideY = cy >= verticalOffset && cy < (verticalOffset + halfDimY) && cx < halfDimX;
	const bool isInRightSideY = cy >= verticalOffset && cy < (verticalOffset + halfDimY) && cx >= halfDimX && cx < dimX;
	const bool isInLeftSideUV = cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx < halfDimX;
	const bool isInRightSideUV = cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx >= halfDimX && cx < dimX;
	unsigned short blendedFrameValue;

	// Early exit if thread indices are out of bounds
    if (cz > 1 || cy >= dimY || cx >= dimX || (cz == 1 && cy >= halfDimY)) return;

	// --- Blending ---
	// Y Channel
	if (isYChannel && isInRightSideY) {
		blendedFrameValue = 
			(unsigned short)(
			    (float)(warpedFrame1[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) * frame1Scalar + 
				(float)(warpedFrame2[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) * frame2Scalar
			);
	// U/V Channels
	} else if (isUVChannel && isInRightSideUV) {
		blendedFrameValue = 
			(unsigned short)(
				(float)(warpedFrame1[channelIdxOffset + 2 * (cy - (verticalOffset >> 1)) * dimX + ((cx - halfDimX) << 1) + isVChannel]) * frame1Scalar + 
				(float)(warpedFrame2[channelIdxOffset + 2 * (cy - (verticalOffset >> 1)) * dimX + ((cx - halfDimX) << 1) + isVChannel]) * frame2Scalar
			);
	}

	// --- Insertion ---
	if (isYChannel) {
		// Y Channel Left Side
		if (isInLeftSideY) {
            outputFrame[cy * dimX + cx] = fmax(fmin(((float)(frame1[((cy - verticalOffset) << 1) * dimX + (cx << 1)]) - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f);
		// Y Channel Right Side
		} else if (isInRightSideY) {
            outputFrame[cy * dimX + cx] = fmax(fmin(((float)blendedFrameValue - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f);
		// Y Channel Black Frame
		} else {
			outputFrame[cy * dimX + cx] = 0;
		}
	} else if (isUVChannel) {
		// UV Channels Left Side
		if (isInLeftSideUV) {
			outputFrame[dimY * dimX + cy * dimX + cx] = frame1[channelIdxOffset + ((cy - (verticalOffset >> 1)) << 1) * dimX + (cx << 1) + isVChannel];
		// UV Channels Right Side
		} else if (isInRightSideUV) {
			outputFrame[dimY * dimX + cy * dimX + cx] = blendedFrameValue;
		// UV Channels Black Frame
		} else {
			outputFrame[dimY * dimX + cy * dimX + cx] = middleValue;
		}
	}
}