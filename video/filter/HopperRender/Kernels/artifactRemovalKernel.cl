// Kernel that removes artifacts from the warped frame
__kernel void artifactRemovalKernel(__global const unsigned short* frame1,
									__global const float* blurredMask,
									__global unsigned short* warpedFrame,
									const int dimY,
									const int dimX,
									const int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);
	const int threadIndex2D = cy * dimX + cx; // Standard thread index without Z-Dim

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		warpedFrame[threadIndex2D] = (float)warpedFrame[threadIndex2D] * (1.0f - blurredMask[threadIndex2D]) + (float)frame1[threadIndex2D] * blurredMask[threadIndex2D];
		//warpedFrame[threadIndex2D] = (unsigned short)(blurredMask[threadIndex2D] * 65535.0f);

	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		warpedFrame[channelIdxOffset + threadIndex2D] = (float)warpedFrame[channelIdxOffset + threadIndex2D] * (1.0f - blurredMask[(cy << 1) * dimX + cx]) + (float)frame1[channelIdxOffset + threadIndex2D] * blurredMask[(cy << 1) * dimX + cx];
		//warpedFrame[channelIdxOffset + threadIndex2D] = 32768;
	}
}