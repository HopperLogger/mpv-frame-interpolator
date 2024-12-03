// Kernel that removes artifacts from the warped frame
__kernel void artifactRemovalKernel(__global const unsigned short* frame1,
									__global const int* hitCount,
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
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[threadIndex2D] = frame1[threadIndex2D];
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[channelIdxOffset + cy * dimX + cx] = frame1[channelIdxOffset + threadIndex2D];
		}
	}
}