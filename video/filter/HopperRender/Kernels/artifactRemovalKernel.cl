// Kernel that removes artifacts from the warped frame
__kernel void artifactRemovalKernel(__global const int* warpedFrameint,
									__global const int* hitCount,
									__global unsigned short* frame,
									__global unsigned short* warpedFrame,
									const int isNewFrame,
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
		if (hitCount[threadIndex2D] != 0) {
			warpedFrame[threadIndex2D] = warpedFrameint[threadIndex2D] / hitCount[threadIndex2D];
		} else if (isNewFrame) {
			warpedFrame[threadIndex2D] = frame[cy * dimX + cx];
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		if (hitCount[channelIdxOffset + threadIndex2D] != 0) {
			warpedFrame[channelIdxOffset + threadIndex2D] = warpedFrameint[channelIdxOffset + threadIndex2D] / hitCount[channelIdxOffset + threadIndex2D];
		} else if (isNewFrame) {
			warpedFrame[channelIdxOffset + threadIndex2D] = frame[channelIdxOffset + cy * dimX + cx];
		}
	}
}