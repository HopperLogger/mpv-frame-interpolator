// Kernel that normalizes all the pixel deltas of each window
__kernel void normalizeDeltaSumsKernel(__global const unsigned int* summedUpDeltaArray,
									   __global double* normalizedDeltaArray,
									   __global const char* offsetArray,
									   const int windowDim,
									   int numPixels,
									   const int directionIdxOffset,
									   const int layerIdxOffset,
									   const int lowDimY,
									   const int lowDimX) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);
	const int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim
	bool isWindowRepresent = cy % windowDim == 0 && cx % windowDim == 0;

	// Check if the thread is a window represent
	if (isWindowRepresent) {
		// Get the current window information
		const char offsetX = -offsetArray[cz * layerIdxOffset + threadIndex2D];
		const char offsetY = -offsetArray[directionIdxOffset + cz * layerIdxOffset + threadIndex2D];

		// Calculate the not overlapping pixels
		int overlapX;
		int overlapY;

		// Calculate the number of not overlapping pixels
		if ((cx + windowDim + offsetX > lowDimX) || (cx + offsetX < 0)) {
			overlapX = abs(offsetX) + (windowDim > lowDimX) ? (windowDim - lowDimX) : 0;
		} else {
			overlapX = 0;
		}

		if ((cy + windowDim + offsetY > lowDimY) || (cy + offsetY < 0)) {
			overlapY = abs(offsetY) + (windowDim > lowDimY) ? (windowDim - lowDimY) : 0;
		} else {
			overlapY = 0;
		}

		const int numNotOverlappingPixels = overlapY * overlapX;
		numPixels -= numNotOverlappingPixels;
		numPixels = max(numPixels, 1);

		// Normalize the summed up delta
		normalizedDeltaArray[cz * lowDimY * lowDimX + threadIndex2D] = (double)summedUpDeltaArray[cz * lowDimY * lowDimX + threadIndex2D] / (double)numPixels;
	}
}