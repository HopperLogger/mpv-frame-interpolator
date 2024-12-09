// Ensures the addition of a and b does not overflow
char safe_add(char a, char b) {
    if (a > 0 && b > 0 && a > CHAR_MAX - b) {
        return CHAR_MAX;
    }
    if (a < 0 && b < 0 && a < CHAR_MIN - b) {
        return CHAR_MIN;
    }
    return a + b;
}

// Kernel that adjusts the offset array based on the comparison results
__kernel void adjustOffsetArrayKernel(__global char* offsetArray,
									  __global const unsigned short* globalLowestLayerArray,
									  const int windowDim,
									  const int directionIdxOffset,
									  const int layerIdxOffset, 
									  const int searchWindowSize,
									  const int numLayers,
									  const int lowDimY,
									  const int lowDimX,
									  const int lastRun) {

	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim

	if (cy < lowDimY && cx < lowDimX) {
		// We only need the lowestLayer if we are still searching
		const int wx = (cx / windowDim) * windowDim;
		const int wy = (cy / windowDim) * windowDim;
		unsigned short lowestLayer = globalLowestLayerArray[wy * lowDimX + wx];

		char idealX = offsetArray[lowestLayer * layerIdxOffset + threadIndex2D];
		char idealY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];

		// If this is the last run, we need to adjust the offset array for the next iteration
		if (lastRun) {
			offsetArray[threadIndex2D] = idealX;
			offsetArray[directionIdxOffset + threadIndex2D] = idealY;
			return;
		}
		int i = 0;
		for (int cz = 0; cz < numLayers; cz++) {
			char offsetX = (i % searchWindowSize) - (searchWindowSize / 2);
			char offsetY = (i / searchWindowSize) - (searchWindowSize / 2);
			if (offsetX == 0 && offsetY == 0) {
				i++;
				continue;
			}
			if (cz != lowestLayer) {
				offsetArray[cz * layerIdxOffset + threadIndex2D] = safe_add(idealX, offsetX);
				offsetArray[directionIdxOffset + cz * layerIdxOffset + threadIndex2D] = safe_add(idealY, offsetY);
				i++;
			}
		}
	}
}