// Kernel that sets the initial offset array
__kernel void setInitialOffsetKernel(__global char* offsetArray,
									 const int searchWindowSize,
									 const int lowDimY,
									 const int lowDimX,
									 const int directionIdxOffset,
									 const int layerIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);
	
	if (cy < lowDimY && cx < lowDimX) {
		char offsetX = (cz % searchWindowSize) - (searchWindowSize / 2);
		char offsetY = (cz / searchWindowSize) - (searchWindowSize / 2);

		offsetArray[cz * layerIdxOffset + cy * lowDimX + cx] = offsetX;
		offsetArray[directionIdxOffset + cz * layerIdxOffset + cy * lowDimX + cx] = offsetY;
	}
}