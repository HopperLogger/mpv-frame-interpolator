// Kernel that creates an grayscale flow image from the offset array
__kernel void convertFlowToGrayscaleKernel(__global const char* flowArray,
										   __global unsigned short* outputFrame,
										   __global const unsigned short* frame1,
										   const int lowDimY,
										   const int lowDimX,
										   const int dimY,
										   const int dimX,
										   const int resolutionScalar,
										   const int directionIdxOffset,
										   const int channelIdxOffset,
										   const int maxVal,
										   const int middleValue) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);

	const int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
	const int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array

	// Get the current flow values
	char x;
	char y;
	if (cz == 0 && cy < dimY && cx < dimX) {
		x = flowArray[scaledCy * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + scaledCy * lowDimX + scaledCx];
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX){
		x = flowArray[(scaledCy << 1) * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + (scaledCy << 1) * lowDimX + scaledCx];
	}

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		outputFrame[cy * dimX + cx] = min((abs(x) + abs(y)) << (10), maxVal);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		outputFrame[channelIdxOffset + cy * dimX + cx] = middleValue;
	}
}