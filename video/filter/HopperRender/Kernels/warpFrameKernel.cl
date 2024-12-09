// Kernel that warps a frame according to the offset array
__kernel void warpFrameKernel(__global const unsigned short* frame1,
							  __global const char* offsetArray,
							  __global int* hitCount,
							  __global unsigned short* warpedFrame,
							  const float frameScalar,
							  const int lowDimY,
							  const int lowDimX,
							  const int dimY,
							  const int dimX,
							  const int resolutionScalar,
							  const int directionIdxOffset,
							  const int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
		const int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array
		const int offsetX = (int)round((float)(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar) ;
		const int offsetY = (int)round((float)(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx]) * frameScalar) ;
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < dimX) {
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomic_add(&hitCount[newCy * dimX + newCx], 1);
		}

	// U/V-Channel
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		const int scaledCx = (cx >> resolutionScalar) & ~1; // The X-Index of the current thread in the offset array
		const int scaledCy = (cy >> resolutionScalar) << 1; // The Y-Index of the current thread in the offset array
		const int offsetX = (int)round((float)(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
		const int offsetY = (int)round((float)(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx]) * frameScalar * 0.5);
		
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < (dimY >> 1) && newCx >= 0 && newCx < dimX) {
			// U-Channel
			if ((cx & 1) == 0) {
				warpedFrame[channelIdxOffset + newCy * dimX + (newCx & ~1)] = frame1[channelIdxOffset + cy * dimX + cx];

			// V-Channel
			} else {
				warpedFrame[channelIdxOffset + newCy * dimX + (newCx & ~1) + 1] = frame1[channelIdxOffset + cy * dimX + cx];
			}
		}
	}
}