// Helper kernel for the calcDeltaSums kernel
void warpReduce8x8(volatile __local unsigned int* partialSums, int tIdx) {
	partialSums[tIdx] += partialSums[tIdx + 32];
	partialSums[tIdx] += partialSums[tIdx + 16];
	partialSums[tIdx] += partialSums[tIdx + 8];
	partialSums[tIdx] += partialSums[tIdx + 4];
	partialSums[tIdx] += partialSums[tIdx + 2];
	partialSums[tIdx] += partialSums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
void warpReduce4x4(volatile __local unsigned int* partialSums, int tIdx) {
	partialSums[tIdx] += partialSums[tIdx + 16];
	partialSums[tIdx] += partialSums[tIdx + 8];
	partialSums[tIdx] += partialSums[tIdx + 2];
	partialSums[tIdx] += partialSums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
void warpReduce2x2(volatile __local unsigned int* partialSums, int tIdx) {
	partialSums[tIdx] += partialSums[tIdx + 8];
	partialSums[tIdx] += partialSums[tIdx + 1];
}

// Kernel that sums up all the pixel deltas of each window
__kernel void calcDeltaSumsKernel(__global unsigned int* summedUpDeltaArray,
								  __global const unsigned short* frame1,
								  __global const unsigned short* frame2,
								  __global const char* offsetArray,
								  const int directionIndexOffset,
						    	  const int dimY,
								  const int dimX,
								  const int lowDimY,
								  const int lowDimX,
								  const int windowSize,
								  const int resolutionScalar) {
	// Shared memory for the partial sums of the current block
	__local unsigned int partialSums[64];

	// Current entry to be computed by the thread
	int cx = get_global_id(0);
	int cy = get_global_id(1);
	cx = min(cx, lowDimX - 1);
	cy = min(cy, lowDimY - 1);
	const int cz = get_global_id(2);
	const int tIdx = get_local_id(1) * get_local_size(0) + get_local_id(0);
	const int layerOffset = cz * 2 * lowDimY * lowDimX; // Offset to index the layer of the current thread
	const int scaledCx = cx << resolutionScalar; // The X-Index of the current thread in the input frames
	const int scaledCy = cy << resolutionScalar; // The Y-Index of the current thread in the input frames
	const int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim

	// Calculate the image delta
	char offsetX = offsetArray[layerOffset + threadIndex2D];
	char offsetY = offsetArray[directionIndexOffset + layerOffset + threadIndex2D];
	int newCx = scaledCx + offsetX;
	int newCy = scaledCy + offsetY;

	// Window size of 1x1 (SHOULD BE REMOVED!)
	if (windowSize == 1) {
		summedUpDeltaArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] = (scaledCy < 0 || scaledCy >= dimY || scaledCx < 0 || scaledCx >= dimX || newCy < 0 || newCx < 0 || newCy >= dimY || newCx >= dimX) ? 0 : 
			abs((int)frame1[scaledCy * dimX + scaledCx] - (int)frame2[newCy * dimX + newCx]) + abs(offsetX) + abs(offsetY);
		return;
	// All other window sizes
	} else {
	partialSums[tIdx] = (scaledCy < 0 || scaledCy >= dimY || scaledCx < 0 || scaledCx >= dimX || newCy < 0 || newCx < 0 || newCy >= dimY || newCx >= dimX) ? 0 : 
		abs((int)frame1[scaledCy * dimX + scaledCx] - (int)frame2[newCy * dimX + newCx]) + abs(offsetX) + abs(offsetY);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// Sum up the remaining pixels for the current window
	for (int s = (get_local_size(1) * get_local_size(0)) >> 1; s > 32; s >>= 1) {
		if (tIdx < s) {
			partialSums[tIdx] += partialSums[tIdx + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Loop over the remaining values
	// Window size of 8x8 or larger
	if (windowSize >= 8) {
		if (tIdx < 32) {
			warpReduce8x8(partialSums, tIdx);
		}
	// Window size of 4x4
	} else if (windowSize == 4) {
		// Top 4x4 Blocks
		if (get_local_id(1) < 2) {
			warpReduce4x4(partialSums, tIdx);
		// Bottom 4x4 Blocks
		} else if (get_local_id(1) >= 4 && get_local_id(1) < 6) {
			warpReduce4x4(partialSums, tIdx);
		}
	// Window size of 2x2
	} else if (windowSize == 2) {
		if ((get_local_id(1) & 1) == 0) {
			warpReduce2x2(partialSums, tIdx);
		}
	}
	
	// Sync all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Sum up the results of all blocks
	if ((windowSize >= 8 && tIdx == 0) || 
		(windowSize == 4 && (tIdx == 0 || tIdx == 4 || tIdx == 32 || tIdx == 36)) || 
		(windowSize == 2 && ((tIdx & 1) == 0 && (get_local_id(1) & 1) == 0))) {
		const int windowIndexX = cx / windowSize;
		const int windowIndexY = cy / windowSize;
		atomic_add(&summedUpDeltaArray[cz * lowDimY * lowDimX + (windowIndexY * windowSize) * lowDimX + (windowIndexX * windowSize)], partialSums[tIdx]);
	}
}