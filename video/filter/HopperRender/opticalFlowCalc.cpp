#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "opticalFlowCalc.h"
#include <algorithm>

#define SCALE_FLOW 0

#define HIP_CHECK(call)                                              \
    {                                                                \
        hipError_t err = call;                                       \
        if (err != hipSuccess) {                                     \
            printf("HIP error at %s:%d code=%d (%s) \n", __FILE__, __LINE__, err, hipGetErrorString(err)); \
			exit(1);                                                 \
		}                                                            \
    }

struct priv {
	// HIP streams
	hipStream_t m_csOFCStream1, m_csOFCStream2; // HIP streams used for the optical flow calculation
	hipStream_t m_csWarpStream1, m_csWarpStream2; // HIP streams used for the warping

	// Grids
	dim3 m_lowGrid32x32x1;
	dim3 m_lowGrid16x16x5;
	dim3 m_lowGrid16x16x4;
	dim3 m_lowGrid16x16x1;
	dim3 m_lowGrid8x8x5;
	dim3 m_lowGrid8x8x1;
	dim3 m_grid16x16x1;
	dim3 m_halfGrid16x16x1;
	dim3 m_grid8x8x1;
	
	// Threads
	dim3 m_threads32x32x1;
	dim3 m_threads16x16x2;
	dim3 m_threads16x16x1;
	dim3 m_threads8x8x5;
	dim3 m_threads8x8x2;
	dim3 m_threads8x8x1;
};

// Applies the black and white values to the Y-Channel
__device__ void applyShaderY(unsigned short& output, const unsigned short input, const float blackLevel, const float whiteLevel, const float maxVal) {
	output = static_cast<unsigned short>(fmaxf(fminf((static_cast<float>(input) - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f));
}

// Kernel that runs a tearing test on the GPU
__global__ void tearingTestKernel(unsigned short* outputFrame, const unsigned int dimY, const unsigned int dimX, const int width, const int pos_x) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	if (cy < dimY && cx < dimX && cx >= pos_x && cx < pos_x + width) {
		outputFrame[cy * dimX + cx] = 65535;
	}
}

// Kernel that simply applies a shader to the frame and copies it to the output frame
__global__ void processFrameKernel(const unsigned short* frame, unsigned short* outputFrame,
                                 const unsigned int dimY,
                                 const unsigned int dimX, const float blackLevel, const float whiteLevel, const float maxVal) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Y Channel
	if (cy < dimY && cx < dimX) {
		applyShaderY(outputFrame[cy * dimX + cx], frame[cy * dimX + cx], blackLevel, whiteLevel, maxVal);
	}
}

// Kernel that blurs a 2D plane along the X direction
__global__ void blurFrameKernel(const unsigned short* frameArray, unsigned short* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const unsigned short dimY, const unsigned short dimX) {
	// Shared memory for the frame to prevent multiple global memory accesses
	extern __shared__ unsigned int sharedFrameArray[];
	// Current entry to be computed by the thread
	const unsigned short cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned short cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the current thread is supposed to perform calculations
	if (cy >= dimY || cx >= dimX) {
		return;
	}

	const unsigned short trX = blockIdx.x * blockDim.x;
	const unsigned short trY = blockIdx.y * blockDim.y;
	unsigned char offsetX;
	unsigned char offsetY;
	int newX;
	int newY;
    // Calculate the number of entries to fill for this thread
    const unsigned short threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned char entriesToFill = avgEntriesPerThread + (threadIndex < remainder ? 1 : 0);
    // Calculate the starting index for this thread
    unsigned short startIndex = 0;
    for (unsigned short i = 0; i < threadIndex; ++i) {
        startIndex += avgEntriesPerThread + (i < remainder ? 1 : 0);
    }
    // Fill the shared memory for this thread
    for (unsigned short i = 0; i < entriesToFill; ++i) {
		offsetX = (startIndex + i) % chacheSize;
		offsetY = (startIndex + i) / chacheSize;
		newX = trX - boundsOffset + offsetX;
		newY = trY - boundsOffset + offsetY;
		if (newY < dimY && newY >= 0 && newX < dimX && newX >= 0) {
			sharedFrameArray[startIndex + i] = frameArray[newY * dimX + newX];
		} else {
			sharedFrameArray[startIndex + i] = 0;
		}
	}

    // Ensure all threads have finished loading before continuing
    __syncthreads();

	// Don't blur the edges of the frame
	if (cy < kernelSize / 2 || cy >= dimY - kernelSize / 2 || cx < kernelSize / 2 || cx >= dimX - kernelSize / 2) {
		blurredFrameArray[cy * dimX + cx] = 0;
		return;
	}

	unsigned int blurredPixel = 0;
	// Collect the sum of the surrounding pixels
	for (char y = lumStart; y < lumEnd; y++) {
		for (char x = lumStart; x < lumEnd; x++) {
			if ((cy + y) < dimY && (cy + y) >= 0 && (cx + x) < dimX && (cx + x) >= 0) {
				blurredPixel += sharedFrameArray[(threadIdx.y + boundsOffset + y) * chacheSize + threadIdx.x + boundsOffset + x];
			} else {
				blurredPixel += sharedFrameArray[(threadIdx.y + boundsOffset) * chacheSize + threadIdx.x + boundsOffset];
			}
		}
	}
	blurredPixel /= lumPixelCount;
	blurredFrameArray[cy * dimX + cx] = blurredPixel;
}

// Kernel that sets the initial offset array
__global__ void setInitialOffset(int* offsetArray, const unsigned int numLayers, const unsigned int lowDimY, 
								 const unsigned int lowDimX, const unsigned int layerIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z;

	if (cy < lowDimY && cx < lowDimX) {
		switch (cz) {
			// Set the X direction layer 1 to a -2 offset
			case 0:
				offsetArray[layerIdxOffset + cy * lowDimX + cx] = -2;
				return;
			// Set the X direction layer 2 to a -1 offset
			case 1:
				offsetArray[2 * layerIdxOffset + cy * lowDimX + cx] = -1;
				return;
			// Set the X direction layer 3 to a +1 offset
			case 2:
				offsetArray[3 * layerIdxOffset + cy * lowDimX + cx] = 1;
				return;
			// Set the X direction layer 4 to a +2 offset
			case 3:
				offsetArray[4 * layerIdxOffset + cy * lowDimX + cx] = 2;
				return;
		}
	}
}

// Helper kernel for the calcDeltaSums kernel
__device__ void warpReduce8x8(volatile unsigned int* partial_sums, int tIdx) {
	partial_sums[tIdx] += partial_sums[tIdx + 32];
	partial_sums[tIdx] += partial_sums[tIdx + 16];
	partial_sums[tIdx] += partial_sums[tIdx + 8];
	partial_sums[tIdx] += partial_sums[tIdx + 4];
	partial_sums[tIdx] += partial_sums[tIdx + 2];
	partial_sums[tIdx] += partial_sums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
__device__ void warpReduce4x4(volatile unsigned int* partial_sums, int tIdx) {
	partial_sums[tIdx] += partial_sums[tIdx + 16];
	partial_sums[tIdx] += partial_sums[tIdx + 8];
	partial_sums[tIdx] += partial_sums[tIdx + 2];
	partial_sums[tIdx] += partial_sums[tIdx + 1];
}

// Helper kernel for the calcDeltaSums kernel
__device__ void warpReduce2x2(volatile unsigned int* partial_sums, int tIdx) {
	partial_sums[tIdx] += partial_sums[tIdx + 8];
	partial_sums[tIdx] += partial_sums[tIdx + 1];
}

// Kernel that sums up all the pixel deltas of each window
__global__ void calcDeltaSums(unsigned int* summedUpDeltaArray, const unsigned short* frame1, const unsigned short* frame2,
							  const int* offsetArray, const unsigned int layerIdxOffset, const unsigned int directionIdxOffset,
						      const unsigned int dimY, const unsigned int dimX, const unsigned int lowDimY, const unsigned int lowDimX,
							  const unsigned int windowDim, const unsigned char resolutionScalar) {
	// Handle used to synchronize all threads
	auto g = cooperative_groups::this_thread_block();

	// Shared memory for the partial sums of the current block
	extern __shared__ unsigned int partial_sums[];

	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = blockIdx.z;
	const unsigned int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned int layerOffset = blockIdx.z * lowDimY * lowDimX; // Offset to index the layer of the current thread
	const unsigned int scaledCx = cx << resolutionScalar; // The X-Index of the current thread in the input frames
	const unsigned int scaledCy = cy << resolutionScalar; // The Y-Index of the current thread in the input frames
	const unsigned int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim

	if (cy < lowDimY && cx < lowDimX) {
		// Calculate the image delta
		int offsetX = -offsetArray[layerOffset + threadIndex2D];
		int offsetY = -offsetArray[directionIdxOffset + layerOffset + threadIndex2D];
		int newCx = scaledCx + (offsetX * 1 << (resolutionScalar * SCALE_FLOW));
		int newCy = scaledCy + (offsetY * 1 << (resolutionScalar * SCALE_FLOW));

		// Window size of 1x1
		if (windowDim == 1) {
			summedUpDeltaArray[cz * layerIdxOffset + cy * lowDimX + cx] = (newCy < 0 || newCx < 0 || newCy >= dimY || newCx >= dimX) ? 0 : 
				abs(frame1[newCy * dimX + newCx] - frame2[scaledCy * dimX + scaledCx]);
		    return;
		// All other window sizes
		} else {
			partial_sums[tIdx] = (newCy < 0 || newCx < 0 || newCy >= dimY || newCx >= dimX) ? 0 : 
				abs(frame1[newCy * dimX + newCx] - frame2[scaledCy * dimX + scaledCx]);
		}
		
		__syncthreads();

		// Sum up the remaining pixels for the current window
		for (int s = (blockDim.y * blockDim.x) >> 1; s > 32; s >>= 1) {
			if (tIdx < s) {
				partial_sums[tIdx] += partial_sums[tIdx + s];
			}
			__syncthreads();
		}

		// Loop over the remaining values
		// Window size of 8x8 or larger
		if (windowDim >= 8) {
			if (tIdx < 32) {
				warpReduce8x8(partial_sums, tIdx);
			}
		// Window size of 4x4
		} else if (windowDim == 4) {
			// Top 4x4 Blocks
			if (threadIdx.y < 2) {
				warpReduce4x4(partial_sums, tIdx);
			// Bottom 4x4 Blocks
			} else if (threadIdx.y >= 4 && threadIdx.y < 6) {
				warpReduce4x4(partial_sums, tIdx);
			}
		// Window size of 2x2
		} else if (windowDim == 2) {
			if ((threadIdx.y & 1) == 0) {
				warpReduce2x2(partial_sums, tIdx);
			}
		}
		
		// Sync all threads
		g.sync();

		// Sum up the results of all blocks
		if ((windowDim >= 8 && tIdx == 0) || 
			(windowDim == 4 && (tIdx == 0 || tIdx == 4 || tIdx == 32 || tIdx == 36)) || 
			(windowDim == 2 && ((tIdx & 1) == 0 && (threadIdx.y & 1) == 0))) {
			const unsigned int windowIndexX = cx / windowDim;
			const unsigned int windowIndexY = cy / windowDim;
			atomicAdd(&summedUpDeltaArray[cz * layerIdxOffset + (windowIndexY * windowDim) * lowDimX + (windowIndexX * windowDim)], partial_sums[tIdx]);
		}
	}
}

// Kernel that normalizes all the pixel deltas of each window
__global__ void normalizeDeltaSums(const unsigned int* summedUpDeltaArray, unsigned char* globalLowestLayerArray,
                                   const int* offsetArray, const unsigned int windowDim, int numPixels,
								   const unsigned int directionIdxOffset, const unsigned int layerIdxOffset, 
								   const unsigned int numLayers, const unsigned int lowDimY, const unsigned int lowDimX) {
	// Allocate shared memory to share values across layers
	__shared__ double normalizedDeltaArray[5 * 8 * 8 * 8];
	
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim
	bool isWindowRepresent = cy % windowDim == 0 && cx % windowDim == 0;

	// Check if the thread is a window represent
	if (isWindowRepresent) {
		// Get the current window information
		const int offsetX = offsetArray[cz * layerIdxOffset + threadIndex2D];
		const int offsetY = offsetArray[directionIdxOffset + cz * layerIdxOffset + threadIndex2D];

		// Calculate the not overlapping pixels
		int overlapX;
		int overlapY;

		// Calculate the number of not overlapping pixels
		if ((cx + windowDim + abs(offsetX) > lowDimX) || (cx - offsetX < 0)) {
			overlapX = abs(offsetX);
		} else {
			overlapX = 0;
		}

		if ((cy + windowDim + abs(offsetY) > lowDimY) || (cy - offsetY < 0)) {
			overlapY = abs(offsetY);
		} else {
			overlapY = 0;
		}

		const int numNotOverlappingPixels = overlapY * overlapX;
		numPixels -= numNotOverlappingPixels;
		numPixels = max(numPixels, 1);

		// Normalize the summed up delta
		normalizedDeltaArray[cz * 8 * 8 + threadIdx.y * 8 + threadIdx.x] = static_cast<double>(summedUpDeltaArray[cz * layerIdxOffset + threadIndex2D]) / static_cast<double>(numPixels);
	}

	// Wait for all threads to finish filling the values
	__syncthreads();

	// Find the layer with the lowest value
	if (cz == 0 && isWindowRepresent) {
		unsigned char lowestLayer = 0;

		for (unsigned char z = 1; z < numLayers; ++z) {
			if (normalizedDeltaArray[z * 8 * 8 + threadIdx.y * 8 + threadIdx.x] < 
				normalizedDeltaArray[lowestLayer * 8 * 8 + threadIdx.y * 8 + threadIdx.x]) {
				lowestLayer = z;
			}
		}

		globalLowestLayerArray[threadIndex2D] = lowestLayer;
	}
}

// Kernel that adjusts the offset array based on the comparison results
__global__ void adjustOffsetArray(int* offsetArray, const unsigned char* globalLowestLayerArray, unsigned char* statusArray,
								  const unsigned int windowDim, const unsigned int directionIdxOffset, const unsigned int layerIdxOffset, 
								  const unsigned int numLayers, const unsigned int lowDimY, const unsigned int lowDimX, const bool lastRun) {

	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim

	/*
	* Status Array Key:
	* 0: Find the initial x direction
	* 1: Find extended positive x direction
	* 2: Find extended negative x direction
	* 3: Find the initial y direction
	* 4: Find extended positive y direction
	* 5: Find extended negative y direction
	* 6: Search complete
	*/

	if (cy < lowDimY && cx < lowDimX) {
		const unsigned char currentStatus = statusArray[threadIndex2D];

		// We are done searching and we only need to do cleanup on the last run, so we exit here
		if (currentStatus == 6 && !lastRun) {
			return;
		}

		// We only need the lowestLayer if we are still searching
		unsigned char lowestLayer = 0;
		if (currentStatus != 6) {
			const unsigned int wx = (cx / windowDim) * windowDim;
			const unsigned int wy = (cy / windowDim) * windowDim;
			lowestLayer = globalLowestLayerArray[wy * lowDimX + wx];
		}

		int currX;
		int currY;

		// If this is the last run, we need to adjust the offset array for the next iteration
		if (lastRun) {
			switch (currentStatus) {
				// We are still trying to find the perfect x direction
				case 0:
				case 1:
				case 2:
					currX = offsetArray[lowestLayer * layerIdxOffset + threadIndex2D];

					// Shift the X direction layer 0 to the ideal X direction
					offsetArray[threadIndex2D] = currX;
					// Shift the X direction layer 1 by -2
					offsetArray[layerIdxOffset + threadIndex2D] = currX - 2;
					// Shift the X direction layer 2 by -1
					offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 1;
					// Shift the X direction layer 3 by +1
					offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
					// Shift the X direction layer 4 by +2
					offsetArray[4 * layerIdxOffset + threadIndex2D] = currX + 2;
					return;

				// We are still trying to find the perfect y direction
				case 3:
				case 4:
				case 5:
					currX = offsetArray[threadIndex2D];
					currY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];

					// Set all Y direction layers to the ideal Y direction
					for (unsigned int z = 0; z < numLayers; z++) {
						offsetArray[directionIdxOffset + z * layerIdxOffset + threadIndex2D] = currY;
					}

					// Shift the X direction layer 1 by -2
					offsetArray[layerIdxOffset + threadIndex2D] = currX - 2;
					// Shift the X direction layer 2 by -1
					offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 1;
					// Shift the X direction layer 3 by +1
					offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
					// Shift the X direction layer 4 by +2
					offsetArray[4 * layerIdxOffset + threadIndex2D] = currX + 2;
					return;

				// Search completed, so we adjust the offset array for the next run
				default:
					currX = offsetArray[threadIndex2D];
					currY = offsetArray[directionIdxOffset + threadIndex2D];

					// Set all Y direction layers to the ideal Y direction
					for (unsigned int z = 1; z < numLayers; z++) {
						offsetArray[directionIdxOffset + z * layerIdxOffset + threadIndex2D] = currY;
					}

					// Shift the X direction layer 1 by -2
					offsetArray[layerIdxOffset + threadIndex2D] = currX - 2;
					// Shift the X direction layer 2 by -1
					offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 1;
					// Shift the X direction layer 3 by +1
					offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
					// Shift the X direction layer 4 by +2
					offsetArray[4 * layerIdxOffset + threadIndex2D] = currX + 2;
					return;
			}
			return;
		}

		// If we are still calculating, adjust the offset array based on the current status and lowest layer
		int idealX;
		int idealY;
		switch (currentStatus) {
			/*
			* X - DIRECTION
			*/
			// Find the initial x direction
			case 0:
				switch (lowestLayer) {
					// If the lowest layer is 0, no x direction is needed -> continue to y direction
					case 0:
						statusArray[threadIndex2D] = 3;
						currX = offsetArray[threadIndex2D];
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						for (int z = 0; z < numLayers; z++) {
							offsetArray[z * layerIdxOffset + threadIndex2D] = currX;
						}
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;
					
					// If the lowest layer is 1 -> continue moving in the negative x direction
					case 1:
						statusArray[threadIndex2D] = 2;
						currX = offsetArray[layerIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX - 1;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX - 3;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX - 4;
						return;

					// If the lowest layer is 2, ideal x direction found -> continue to y direction
					case 2:
						statusArray[threadIndex2D] = 3;
						currX = offsetArray[2 * layerIdxOffset + threadIndex2D];
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;

					// If the lowest layer is 3, ideal x direction found -> continue to y direction
					case 3:
						statusArray[threadIndex2D] = 3;
						currX = offsetArray[3 * layerIdxOffset + threadIndex2D];
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;

					// If the lowest layer is 4 -> continue moving in the positive x direction
					case 4:
						statusArray[threadIndex2D] = 1;
						currX = offsetArray[4 * layerIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX + 4;
						offsetArray[layerIdxOffset + threadIndex2D] = currX + 3;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX + 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
						return;
				}
				return;

			// Find extended positive x direction
			case 1:
				switch (lowestLayer) {
					// If the lowest layer is 0 -> continue moving in x direction
					case 0:
						currX = offsetArray[threadIndex2D];
						offsetArray[threadIndex2D] = currX + 4;
						offsetArray[layerIdxOffset + threadIndex2D] = currX + 3;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX + 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX + 1;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX;
						return;

					// If the lowest layer is not 0, no x further direction is needed -> continue to y direction
					default:
						statusArray[threadIndex2D] = 3;
						idealX = offsetArray[lowestLayer * layerIdxOffset + threadIndex2D];
						for (unsigned int z = 0; z < numLayers; z++) {
							offsetArray[z * layerIdxOffset + threadIndex2D] = idealX;
						}
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;
				}
				return;

			// Find extended negative x direction
			case 2:
				switch (lowestLayer) {
					// If the lowest layer is not 4, no x further direction is needed -> continue to y direction
					case 0:
					case 1:
					case 2:
					case 3:
						statusArray[threadIndex2D] = 3;
						idealX = offsetArray[lowestLayer * layerIdxOffset + threadIndex2D];
						for (unsigned int z = 0; z < numLayers; z++) {
							offsetArray[z * layerIdxOffset + threadIndex2D] = idealX;
						}
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY + 2;
						return;

					// If the lowest layer is 4 -> continue moving in x direction
					case 4:
						currX = offsetArray[4 * layerIdxOffset + threadIndex2D];
						offsetArray[threadIndex2D] = currX;
						offsetArray[layerIdxOffset + threadIndex2D] = currX - 1;
						offsetArray[2 * layerIdxOffset + threadIndex2D] = currX - 2;
						offsetArray[3 * layerIdxOffset + threadIndex2D] = currX - 3;
						offsetArray[4 * layerIdxOffset + threadIndex2D] = currX - 4;
						return;
				}
				return;

			/*
			* Y - DIRECTION
			*/
			// Find the initial y direction
			case 3:
				switch (lowestLayer) {
					// If the lowest layer is 0, 2, or 3, no y direction is needed -> we are done
					case 0:
					case 2:
					case 3:
						statusArray[threadIndex2D] = 6;
						if (lowestLayer != 0) {
							currY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];
							offsetArray[directionIdxOffset + threadIndex2D] = currY;
						}
						return;

					// If the lowest layer is 1 -> continue moving in the negative y direction
					case 1:
						statusArray[threadIndex2D] = 5;
						currY = offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY - 3;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY - 4;
						return;

					// If the lowest layer is 4 -> continue moving in the positive y direction
					case 4:
						statusArray[threadIndex2D] = 4;
						currY = offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY + 4;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY + 3;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY + 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						return;
				}
				return;

			// Find extended positive y direction
			case 4:
				switch (lowestLayer) {
					// If the lowest layer is 0 -> continue moving in y direction
					case 0:
						currY = offsetArray[directionIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY + 4;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY + 3;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY + 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY + 1;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY;
						return;

					// If the lowest layer is not 0, no y further direction is needed -> we are done
					default:
						statusArray[threadIndex2D] = 6;
						idealY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = idealY;
						return;
				}
				return;

			// Find extended negative y direction
			case 5:
				switch (lowestLayer) {
					// If the lowest layer is not 4, no y further direction is needed -> we are done
					case 0:
					case 1:
					case 2:
					case 3:
						statusArray[threadIndex2D] = 6;
						idealY = offsetArray[directionIdxOffset + lowestLayer * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = idealY;
						return;

					// If the lowest layer is 4 -> continue moving in y direction
					case 4:
						currY = offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D];
						offsetArray[directionIdxOffset + threadIndex2D] = currY;
						offsetArray[directionIdxOffset + layerIdxOffset + threadIndex2D] = currY - 1;
						offsetArray[directionIdxOffset + 2 * layerIdxOffset + threadIndex2D] = currY - 2;
						offsetArray[directionIdxOffset + 3 * layerIdxOffset + threadIndex2D] = currY - 3;
						offsetArray[directionIdxOffset + 4 * layerIdxOffset + threadIndex2D] = currY - 4;
						return;
				}
				return;

			// Search is complete
			default:
				return;
		}
	}
}

// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__global__ void flipFlowKernel(const int* flowArray12, int* flowArray21, const int lowDimY, const int lowDimX, 
							   const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
							   const unsigned int layerIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	// Check if we are inside the flow array
	if (cy < lowDimY && cx < lowDimX) {
		// Get the current flow values
		const int x = flowArray12[cy * lowDimX + cx];
		const int y = flowArray12[directionIdxOffset + cy * lowDimX + cx];
		const int scaledX = x >> resolutionScalar;
		const int scaledY = y >> resolutionScalar;

		// Project the flow values onto the flow array from frame 2 to frame 1
		// X-Layer
		if (cz == 0 && (cy + scaledY) < lowDimY && (cy + scaledY) >= 0 && (cx + scaledX) < lowDimX && (cx + scaledX) >= 0) {
			flowArray21[(cy + scaledY) * lowDimX + cx + scaledX] = -x;
		// Y-Layer
		} else if (cz == 1 && (cy + scaledY) < lowDimY && (cy + scaledY) >= 0 && (cx + scaledX) < lowDimX && (cx + scaledX) >= 0) {
			flowArray21[layerIdxOffset + (cy + scaledY) * lowDimX + cx + scaledX] = -y;
		}
	}
}

// Kernel that blurs a flow array
__global__ void blurFlowKernel(const int* flowArray, int* blurredFlowArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char start,
								const unsigned char end, const unsigned short pixelCount, const unsigned short numLayers,
								const unsigned short lowDimY, const unsigned short lowDimX) {
	// Shared memory for the flow to prevent multiple global memory accesses
	extern __shared__ int sharedFlowArray[];

	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = blockIdx.z;

	// Current threadblock index
	const int trX = blockIdx.x * blockDim.x;
	const int trY = blockIdx.y * blockDim.y;
	unsigned char offsetX;
	unsigned char offsetY;
	int newX;
	int newY;

    // Calculate the number of entries to fill for this thread
    const unsigned short threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned char entriesToFill = avgEntriesPerThread + (threadIndex < remainder ? 1 : 0);

    // Calculate the starting index for this thread
    unsigned short startIndex = 0;
    for (unsigned short i = 0; i < threadIndex; ++i) {
        startIndex += avgEntriesPerThread + (i < remainder ? 1 : 0);
    }

    // Fill the shared memory for this thread
    for (unsigned short i = 0; i < entriesToFill; ++i) {
		offsetX = (startIndex + i) % chacheSize;
		offsetY = (startIndex + i) / chacheSize;
		newX = trX - boundsOffset + offsetX;
		newY = trY - boundsOffset + offsetY;
		if (newY < lowDimY && newY >= 0 && newX < lowDimX && newX >= 0) {
			sharedFlowArray[startIndex + i] = flowArray[cz * numLayers * lowDimY * lowDimX + newY * lowDimX + newX];
		} else {
			sharedFlowArray[startIndex + i] = 0;
		}
	}

    // Ensure all threads have finished loading before continuing
    __syncthreads();

	// Check if we are inside the flow array
	if (cy < lowDimY && cy >= 0 && cx < lowDimX && cx >= 0) {
		// Calculate the x and y boundaries of the kernel
		int blurredOffset = 0;

		// Collect the sum of the surrounding values
		for (char y = start; y < end; y++) {
			for (char x = start; x < end; x++) {
				if ((cy + y) < lowDimY && (cy + y) >= 0 && (cx + x) < lowDimX && (cx + x) >= 0) {
					blurredOffset += sharedFlowArray[(threadIdx.y + boundsOffset + y) * chacheSize + threadIdx.x + boundsOffset + x];
				} else {
					blurredOffset += sharedFlowArray[(threadIdx.y + boundsOffset) * chacheSize + threadIdx.x + boundsOffset];
				}
			}
		}
		blurredOffset /= pixelCount;
		blurredFlowArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] = blurredOffset;
	}

}

// Kernel that cleans a flow array
__global__ void cleanFlowKernel(const int* flowArray, int* blurredFlowArray, 
								const unsigned short lowDimY, const unsigned short lowDimX) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = threadIdx.z;

	int offsetX = flowArray[cy * lowDimX + cx];
	int offsetY = flowArray[lowDimY * lowDimX + cy * lowDimX + cx];

    if (abs(offsetY) <= 2 && abs(offsetX <= 2)) {
		for (int y = -6; y < 6; y++) {
			for (int x = -6; x < 6; x++) {
				if ((cy + y) < lowDimY && (cy + y) >= 0 && (cx + x) < lowDimX && (cx + x) >= 0) {
					blurredFlowArray[cz * lowDimY * lowDimX + (cy + y) * lowDimX + (cx + x)] = flowArray[cz * lowDimY * lowDimX + cy * lowDimX + cx];
				}
			}
		}
		//blurredFlowArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] = 0;
	}
}

// Kernel that warps a frame according to the offset array
__global__ void warpFrameKernel(const unsigned short* frame1, const int* offsetArray, int* hitCount,
								unsigned short* warpedFrame, const float frameScalar, const unsigned int lowDimY, const unsigned int lowDimX,
								const unsigned int dimY, const int dimX, const unsigned char resolutionScalar,
								const unsigned int directionIdxOffset, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		// Get the current offsets to use
		const int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
		const int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar) ;
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar) ;
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < dimX) {
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], 1);
		}

	// U/V-Channel
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		const int scaledCx = (cx >> resolutionScalar) & ~1; // The X-Index of the current thread in the offset array
		const int scaledCy = (cy >> resolutionScalar) << 1; // The Y-Index of the current thread in the offset array
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar) >> 1;
		
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

// Kernel that removes artifacts from the warped frame
__global__ void artifactRemovalKernel(const unsigned short* frame1, const int* hitCount, unsigned short* warpedFrame,
												 const unsigned int dimY, const unsigned int dimX, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int threadIndex2D = cy * dimX + cx; // Standard thread index without Z-Dim

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

// Kernel that blends warpedFrame1 to warpedFrame2
__global__ void blendFrameKernel(const unsigned short* warpedFrame1, const unsigned short* warpedFrame2, unsigned short* outputFrame,
                                 const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                 const unsigned int dimX, const unsigned int channelIdxOffset, const float blackLevel, const float whiteLevel, const float maxVal) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	float pixelValue;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		pixelValue = static_cast<float>(warpedFrame1[cy * dimX + cx]) * frame1Scalar + 
					 static_cast<float>(warpedFrame2[cy * dimX + cx]) * frame2Scalar;
		applyShaderY(outputFrame[cy * dimX + cx], pixelValue, blackLevel, whiteLevel, maxVal);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		pixelValue = static_cast<float>(warpedFrame1[channelIdxOffset + cy * dimX + cx]) * frame1Scalar + 
					 static_cast<float>(warpedFrame2[channelIdxOffset + cy * dimX + cx]) * frame2Scalar;
		outputFrame[channelIdxOffset + cy * dimX + cx] = pixelValue;
	}
}

// Kernel that places half of frame 1 over the outputFrame
__global__ void insertFrameKernel(const unsigned short* frame1, unsigned short* outputFrame, const unsigned int dimY,
                                  const unsigned int dimX,
								  const unsigned int channelIdxOffset, const float blackLevel, 
								  const float whiteLevel, const float maxVal) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < (dimX >> 1)) {
		applyShaderY(outputFrame[cy * dimX + cx], frame1[cy * dimX + cx], blackLevel, whiteLevel, maxVal);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < (dimX >> 1)) {
		outputFrame[channelIdxOffset + cy * dimX + cx] = frame1[channelIdxOffset + cy * dimX + cx];
	}
}

// Kernel that places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
__global__ void sideBySideFrameKernel(const unsigned short* frame1, const unsigned short* warpedFrame1, const unsigned short* warpedFrame2, unsigned short* outputFrame, 
									  const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                      const unsigned int dimX, const unsigned int halfDimY, 
									  const unsigned int halfDimX, const unsigned int channelIdxOffset,
									  const float blackLevel, const float whiteLevel, const float maxVal, const unsigned short middleValue) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int verticalOffset = dimY >> 2;
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
			static_cast<unsigned short>(
				static_cast<float>(warpedFrame1[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) * frame1Scalar + 
				static_cast<float>(warpedFrame2[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) * frame2Scalar
			);
	// U/V Channels
	} else if (isUVChannel && isInRightSideUV) {
		blendedFrameValue = 
			static_cast<unsigned short>(
				static_cast<float>(warpedFrame1[channelIdxOffset + 2 * (cy - (verticalOffset >> 1)) * dimX + ((cx - halfDimX) << 1) + isVChannel]) * frame1Scalar + 
				static_cast<float>(warpedFrame2[channelIdxOffset + 2 * (cy - (verticalOffset >> 1)) * dimX + ((cx - halfDimX) << 1) + isVChannel]) * frame2Scalar
			);
	}

	// --- Insertion ---
	if (isYChannel) {
		// Y Channel Left Side
		if (isInLeftSideY) {
			applyShaderY(outputFrame[cy * dimX + cx], static_cast<unsigned short>(frame1[((cy - verticalOffset) << 1) * dimX + (cx << 1)]), blackLevel, whiteLevel, maxVal);
		// Y Channel Right Side
		} else if (isInRightSideY) {
			applyShaderY(outputFrame[cy * dimX + cx], blendedFrameValue, blackLevel, whiteLevel, maxVal);
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

// Kernel that creates an HSV flow image from the offset array
__global__ void convertFlowToHSVKernel(const int* flowArray, unsigned short* outputFrame, const unsigned short* frame1,
                                       const float blendScalar, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX, 
									   const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	const unsigned int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
	const unsigned int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array

	// Get the current flow values
	float x;
	float y;
	if (cz == 0 && cy < dimY && cx < dimX) {
		x = flowArray[scaledCy * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + scaledCy * lowDimX + scaledCx];
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX){
		x = flowArray[(scaledCy << 1) * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + (scaledCy << 1) * lowDimX + scaledCx];
	}

	// RGB struct
	struct RGB {
		unsigned char r, g, b;
	};

	// Calculate the angle in radians
	const float angle_rad = std::atan2(y, x);

	// Convert radians to degrees
	float angle_deg = angle_rad * (180.0f / 3.14159265359f);

	// Ensure the angle is positive
	if (angle_deg < 0) {
		angle_deg += 360.0f;
	}

	// Normalize the angle to the range [0, 360]
	angle_deg = fmodf(angle_deg, 360.0f);
	if (angle_deg < 0) {
		angle_deg += 360.0f;
	}

	// Map the angle to the hue value in the HSV model
	const float hue = angle_deg / 360.0f;

	// Convert HSV to RGB
	const int h_i = static_cast<int>(hue * 6.0f);
	const float f = hue * 6.0f - h_i;
	const float q = 1.0f - f;

	RGB rgb;
	switch (h_i % 6) {
		case 0: rgb = { 255, static_cast<unsigned char>(f * 255.0f), 0 }; break; // Red - Yellow
		case 1: rgb = { static_cast<unsigned char>(q * 255.0f), 255, 0 }; break; // Yellow - Green
		case 2: rgb = { 0, 255, static_cast<unsigned char>(f * 255.0f) }; break; // Green - Cyan
		case 3: rgb = { 0, static_cast<unsigned char>(q * 255.0f), 255 }; break; // Cyan - Blue
		case 4: rgb = { static_cast<unsigned char>(f * 255.0f), 0, 255 }; break; // Blue - Magenta
		case 5: rgb = { 255, 0, static_cast<unsigned char>(q * 255.0f) }; break; // Magenta - Red
		default: rgb = { 0, 0, 0 }; break;
	}

	// Adjust the color intensity based on the flow magnitude
	rgb.r = fmaxf(fminf((float)rgb.r / 255.0f * (fabsf(x) + fabsf(y)) * 4.0f, 255.0f), 0.0f);
	rgb.g = fmaxf(fminf((float)rgb.g / 255.0f * fabsf(y) * 8.0f, 255.0f), 0.0f);
	rgb.b = fmaxf(fminf((float)rgb.b / 255.0f * (fabsf(x) + fabsf(y)) * 4.0f, 255.0f), 0.0f);

	// Prevent random colors when there is no flow
	if (fabsf(x) < 1.0f && fabsf(y) < 1.0f) {
		rgb = { 0, 0, 0 };
	}

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		outputFrame[cy * dimX + cx] = (static_cast<unsigned short>(
				(fmaxf(fminf(rgb.r * 0.299f + rgb.g * 0.587f + rgb.b * 0.114f, 255.0f), 0.0f)) * blendScalar) << 8) + 
				frame1[cy * dimX + cx] * (1.0f - blendScalar);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		// U Channel
		if ((cx & 1) == 0) {
			outputFrame[channelIdxOffset + cy * dimX + (cx & ~1)] = static_cast<unsigned short>(
						fmaxf(fminf(rgb.r * -0.168736f + rgb.g * -0.331264f + rgb.b * 0.5f + 128.0f, 255.0f), 0.0f)) << 8;
		// V Channel
		} else {
			outputFrame[channelIdxOffset + cy * dimX + (cx & ~1) + 1] = static_cast<unsigned short>(
						fmaxf(fminf(rgb.r * 0.5f + rgb.g * -0.418688f + rgb.b * -0.081312f + 128.0f, 255.0f), 0.0f)) << 8;
		}
	}
}

// Kernel that creates an greyscale flow image from the offset array
__global__ void convertFlowToGreyscaleKernel(const int* flowArray, unsigned short* outputFrame, const unsigned short* frame1,
                                       const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX, 
									   const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset, const int maxVal, const unsigned short middleValue) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	const unsigned int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
	const unsigned int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array

	// Get the current flow values
	int x;
	int y;
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

/*
* Frees the memory of the optical flow calculator
*
* @param ofc: Pointer to the optical flow calculator
*/
void free(struct OpticalFlowCalc *ofc) {
	struct priv *priv = (struct priv*)ofc->priv;
	HIP_CHECK(hipFree(ofc->m_frame[0]));
	HIP_CHECK(hipFree(ofc->m_frame[1]));
	HIP_CHECK(hipFree(ofc->m_frame[2]));
	HIP_CHECK(hipFree(ofc->m_blurredFrame[0]));
	HIP_CHECK(hipFree(ofc->m_blurredFrame[1]));
	HIP_CHECK(hipFree(ofc->m_blurredFrame[2]));
	HIP_CHECK(hipFree(ofc->m_warpedFrame12));
	HIP_CHECK(hipFree(ofc->m_warpedFrame21));
	HIP_CHECK(hipFree(ofc->m_outputFrame));
	HIP_CHECK(hipFree(ofc->m_offsetArray12));
	HIP_CHECK(hipFree(ofc->m_offsetArray21));
	HIP_CHECK(hipFree(ofc->m_blurredOffsetArray12[0]));
	HIP_CHECK(hipFree(ofc->m_blurredOffsetArray21[0]));
	HIP_CHECK(hipFree(ofc->m_blurredOffsetArray12[1]));
	HIP_CHECK(hipFree(ofc->m_blurredOffsetArray21[1]));
	HIP_CHECK(hipFree(ofc->m_statusArray));
	HIP_CHECK(hipFree(ofc->m_summedUpDeltaArray));
	HIP_CHECK(hipFree(ofc->m_lowestLayerArray));
	HIP_CHECK(hipFree(ofc->m_hitCount12));
	HIP_CHECK(hipFree(ofc->m_hitCount21));

	HIP_CHECK(hipStreamDestroy(priv->m_csOFCStream1));
	HIP_CHECK(hipStreamDestroy(priv->m_csOFCStream2));
	HIP_CHECK(hipStreamDestroy(priv->m_csWarpStream1));
	HIP_CHECK(hipStreamDestroy(priv->m_csWarpStream2));
}

/*
* Checks if there are any recent HIP errors and prints the error if there is one.
*
* @param functionName: Name of the function that called this function
*/
void checkHIPError(const char* functionName) {
	hipError_t hipError_t = hipGetLastError();
	if (hipError_t != hipSuccess) {
		printf("HIP Error in function %s: %s\n", functionName, hipGetErrorString(hipError_t));
		exit(1);
	}
}

/*
* Readjusts internal structs for the new resolution
*
* @param ofc: Pointer to the optical flow calculator
*/
void adjustFrameScalar(struct OpticalFlowCalc *ofc) {
	struct priv *priv = (struct priv*)ofc->priv;
	priv->m_lowGrid32x32x1.x = (int)(fmax(ceil((double)(ofc->m_iLowDimX) / 32.0), 1.0));
	priv->m_lowGrid32x32x1.y = (int)(fmax(ceil((double)(ofc->m_iLowDimY) / 32.0), 1.0));
	priv->m_lowGrid16x16x5.x = (int)(fmax(ceil((double)(ofc->m_iLowDimX) / 16.0), 1.0));
	priv->m_lowGrid16x16x5.y = (int)(fmax(ceil((double)(ofc->m_iLowDimY) / 16.0), 1.0));
	priv->m_lowGrid16x16x4.x = (int)(fmax(ceil((double)(ofc->m_iLowDimX) / 16.0), 1.0));
	priv->m_lowGrid16x16x4.y = (int)(fmax(ceil((double)(ofc->m_iLowDimY) / 16.0), 1.0));
	priv->m_lowGrid16x16x1.x = (int)(fmax(ceil((double)(ofc->m_iLowDimX) / 16.0), 1.0));
	priv->m_lowGrid16x16x1.y = (int)(fmax(ceil((double)(ofc->m_iLowDimY) / 16.0), 1.0));
	priv->m_lowGrid8x8x5.x = (int)(fmax(ceil((double)(ofc->m_iLowDimX) / 8.0), 1.0));
	priv->m_lowGrid8x8x5.y = (int)(fmax(ceil((double)(ofc->m_iLowDimY) / 8.0), 1.0));
	priv->m_lowGrid8x8x1.x = (int)(fmax(ceil((double)(ofc->m_iLowDimX) / 8.0), 1.0));
	priv->m_lowGrid8x8x1.y = (int)(fmax(ceil((double)(ofc->m_iLowDimY) / 8.0), 1.0));
}

/*
* Blurs a frame
*
* @param ofc: Pointer to the optical flow calculator
* @param frame: Pointer to the frame to blur
* @param blurredFrame: Pointer to the blurred frame
* @param kernelSize: Size of the kernel to use for the blur
* @param directOutput: Whether to output the blurred frame directly
*/
void blurFrameArray(struct OpticalFlowCalc *ofc, const unsigned short* frame, unsigned short* blurredFrame, const int kernelSize, const bool directOutput) {
	struct priv *priv = (struct priv*)ofc->priv;
	// Early exit if kernel size is too small to blur
	if (kernelSize < 4) {
		HIP_CHECK(hipMemcpy(blurredFrame, frame, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToDevice));
		return;
	}
	// Calculate useful constants
	const unsigned char boundsOffset = kernelSize >> 1;
	const unsigned char cacheSize = kernelSize + (boundsOffset << 1);
	const size_t sharedMemSize = cacheSize * cacheSize * sizeof(unsigned int);
	const unsigned short totalThreads = std::max(kernelSize * kernelSize, 1);
    const unsigned short totalEntries = cacheSize * cacheSize;
    const unsigned char avgEntriesPerThread = totalEntries / totalThreads;
	const unsigned short remainder = totalEntries % totalThreads;
	const char lumStart = -(kernelSize >> 1);
	const unsigned char lumEnd = (kernelSize >> 1);
	const unsigned short lumPixelCount = (lumEnd - lumStart) * (lumEnd - lumStart);

	// Calculate grid and thread dimensions
	const unsigned int numBlocksX = std::max(static_cast<int>(ceil(static_cast<double>(ofc->m_iDimX) / std::min(kernelSize, 32))), 1);
	const unsigned int numBlocksY = std::max(static_cast<int>(ceil(static_cast<double>(ofc->m_iDimY) / std::min(kernelSize, 32))), 1);
	dim3 gridDim(numBlocksX, numBlocksY, 1);
	dim3 threadDim(std::min(kernelSize, 32), std::min(kernelSize, 32), 1);

	// Launch kernel
	blurFrameKernel<<<gridDim, threadDim, sharedMemSize, priv->m_csWarpStream1>>>(
		frame, blurredFrame, kernelSize, cacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, ofc->m_iDimY, ofc->m_iDimX);
	HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));

	// Handle direct output if necessary
	if (directOutput) {
		HIP_CHECK(hipMemcpy(ofc->m_outputFrame, blurredFrame, 2 * ofc->m_iDimY * ofc->m_iDimX, hipMemcpyDeviceToDevice));
		HIP_CHECK(hipMemcpy(ofc->m_outputFrame + ofc->m_iDimY * ofc->m_iDimX, frame + ofc->m_iDimY * ofc->m_iDimX, 2 * ((ofc->m_iDimY / 2) * ofc->m_iDimX), hipMemcpyDeviceToDevice));
	}

	// Check for HIP errors
	checkHIPError("blurFrameArray");
}

/*
* Updates the frame arrays and blurs them if necessary
*
* @param ofc: Pointer to the optical flow calculator
* @param pInBuffer: Pointer to the input frame
* @param frameKernelSize: Size of the kernel to use for the frame blur
* @param flowKernelSize: Size of the kernel to use for the flow blur
* @param directOutput: Whether to output the blurred frame directly
*/
void updateFrame(struct OpticalFlowCalc *ofc, unsigned char** pInBuffer, const unsigned int frameKernelSize, const unsigned int flowKernelSize, const bool directOutput) {
	struct priv *priv = (struct priv*)ofc->priv;
	ofc->m_iFlowBlurKernelSize = flowKernelSize;

	HIP_CHECK(hipMemcpy(ofc->m_frame[0], pInBuffer[0], ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(ofc->m_frame[0] + ofc->m_iDimY * ofc->m_iDimX, pInBuffer[1], (ofc->m_iDimY / 2) * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyHostToDevice));

	// Check for HIP Errors
	checkHIPError("updateFrame");
	
	// Blur the frame
	blurFrameArray(ofc, ofc->m_frame[0], ofc->m_blurredFrame[0], frameKernelSize, directOutput);
	
	// Swap the frame buffers
	unsigned short* temp1 = ofc->m_frame[0];
	ofc->m_frame[0] = ofc->m_frame[1];
	ofc->m_frame[1] = ofc->m_frame[2];
	ofc->m_frame[2] = temp1;

	temp1 = ofc->m_blurredFrame[0];
	ofc->m_blurredFrame[0] = ofc->m_blurredFrame[1];
	ofc->m_blurredFrame[1] = ofc->m_blurredFrame[2];
	ofc->m_blurredFrame[2] = temp1;
}

/*
* Downloads the output frame from the GPU to the CPU
*
* @param ofc: Pointer to the optical flow calculator
* @param pOutBuffer: Pointer to the output buffer
*/
void downloadFrame(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer) {
	struct priv *priv = (struct priv*)ofc->priv;

	HIP_CHECK(hipMemcpy(pOutBuffer[0], ofc->m_outputFrame, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToHost));
	HIP_CHECK(hipMemcpy(pOutBuffer[1], ofc->m_outputFrame + ofc->m_iDimY * ofc->m_iDimX, (ofc->m_iDimY >> 1) * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToHost));
	
	// Check for HIP Errors
	checkHIPError("downloadFrame");
}

/*
* Copies the frame in the correct format to the output buffer
*
* @param ofc: Pointer to the optical flow calculator
* @param pOutBuffer: Pointer to the output frame
* @param firstFrame: Whether this is the first frame
*/
void processFrame(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer, const bool firstFrame) {
	struct priv *priv = (struct priv*)ofc->priv;

	if (ofc->m_fBlackLevel == 0.0f && ofc->m_fWhiteLevel == 1023.0f) {
		HIP_CHECK(hipMemcpy(ofc->m_outputFrame, firstFrame ? ofc->m_frame[2] : ofc->m_frame[1], ofc->m_iDimY * ofc->m_iDimX * 3, hipMemcpyDeviceToDevice));
		downloadFrame(ofc, pOutBuffer);
	} else {
		processFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(firstFrame ? ofc->m_frame[2] : ofc->m_frame[1],
												ofc->m_outputFrame,
												ofc->m_iDimY, ofc->m_iDimX, ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
		HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));
		HIP_CHECK(hipMemcpy(ofc->m_outputFrame + ofc->m_iChannelIdxOffset, (firstFrame ? ofc->m_frame[2] : ofc->m_frame[1]) + ofc->m_iChannelIdxOffset, ofc->m_iDimY * (ofc->m_iDimX >> 1) * sizeof(unsigned short), hipMemcpyDeviceToDevice));
		downloadFrame(ofc, pOutBuffer);
	}

	// Check for HIP Errors
	checkHIPError("processFrame");
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param ofc: Pointer to the optical flow calculator
* @param iNumIterations: Number of iterations to calculate the optical flow
* @param iNumSteps: Number of steps executed to find the ideal offset (limits the maximum offset)
*/
void calculateOpticalFlow(struct OpticalFlowCalc *ofc, unsigned int iNumIterations, unsigned int iNumSteps) {
	struct priv *priv = (struct priv*)ofc->priv;
	// Reset variables
	unsigned int iNumStepsPerIter = iNumSteps; // Number of steps executed to find the ideal offset (limits the maximum offset)

	// We set the initial window size to the next larger power of 2
	unsigned int windowDim = 1;
	unsigned int maxDim = std::max(ofc->m_iLowDimX, ofc->m_iLowDimY);
    if (maxDim && !(maxDim & (maxDim - 1))) {
		windowDim = maxDim;
	} else {
		while (maxDim & (maxDim - 1)) {
			maxDim &= (maxDim - 1);
		}
		windowDim = maxDim << 1;
	}

	if (iNumIterations == 0 || static_cast<double>(iNumIterations) > ceil(log2(windowDim))) {
		iNumIterations = static_cast<unsigned int>(ceil(log2(windowDim))) + 1;
	}

	size_t sharedMemSize = 16 * 16 * sizeof(unsigned int);

	// Set layer 0 of the X-Dir to 0
	HIP_CHECK(hipMemset(ofc->m_offsetArray12, 0, ofc->m_iLayerIdxOffset * sizeof(int)));
	// Set layers 0-5 of the Y-Dir to 0
	HIP_CHECK(hipMemset(ofc->m_offsetArray12 + ofc->m_iDirectionIdxOffset, 0, ofc->m_iDirectionIdxOffset * sizeof(int)));
	// Set layers 1-4 of the X-Dir to -2,-1,1,2
	setInitialOffset<<<priv->m_lowGrid16x16x4, priv->m_threads16x16x1, 0, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iLayerIdxOffset);
	HIP_CHECK(hipStreamSynchronize(priv->m_csOFCStream1));
	
	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Calculate the number of steps for this iteration executed to find the ideal offset (limits the maximum offset)
	    //iNumStepsPerIter = static_cast<unsigned int>(static_cast<double>(iNumSteps) - static_cast<double>(iter) * (static_cast<double>(iNumSteps) / static_cast<double>(iNumIterations)));
		iNumStepsPerIter = std::max(static_cast<int>(static_cast<double>(iNumSteps) * exp(-static_cast<double>(3 * iter) / static_cast<double>(iNumIterations))), 1);

		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumStepsPerIter; step++) {
			// Reset the summed up delta array
			HIP_CHECK(hipMemset(ofc->m_summedUpDeltaArray, 0, 5 * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(unsigned int)));

			// 1. Calculate the image delta and sum up the deltas of each window
			calcDeltaSums<<<iter == 0 ? priv->m_lowGrid16x16x5 : priv->m_lowGrid8x8x5, iter == 0 ? priv->m_threads16x16x1 : priv->m_threads8x8x1, sharedMemSize, priv->m_csOFCStream1>>>(ofc->m_summedUpDeltaArray, 
															ofc->m_blurredFrame[1],
															ofc->m_blurredFrame[2],
															ofc->m_offsetArray12, ofc->m_iLayerIdxOffset, ofc->m_iDirectionIdxOffset,
															ofc->m_iDimY, ofc->m_iDimX, ofc->m_iLowDimY, ofc->m_iLowDimX, windowDim, ofc->m_cResolutionScalar);
			
			HIP_CHECK(hipStreamSynchronize(priv->m_csOFCStream1));

			// 2. Normalize the summed up delta array and find the best layer
			normalizeDeltaSums<<<priv->m_lowGrid8x8x1, priv->m_threads8x8x5, 0, priv->m_csOFCStream1>>>(ofc->m_summedUpDeltaArray, ofc->m_lowestLayerArray,
															   ofc->m_offsetArray12, windowDim, windowDim * windowDim,
															   ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset, ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX);
			HIP_CHECK(hipStreamSynchronize(priv->m_csOFCStream1));

			// 3. Adjust the offset array based on the comparison results
			adjustOffsetArray<<<priv->m_lowGrid32x32x1, priv->m_threads32x32x1, 0, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_lowestLayerArray,
															  ofc->m_statusArray, windowDim, ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset,
															  ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX, step == iNumStepsPerIter - 1);
			HIP_CHECK(hipStreamSynchronize(priv->m_csOFCStream1));
		}

		// 4. Adjust variables for the next iteration
		windowDim = std::max(windowDim >> 1, (unsigned int)1);
		sharedMemSize = 8 * 8 * sizeof(unsigned int);
		if (windowDim == 1) sharedMemSize = 0;

		// Reset the status array
		HIP_CHECK(hipMemset(ofc->m_statusArray, 0, ofc->m_iLowDimY * ofc->m_iLowDimX));
	}

	// Check for HIP Errors
	checkHIPError("calculateOpticalFlow");
}

/*
* Warps the frames according to the calculated optical flow
*
* @param ofc: Pointer to the optical flow calculator
* @param fScalar: The scalar to blend the frames with
* @param outputMode: The mode to output the frames in (0: WarpedFrame 1->2, 1: WarpedFrame 2->1, 2: Both for blending)
*/
void warpFrames(struct OpticalFlowCalc *ofc, float fScalar, const int outputMode) {
	struct priv *priv = (struct priv*)ofc->priv;
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = 1.0f - fScalar;

	// Reset the hit count array
	HIP_CHECK(hipMemset(ofc->m_hitCount12, 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(int)));
	HIP_CHECK(hipMemset(ofc->m_hitCount21, 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(int)));

	// #####################
	// ###### WARPING ######
	// #####################
	// Frame 1 to Frame 2
	if (outputMode != 1) {
		warpFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frame[0],
																									ofc->m_blurredOffsetArray12[0],
																									ofc->m_hitCount12,
																									(outputMode < 2) ? ofc->m_outputFrame : ofc->m_warpedFrame12,
																									frameScalar12,
																									ofc->m_iLowDimY,
																									ofc->m_iLowDimX,
																									ofc->m_iDimY,
																									ofc->m_iDimX,
																									ofc->m_cResolutionScalar,
																									ofc->m_iLayerIdxOffset,
																									ofc->m_iChannelIdxOffset);
		
	}
	// Frame 2 to Frame 1
	if (outputMode != 0) {
		warpFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream2>>>(ofc->m_frame[1],
																									ofc->m_blurredOffsetArray21[0],
																									ofc->m_hitCount21,
																									(outputMode < 2) ? ofc->m_outputFrame : ofc->m_warpedFrame21,
																									frameScalar21,
																									ofc->m_iLowDimY,
																									ofc->m_iLowDimX,
																									ofc->m_iDimY,
																									ofc->m_iDimX,
																									ofc->m_cResolutionScalar,
																									ofc->m_iLayerIdxOffset,
																									ofc->m_iChannelIdxOffset);
	}
	if (outputMode != 1) HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));
	if (outputMode != 0) HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream2));
	
	// ##############################
	// ###### ARTIFACT REMOVAL ######
	// ##############################
	// Frame 1 to Frame 2
	if (outputMode != 1) {
		artifactRemovalKernel<<<priv->m_grid8x8x1, priv->m_threads8x8x2, 0, priv->m_csWarpStream1>>>(ofc->m_frame[0],
																									ofc->m_hitCount12,
																									(outputMode < 2) ? ofc->m_outputFrame : ofc->m_warpedFrame12,
																									ofc->m_iDimY,
																									ofc->m_iDimX,
																									ofc->m_iChannelIdxOffset);
	}
	// Frame 2 to Frame 1
	if (outputMode != 0) {
		artifactRemovalKernel<<<priv->m_grid8x8x1, priv->m_threads8x8x2, 0, priv->m_csWarpStream2>>>(ofc->m_frame[1],
																									ofc->m_hitCount21,
																									(outputMode < 2) ? ofc->m_outputFrame : ofc->m_warpedFrame21,
																									ofc->m_iDimY,
																									ofc->m_iDimX,
																									ofc->m_iChannelIdxOffset);
	}
	if (outputMode != 1) HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));
	if (outputMode != 0) HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream2));

	// Check for HIP Errors
	checkHIPError("warpFrames");
}

/*
* Blends warpedFrame1 to warpedFrame2
*
* @param ofc: Pointer to the optical flow calculator
* @param fScalar: The scalar to blend the frames with
*/
void blendFrames(struct OpticalFlowCalc *ofc, float fScalar) {
	struct priv *priv = (struct priv*)ofc->priv;
	// Calculate the blend scalar
	const float frame1Scalar = 1.0f - fScalar;
	const float frame2Scalar = fScalar;

	// Blend the frames
	blendFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_warpedFrame12, ofc->m_warpedFrame21,
												ofc->m_outputFrame, frame1Scalar, frame2Scalar,
												ofc->m_iDimY, ofc->m_iDimX, ofc->m_iChannelIdxOffset, ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);

	HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));

	// Check for HIP Errors
	checkHIPError("blendFrames");
}

/*
* Places left half of frame1 over the outputFrame
*
* @param ofc: Pointer to the optical flow calculator
*/
void insertFrame(struct OpticalFlowCalc *ofc) {
	struct priv *priv = (struct priv*)ofc->priv;
	insertFrameKernel<<<priv->m_halfGrid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frame[0],
												ofc->m_outputFrame,
												ofc->m_iDimY, ofc->m_iDimX, ofc->m_iChannelIdxOffset,
												ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
	HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));

	// Check for HIP Errors
	checkHIPError("insertFrame");
}

/*
* Places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
*
* @param ofc: Pointer to the optical flow calculator
* @param dScalar: The scalar to blend the frames with
* @param frameCounter: The current frame counter
*/
void sideBySideFrame(struct OpticalFlowCalc *ofc, float fScalar, const unsigned int frameCounter) {
	struct priv *priv = (struct priv*)ofc->priv;
	// Calculate the blend scalar
	const float frame1Scalar = 1.0f - fScalar;
	const float frame2Scalar = fScalar;
	const unsigned int halfDimX = ofc->m_iDimX >> 1;
	const unsigned int halfDimY = ofc->m_iDimY >> 1;

	if (frameCounter == 1) {
		sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frame[2], ofc->m_frame[2], 
													ofc->m_frame[2],
													ofc->m_outputFrame, frame1Scalar, frame2Scalar,
													ofc->m_iDimY, ofc->m_iDimX, halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
													ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
	} else if (frameCounter == 2) {
		sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frame[1], ofc->m_frame[1], 
													ofc->m_frame[1],
													ofc->m_outputFrame, frame1Scalar, frame2Scalar,
													ofc->m_iDimY, ofc->m_iDimX, halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
													ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
	} else if (frameCounter <= 3) {
		sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frame[0], ofc->m_frame[0], 
													ofc->m_frame[0],
													ofc->m_outputFrame, frame1Scalar, frame2Scalar,
													ofc->m_iDimY, ofc->m_iDimX, halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
													ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
	} else {
		sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frame[0], ofc->m_warpedFrame12, 
													ofc->m_warpedFrame21,
													ofc->m_outputFrame, frame1Scalar, frame2Scalar,
													ofc->m_iDimY, ofc->m_iDimX, halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
													ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
	}
	HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));

	// Check for HIP Errors
	checkHIPError("sideBySideFrame");
}

/*
* Draws the flow as an RGB image
*
* @param ofc: Pointer to the optical flow calculator
* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
*/
void drawFlowAsHSV(struct OpticalFlowCalc *ofc, const float blendScalar) {
	struct priv *priv = (struct priv*)ofc->priv;
	convertFlowToHSVKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_blurredOffsetArray12[0], ofc->m_outputFrame,
													ofc->m_frame[1], blendScalar, ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iDimY, ofc->m_iDimX,
													ofc->m_cResolutionScalar, ofc->m_iLayerIdxOffset, ofc->m_iChannelIdxOffset);
	
	HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));

	// Check for HIP Errors
	checkHIPError("drawFlowAsHSV");
}

/*
* Draws the flow as an greyscale image
*
* @param ofc: Pointer to the optical flow calculator
*/
void drawFlowAsGreyscale(struct OpticalFlowCalc *ofc) {
	struct priv *priv = (struct priv*)ofc->priv;
	convertFlowToGreyscaleKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_blurredOffsetArray12[0], ofc->m_outputFrame,
													ofc->m_frame[1], ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iDimY, ofc->m_iDimX,
													ofc->m_cResolutionScalar, ofc->m_iLayerIdxOffset, ofc->m_iChannelIdxOffset, (int)ofc->m_fMaxVal, ofc->m_iMiddleValue);
	
	HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));

	// Check for HIP Errors
	checkHIPError("drawFlowAsGreyscale");
}

/*
* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
*
* @param ofc: Pointer to the optical flow calculator
*/
void flipFlow(struct OpticalFlowCalc *ofc) {
	struct priv *priv = (struct priv*)ofc->priv;
	// Reset the offset array
	HIP_CHECK(hipMemset(ofc->m_offsetArray21, 0, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(int)));

	// Launch kernel
	flipFlowKernel<<<priv->m_lowGrid16x16x1, priv->m_threads16x16x2, 0, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_offsetArray21,
												            ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_cResolutionScalar, ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset);
	HIP_CHECK(hipStreamSynchronize(priv->m_csOFCStream1));

	// Check for HIP Errors
	checkHIPError("flipFlow");
}

/*
* Blurs the offset arrays
*
* @param ofc: Pointer to the optical flow calculator
*/
void blurFlowArrays(struct OpticalFlowCalc *ofc) {
	struct priv *priv = (struct priv*)ofc->priv;
	const unsigned char boundsOffset = ofc->m_iFlowBlurKernelSize >> 1;
	const unsigned char chacheSize = ofc->m_iFlowBlurKernelSize + (boundsOffset << 1);
	const size_t sharedMemSize = chacheSize * chacheSize * sizeof(int);
	const unsigned short totalThreads = std::max(ofc->m_iFlowBlurKernelSize * ofc->m_iFlowBlurKernelSize, (unsigned int)1);
    const unsigned short totalEntries = chacheSize * chacheSize;
    const unsigned char avgEntriesPerThread = totalEntries / totalThreads;
	const unsigned short remainder = totalEntries % totalThreads;
	const char start = -(ofc->m_iFlowBlurKernelSize >> 1);
	const unsigned char end = (ofc->m_iFlowBlurKernelSize >> 1);
	const unsigned short pixelCount = (end - start) * (end - start);

	// Calculate the number of blocks needed
	const unsigned int NUM_BLOCKS_X = std::max(static_cast<int>(ceil(static_cast<double>(ofc->m_iLowDimX) / std::min(ofc->m_iFlowBlurKernelSize, (unsigned int)32))), 1);
	const unsigned int NUM_BLOCKS_Y = std::max(static_cast<int>(ceil(static_cast<double>(ofc->m_iLowDimY) / std::min(ofc->m_iFlowBlurKernelSize, (unsigned int)32))), 1);

	// Use dim3 structs for block and grid size
	dim3 gridBF(NUM_BLOCKS_X, NUM_BLOCKS_Y, 2);
	dim3 threadsBF(std::min(ofc->m_iFlowBlurKernelSize, (unsigned int)32), std::min(ofc->m_iFlowBlurKernelSize, (unsigned int)32), (unsigned int)1);

	// No need to blur the flow if the kernel size is less than 4
	if (ofc->m_iFlowBlurKernelSize < 4) {
		// Offset12 X-Dir
		HIP_CHECK(hipMemcpy(ofc->m_blurredOffsetArray12[1], ofc->m_offsetArray12, ofc->m_iLayerIdxOffset * sizeof(int), hipMemcpyDeviceToDevice));
		// Offset12 Y-Dir
		HIP_CHECK(hipMemcpy(ofc->m_blurredOffsetArray12[1] + ofc->m_iLayerIdxOffset, ofc->m_offsetArray12 + ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset * sizeof(int), hipMemcpyDeviceToDevice));
		// Offset21 X&Y-Dir
		HIP_CHECK(hipMemcpy(ofc->m_blurredOffsetArray21[1], ofc->m_offsetArray21, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(int), hipMemcpyDeviceToDevice));
	} else {
		// Launch kernels
		blurFlowKernel<<<gridBF, threadsBF, sharedMemSize, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_blurredOffsetArray12[1], ofc->m_iFlowBlurKernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, start, end, pixelCount, ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX);
		blurFlowKernel<<<gridBF, threadsBF, sharedMemSize, priv->m_csOFCStream2>>>(ofc->m_offsetArray21, ofc->m_blurredOffsetArray21[1], ofc->m_iFlowBlurKernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, start, end, pixelCount, 1, ofc->m_iLowDimY, ofc->m_iLowDimX);
		//cleanFlowKernel<<<m_lowGrid16x16x1, m_threads16x16x2, 0, blurStream1>>>(m_offsetArray12, m_blurredOffsetArray12, m_iLowDimY, m_iLowDimX);
		//cleanFlowKernel<<<m_lowGrid16x16x1, m_threads16x16x2, 0, blurStream2>>>(m_offsetArray21, m_blurredOffsetArray21, m_iLowDimY, m_iLowDimX);

		// Synchronize streams to ensure completion
		HIP_CHECK(hipStreamSynchronize(priv->m_csOFCStream1));
		HIP_CHECK(hipStreamSynchronize(priv->m_csOFCStream2));
	}

	// Check for HIP Errors
	checkHIPError("blurFlowArrays");
}

/*
* Saves an image to a file
*
* @param ofc: Pointer to the optical flow calculator
* @param filePath: Path to the image file
*/
void saveImage(struct OpticalFlowCalc *ofc, const char* filePath) {
	// Copy the image array to the CPU
	size_t dataSize = 1.5 * ofc->m_iDimY * ofc->m_iDimX;
	HIP_CHECK(hipMemcpy(ofc->m_imageArrayCPU, ofc->m_outputFrame, dataSize, hipMemcpyDeviceToHost));

	// Open file in binary write mode
    FILE *file = fopen(filePath, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the array to the file
    size_t written = fwrite(ofc->m_imageArrayCPU, sizeof(unsigned char), dataSize, file);
    if (written != dataSize) {
        perror("Error writing to file");
        fclose(file);
        return;
    }

    // Close the file
    fclose(file);
}

/*
* Runs a tearing test on the GPU
*
* @param ofc: Pointer to the optical flow calculator
*/
static int counter = 0;
void tearingTest(struct OpticalFlowCalc *ofc) {
	struct priv *priv = (struct priv*)ofc->priv;
	HIP_CHECK(hipMemset(ofc->m_outputFrame, 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short)));
	HIP_CHECK(hipMemset(ofc->m_outputFrame + ofc->m_iDimY * ofc->m_iDimX, 128, (ofc->m_iDimY / 2) * ofc->m_iDimX * sizeof(unsigned short)));
	tearingTestKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(ofc->m_outputFrame, ofc->m_iDimY, ofc->m_iDimX, 10, counter % ofc->m_iDimX);
	counter++;
	HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));
	checkHIPError("tearingTest");
}

/*
* Creates a new GPUArray of type unsigned char
*
* @param size: Number of entries in the array
*/
unsigned char* createGPUArrayUC(const size_t size) {
	// Allocate VRAM
	unsigned char* arrayPtrGPU;
	HIP_CHECK(hipMalloc(&arrayPtrGPU, size));

	// Set all entries to 0
	HIP_CHECK(hipMemset(arrayPtrGPU, 0, size));

	return arrayPtrGPU;
}

/*
* Creates a new GPUArray of type unsigned short
*
* @param size: Number of entries in the array
*/
unsigned short* createGPUArrayUS(const size_t size) {
	// Allocate VRAM
	unsigned short* arrayPtrGPU;
	HIP_CHECK(hipMalloc(&arrayPtrGPU, size * sizeof(unsigned short)));

	// Set all entries to 0
	HIP_CHECK(hipMemset(arrayPtrGPU, 0, size * sizeof(unsigned short)));

	return arrayPtrGPU;
}

/*
* Creates a new GPUArray of type int
*
* @param size: Number of entries in the array
*/
int* createGPUArrayI(const size_t size) {
	// Allocate VRAM
	int* arrayPtrGPU;
	HIP_CHECK(hipMalloc(&arrayPtrGPU, size * sizeof(int)));

	// Set all entries to 0
	HIP_CHECK(hipMemset(arrayPtrGPU, 0, size * sizeof(int)));

	return arrayPtrGPU;
}

/*
* Creates a new GPUArray of type unsigned int
*
* @param size: Number of entries in the array
*/
unsigned int* createGPUArrayUI(const size_t size) {
	// Allocate VRAM
	unsigned int* arrayPtrGPU;
	HIP_CHECK(hipMalloc(&arrayPtrGPU, size * sizeof(unsigned int)));

	// Set all entries to 0
	HIP_CHECK(hipMemset(arrayPtrGPU, 0, size * sizeof(unsigned int)));

	return arrayPtrGPU;
}

/*
* Initializes the optical flow calculator
*
* @param ofc: Pointer to the optical flow calculator
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param resolutionScalar: The scalar to reduce the resolution by
* @param flowBlurKernelSize: The size of the kernel to use for the flow blur
*/
void initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int dimY, const int dimX, unsigned char resolutionScalar, unsigned int flowBlurKernelSize) {
	// Private Data
	ofc->priv = calloc(1, sizeof(struct priv));
	struct priv *priv = (struct priv*)ofc->priv;
	
	// Functions
	ofc->free = free;
	ofc->adjustFrameScalar = adjustFrameScalar;
	ofc->updateFrame = updateFrame;
	ofc->downloadFrame = downloadFrame;
	ofc->processFrame = processFrame;
	ofc->blurFrameArray = blurFrameArray;
	ofc->calculateOpticalFlow = calculateOpticalFlow;
	ofc->flipFlow = flipFlow;
	ofc->blurFlowArrays = blurFlowArrays;
	ofc->warpFrames = warpFrames;
	ofc->blendFrames = blendFrames;
	ofc->insertFrame = insertFrame;
	ofc->sideBySideFrame = sideBySideFrame;
	ofc->drawFlowAsHSV = drawFlowAsHSV;
	ofc->drawFlowAsGreyscale = drawFlowAsGreyscale;
	ofc->saveImage = saveImage;
	ofc->tearingTest = tearingTest;

	// Video properties
	ofc->m_iDimX = dimX;
	ofc->m_iDimY = dimY;
	ofc->m_fBlackLevel = 0.0f;
	ofc->m_fMaxVal = 65535.0f;
	ofc->m_iMiddleValue = 32768;
	ofc->m_fWhiteLevel = ofc->m_fMaxVal;

	// Optical flow calculation
	ofc->m_cResolutionScalar = resolutionScalar;
	ofc->m_iLowDimX = dimX >> ofc->m_cResolutionScalar;
	ofc->m_iLowDimY = dimY >> ofc->m_cResolutionScalar;
	ofc->m_iNumLayers = 5;
	ofc->m_iDirectionIdxOffset = ofc->m_iNumLayers * ofc->m_iLowDimY * ofc->m_iLowDimX;
	ofc->m_iLayerIdxOffset = ofc->m_iLowDimY * ofc->m_iLowDimX;
	ofc->m_iChannelIdxOffset = ofc->m_iDimY * ofc->m_iDimX;
	ofc->m_iFlowBlurKernelSize = flowBlurKernelSize;

	// Girds
	priv->m_lowGrid32x32x1.x = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimX) / 32.0), 1.0));
	priv->m_lowGrid32x32x1.y = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimY) / 32.0), 1.0));
	priv->m_lowGrid32x32x1.z = 1;
	priv->m_lowGrid16x16x5.x = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimX) / 16.0), 1.0));
	priv->m_lowGrid16x16x5.y = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimY) / 16.0), 1.0));
	priv->m_lowGrid16x16x5.z = 5;
	priv->m_lowGrid16x16x4.x = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimX) / 16.0), 1.0));
	priv->m_lowGrid16x16x4.y = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimY) / 16.0), 1.0));
	priv->m_lowGrid16x16x4.z = 4;
	priv->m_lowGrid16x16x1.x = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimX) / 16.0), 1.0));
	priv->m_lowGrid16x16x1.y = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimY) / 16.0), 1.0));
	priv->m_lowGrid16x16x1.z = 1;
	priv->m_lowGrid8x8x5.x = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimX) / 8.0), 1.0));
	priv->m_lowGrid8x8x5.y = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimY) / 8.0), 1.0));
	priv->m_lowGrid8x8x5.z = 5;
	priv->m_lowGrid8x8x1.x = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimX) / 8.0), 1.0));
	priv->m_lowGrid8x8x1.y = static_cast<int>(fmax(ceil(static_cast<double>(ofc->m_iLowDimY) / 8.0), 1.0));
	priv->m_lowGrid8x8x1.z = 1;
	priv->m_grid16x16x1.x = static_cast<int>(fmax(ceil(dimX / 16.0), 1.0));
	priv->m_grid16x16x1.y = static_cast<int>(fmax(ceil(dimY / 16.0), 1.0));
	priv->m_grid16x16x1.z = 1;
	priv->m_halfGrid16x16x1.x = static_cast<int>(fmax(ceil(dimX / 32.0), 1.0));
	priv->m_halfGrid16x16x1.y = static_cast<int>(fmax(ceil(dimY / 16.0), 1.0));
	priv->m_halfGrid16x16x1.z = 1;
	priv->m_grid8x8x1.x = static_cast<int>(fmax(ceil(dimX / 8.0), 1.0));
	priv->m_grid8x8x1.y = static_cast<int>(fmax(ceil(dimY / 8.0), 1.0));
	priv->m_grid8x8x1.z = 1;

	// Threads
	priv->m_threads32x32x1.x = 32;
	priv->m_threads32x32x1.y = 32;
	priv->m_threads32x32x1.z = 1;
	priv->m_threads16x16x2.x = 16;
	priv->m_threads16x16x2.y = 16;
	priv->m_threads16x16x2.z = 2;
	priv->m_threads16x16x1.x = 16;
	priv->m_threads16x16x1.y = 16;
	priv->m_threads16x16x1.z = 1;
	priv->m_threads8x8x5.x = 8;
	priv->m_threads8x8x5.y = 8;
	priv->m_threads8x8x5.z = 5;
	priv->m_threads8x8x2.x = 8;
	priv->m_threads8x8x2.y = 8;
	priv->m_threads8x8x2.z = 2;
	priv->m_threads8x8x1.x = 8;
	priv->m_threads8x8x1.y = 8;
	priv->m_threads8x8x1.z = 1;

	// GPU Arrays
	HIP_CHECK(hipSetDevice(0));
	ofc->m_frame[0] = createGPUArrayUS(1.5 * dimY * dimX);
	ofc->m_frame[1] = createGPUArrayUS(1.5 * dimY * dimX);
	ofc->m_frame[2] = createGPUArrayUS(1.5 * dimY * dimX);
	ofc->m_blurredFrame[0] = createGPUArrayUS(dimY * dimX);
	ofc->m_blurredFrame[1] = createGPUArrayUS(dimY * dimX);
	ofc->m_blurredFrame[2] = createGPUArrayUS(dimY * dimX);
	ofc->m_warpedFrame12 = createGPUArrayUS(1.5 * dimY * dimX);
	ofc->m_warpedFrame21 = createGPUArrayUS(1.5 * dimY * dimX);
	ofc->m_outputFrame = createGPUArrayUS(1.5 * dimY * dimX);
	ofc->m_tempFrame = createGPUArrayUS((dimY / 2) * dimX);
	ofc->m_offsetArray12 = createGPUArrayI(2 * 5 * dimY * dimX);
	ofc->m_offsetArray21 = createGPUArrayI(2 * dimY * dimX);
	ofc->m_blurredOffsetArray12[0] = createGPUArrayI(2 * dimY * dimX);
	ofc->m_blurredOffsetArray21[0] = createGPUArrayI(2 * dimY * dimX);
	ofc->m_blurredOffsetArray12[1] = createGPUArrayI(2 * dimY * dimX);
	ofc->m_blurredOffsetArray21[1] = createGPUArrayI(2 * dimY * dimX);
	ofc->m_statusArray = createGPUArrayUC(dimY * dimX);
	ofc->m_summedUpDeltaArray = createGPUArrayUI(5 * dimY * dimX);
	ofc->m_lowestLayerArray = createGPUArrayUC(dimY * dimX);
	ofc->m_hitCount12 = createGPUArrayI(dimY * dimX);
	ofc->m_hitCount21 = createGPUArrayI(dimY * dimX);

	// Create HIP streams
	HIP_CHECK(hipStreamCreate(&priv->m_csOFCStream1));
	HIP_CHECK(hipStreamCreate(&priv->m_csOFCStream2));
	HIP_CHECK(hipStreamCreate(&priv->m_csWarpStream1));
	HIP_CHECK(hipStreamCreate(&priv->m_csWarpStream2));
}