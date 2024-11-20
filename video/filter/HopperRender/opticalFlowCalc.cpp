#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "opticalFlowCalc.h"
#include <algorithm>

#define SCALE_FLOW 0
#define YUV420P_FMT 1002
#define NV12_FMT 1006
#define HIP_FMT 1026

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
__device__ void applyShaderY(unsigned char& output, const unsigned char input, const float blackLevel, const float whiteLevel, const float maxVal) {
	output = static_cast<unsigned char>(fmaxf(fminf((static_cast<float>(input) - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f));
}
__device__ void applyShaderY(unsigned short& output, const unsigned short input, const float blackLevel, const float whiteLevel, const float maxVal) {
	output = static_cast<unsigned short>(fmaxf(fminf((static_cast<float>(input) - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f));
}

// Kernel that converts a NV12 frame to a YUV420P frame
template <typename T>
__global__ void convertNV12toYUV420PKernel(T* outputFrame, const T* inputFrame, const unsigned int dimY, 
										   const unsigned int dimX, const unsigned int halfDimY, 
										   const unsigned int halfDimX, const unsigned int channelIdxOffset, const unsigned int secondChannelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	if (cy < halfDimY && cx < dimX) {
		// U Channel
		if (!(cx & 1)) {
			outputFrame[cy * halfDimX + (cx >> 1)] = inputFrame[channelIdxOffset + cy * dimX + cx];
		// V Channel
		} else {
			outputFrame[secondChannelIdxOffset + cy * halfDimX + (cx >> 1)] = inputFrame[channelIdxOffset + cy * dimX + cx];
		}
	}
}

// Kernel that converts a YUV420P frame to a NV12 frame
// Note that it expects the frame to only contain the U and V channels (so the first byte should be of the U channel)
template <typename T>
__global__ void convertYUV420PtoNV12Kernel(T* outputFrame, const T* inputFrame, const unsigned int dimY, 
										   const unsigned int dimX, const unsigned int halfDimY, 
										   const unsigned int halfDimX, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	if (cy < halfDimY && cx < dimX) {
		// U Channel
		if (!(cx & 1)) {
			outputFrame[channelIdxOffset + cy * dimX + cx] = inputFrame[cy * halfDimX + (cx >> 1)];
		// V Channel
		} else {
			outputFrame[channelIdxOffset + cy * dimX + cx] = inputFrame[halfDimY * halfDimX + cy * halfDimX + (cx >> 1)];
		}
	}
}

// Kernel that simply applies a shader to the frame and copies it to the output frame
template <typename T>
__global__ void processFrameKernel(const T* frame, T* outputFrame,
                                 const unsigned int dimY,
                                 const unsigned int dimX, const unsigned int realDimX, const float blackLevel, const float whiteLevel, const float maxVal) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Y Channel
	if (cy < dimY && cx < realDimX) {
		applyShaderY(outputFrame[cy * dimX + cx], frame[cy * dimX + cx], blackLevel, whiteLevel, maxVal);
	}
}

// Kernel that blurs a 2D plane along the X direction
template <typename T>
__global__ void blurFrameKernel(const T* frameArray, T* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const unsigned short dimY, const unsigned short dimX,
								const unsigned int realDimX) {
	// Shared memory for the frame to prevent multiple global memory accesses
	extern __shared__ unsigned int sharedFrameArray[];
	// Current entry to be computed by the thread
	const unsigned short cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned short cy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the current thread is supposed to perform calculations
	if (cy >= dimY || cx >= realDimX) {
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
		if (newY < dimY && newY >= 0 && newX < realDimX && newX >= 0) {
			sharedFrameArray[startIndex + i] = frameArray[newY * dimX + newX];
		} else {
			sharedFrameArray[startIndex + i] = 0;
		}
	}

    // Ensure all threads have finished loading before continuing
    __syncthreads();

	// Don't blur the edges of the frame
	if (cy < kernelSize / 2 || cy >= dimY - kernelSize / 2 || cx < kernelSize / 2 || cx >= realDimX - kernelSize / 2) {
		blurredFrameArray[cy * dimX + cx] = 0;
		return;
	}

	unsigned int blurredPixel = 0;
	// Collect the sum of the surrounding pixels
	for (char y = lumStart; y < lumEnd; y++) {
		for (char x = lumStart; x < lumEnd; x++) {
			if ((cy + y) < dimY && (cy + y) >= 0 && (cx + x) < realDimX && (cx + x) >= 0) {
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
template <typename T>
__global__ void calcDeltaSums(unsigned int* summedUpDeltaArray, const T* frame1, const T* frame2,
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
template <typename T>
__global__ void warpFrameKernel(const T* frame1, const int* offsetArray, int* hitCount,
								T* warpedFrame, const float frameScalar, const unsigned int lowDimY, const unsigned int lowDimX,
								const unsigned int dimY, const int dimX, const unsigned int realDimX, const unsigned char resolutionScalar,
								const unsigned int directionIdxOffset, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const int cz = blockIdx.z * blockDim.z + threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < realDimX) {
		// Get the current offsets to use
		const int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
		const int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar) ;
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar) ;
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < realDimX) {
			warpedFrame[newCy * dimX + newCx] = frame1[cy * dimX + cx];
			atomicAdd(&hitCount[newCy * dimX + newCx], 1);
		}

	// U/V-Channel
	} else if (cz == 1 && cy < (dimY >> 1) && cx < realDimX) {
		const int scaledCx = (cx >> resolutionScalar) & ~1; // The X-Index of the current thread in the offset array
		const int scaledCy = (cy >> resolutionScalar) << 1; // The Y-Index of the current thread in the offset array
		const int offsetX = static_cast<int>(static_cast<float>(offsetArray[scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar);
		const int offsetY = static_cast<int>(static_cast<float>(offsetArray[directionIdxOffset + scaledCy * lowDimX + scaledCx] * 1 << (resolutionScalar * SCALE_FLOW)) * frameScalar) >> 1;
		
		const int newCx = cx + offsetX;
		const int newCy = cy + offsetY;

		// Check if the current pixel is inside the frame
		if (newCy >= 0 && newCy < (dimY >> 1) && newCx >= 0 && newCx < realDimX) {
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
template <typename T>
__global__ void artifactRemovalKernel(const T* frame1, const int* hitCount, T* warpedFrame,
												 const unsigned int dimY, const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int threadIndex2D = cy * dimX + cx; // Standard thread index without Z-Dim

	// Y Channel
	if (cz == 0 && cy < dimY && cx < realDimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[threadIndex2D] = frame1[threadIndex2D];
		}

	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < realDimX) {
		if (hitCount[threadIndex2D] != 1) {
			warpedFrame[channelIdxOffset + cy * dimX + cx] = frame1[channelIdxOffset + threadIndex2D];
		}
	}
}

// Kernel that blends warpedFrame1 to warpedFrame2
template <typename T>
__global__ void blendFrameKernel(const T* warpedFrame1, const T* warpedFrame2, T* outputFrame,
                                 const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                 const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset, const float blackLevel, const float whiteLevel, const float maxVal) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	float pixelValue;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < realDimX) {
		pixelValue = static_cast<float>(warpedFrame1[cy * dimX + cx]) * frame1Scalar + 
					 static_cast<float>(warpedFrame2[cy * dimX + cx]) * frame2Scalar;
		applyShaderY(outputFrame[cy * dimX + cx], pixelValue, blackLevel, whiteLevel, maxVal);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < realDimX) {
		pixelValue = static_cast<float>(warpedFrame1[channelIdxOffset + cy * dimX + cx]) * frame1Scalar + 
					 static_cast<float>(warpedFrame2[channelIdxOffset + cy * dimX + cx]) * frame2Scalar;
		outputFrame[channelIdxOffset + cy * dimX + cx] = pixelValue;
	}
}

// Kernel that places half of frame 1 over the outputFrame
template <typename T>
__global__ void insertFrameKernel(const T* frame1, T* outputFrame, const unsigned int dimY,
                                  const unsigned int dimX, const unsigned int realDimX, 
								  const unsigned int channelIdxOffset, const float blackLevel, 
								  const float whiteLevel, const float maxVal) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;

	// Y Channel
	if (cz == 0 && cy < dimY && cx < (realDimX >> 1)) {
		applyShaderY(outputFrame[cy * dimX + cx], frame1[cy * dimX + cx], blackLevel, whiteLevel, maxVal);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < (realDimX >> 1)) {
		outputFrame[channelIdxOffset + cy * dimX + cx] = frame1[channelIdxOffset + cy * dimX + cx];
	}
}

// Kernel that places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
template <typename T>
__global__ void sideBySideFrameKernel(const T* frame1, const T* warpedFrame1, const T* warpedFrame2, T* outputFrame, 
									  const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                      const unsigned int dimX, const unsigned int realDimX, const unsigned int halfDimY, 
									  const unsigned int halfDimX, const unsigned int channelIdxOffset,
									  const float blackLevel, const float whiteLevel, const float maxVal, const unsigned short middleValue) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const unsigned int verticalOffset = dimY >> 2;
	const bool isYChannel = cz == 0 && cy < dimY && cx < realDimX;
	const bool isUVChannel = cz == 1 && cy < halfDimY && cx < realDimX;
	const bool isVChannel = (cx & 1) == 1;
	const bool isInLeftSideY = cy >= verticalOffset && cy < (verticalOffset + halfDimY) && cx < halfDimX;
	const bool isInRightSideY = cy >= verticalOffset && cy < (verticalOffset + halfDimY) && cx >= halfDimX && cx < realDimX;
	const bool isInLeftSideUV = cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx < halfDimX;
	const bool isInRightSideUV = cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx >= halfDimX && cx < realDimX;
	T blendedFrameValue;

	// Early exit if thread indices are out of bounds
    if (cz > 1 || cy >= dimY || cx >= realDimX || (cz == 1 && cy >= halfDimY)) return;

	// --- Blending ---
	// Y Channel
	if (isYChannel && isInRightSideY) {
		blendedFrameValue = 
			static_cast<T>(
				static_cast<float>(warpedFrame1[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) * frame1Scalar + 
				static_cast<float>(warpedFrame2[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) * frame2Scalar
			);
	// U/V Channels
	} else if (isUVChannel && isInRightSideUV) {
		blendedFrameValue = 
			static_cast<T>(
				static_cast<float>(warpedFrame1[channelIdxOffset + 2 * (cy - (verticalOffset >> 1)) * dimX + ((cx - halfDimX) << 1) + isVChannel]) * frame1Scalar + 
				static_cast<float>(warpedFrame2[channelIdxOffset + 2 * (cy - (verticalOffset >> 1)) * dimX + ((cx - halfDimX) << 1) + isVChannel]) * frame2Scalar
			);
	}

	// --- Insertion ---
	if (isYChannel) {
		// Y Channel Left Side
		if (isInLeftSideY) {
			applyShaderY(outputFrame[cy * dimX + cx], static_cast<T>(frame1[((cy - verticalOffset) << 1) * dimX + (cx << 1)]), blackLevel, whiteLevel, maxVal);
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
template <typename T>
__global__ void convertFlowToHSVKernel(const int* flowArray, T* outputFrame, const T* frame1,
                                       const float blendScalar, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX, 
									   const unsigned int realDimX, const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset, const bool isP010) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const bool isHDR = sizeof(T) == 2;

	const unsigned int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
	const unsigned int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array

	// Get the current flow values
	float x;
	float y;
	if (cz == 0 && cy < dimY && cx < realDimX) {
		x = flowArray[scaledCy * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + scaledCy * lowDimX + scaledCx];
	} else if (cz == 1 && cy < (dimY >> 1) && cx < realDimX){
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
	if (cz == 0 && cy < dimY && cx < realDimX) {
		outputFrame[cy * dimX + cx] = isHDR ?
			(static_cast<unsigned short>(
				(fmaxf(fminf(rgb.r * 0.299f + rgb.g * 0.587f + rgb.b * 0.114f, 255.0f), 0.0f)) * blendScalar) << (isP010 ? 8 : 2)) + 
				frame1[cy * dimX + cx] * (1.0f - blendScalar)
			:
			static_cast<unsigned char>(
				(fmaxf(fminf(rgb.r * 0.299f + rgb.g * 0.587f + rgb.b * 0.114f, 255.0f), 0.0f)) * blendScalar + 
				frame1[cy * dimX + cx] * (1.0f - blendScalar)
			);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < realDimX) {
		// U Channel
		if ((cx & 1) == 0) {
			outputFrame[channelIdxOffset + cy * dimX + (cx & ~1)] = isHDR ?
				static_cast<unsigned short>(
						fmaxf(fminf(rgb.r * -0.168736f + rgb.g * -0.331264f + rgb.b * 0.5f + 128.0f, 255.0f), 0.0f)) << (isP010 ? 8 : 2)
				:
				static_cast<unsigned char>(
					fmaxf(fminf(rgb.r * -0.168736f + rgb.g * -0.331264f + rgb.b * 0.5f + 128.0f, 255.0f), 0.0f)
				);
		// V Channel
		} else {
			outputFrame[channelIdxOffset + cy * dimX + (cx & ~1) + 1] = isHDR ?
				static_cast<unsigned short>(
						fmaxf(fminf(rgb.r * 0.5f + rgb.g * -0.418688f + rgb.b * -0.081312f + 128.0f, 255.0f), 0.0f)) << (isP010 ? 8 : 2)
				:
				static_cast<unsigned char>(
					fmaxf(fminf(rgb.r * 0.5f + rgb.g * -0.418688f + rgb.b * -0.081312f + 128.0f, 255.0f), 0.0f)
				);
		}
	}
}

// Kernel that creates an greyscale flow image from the offset array
template <typename T>
__global__ void convertFlowToGreyscaleKernel(const int* flowArray, T* outputFrame, const T* frame1,
                                       const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX, 
									   const unsigned int realDimX, const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset, const bool isP010, const int maxVal, const unsigned short middleValue) {
	// Current entry to be computed by the thread
	const unsigned int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int cz = threadIdx.z;
	const bool isHDR = sizeof(T) == 2;
	const unsigned char fmtScalar = isP010 ? 6 : 0;

	const unsigned int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
	const unsigned int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array

	// Get the current flow values
	int x;
	int y;
	if (cz == 0 && cy < dimY && cx < realDimX) {
		x = flowArray[scaledCy * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + scaledCy * lowDimX + scaledCx];
	} else if (cz == 1 && cy < (dimY >> 1) && cx < realDimX){
		x = flowArray[(scaledCy << 1) * lowDimX + scaledCx];
		y = flowArray[directionIdxOffset + (scaledCy << 1) * lowDimX + scaledCx];
	}

	// Y Channel
	if (cz == 0 && cy < dimY && cx < realDimX) {
		outputFrame[cy * dimX + cx] = isHDR ? min((abs(x) + abs(y)) << (4 + fmtScalar), maxVal) : min((abs(x) + abs(y)) << 2, maxVal);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < realDimX) {
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
	if (ofc->m_bIsHDR) {
		hipFree(ofc->m_frameHDR[0]);
		hipFree(ofc->m_frameHDR[1]);
		hipFree(ofc->m_frameHDR[2]);
		hipFree(ofc->m_blurredFrameHDR[0]);
		hipFree(ofc->m_blurredFrameHDR[1]);
		hipFree(ofc->m_blurredFrameHDR[2]);
		hipFree(ofc->m_warpedFrame12HDR);
		hipFree(ofc->m_warpedFrame21HDR);
		hipFree(ofc->m_outputFrameHDR);
	} else {
		hipFree(ofc->m_frameSDR[0]);
		hipFree(ofc->m_frameSDR[1]);
		hipFree(ofc->m_frameSDR[2]);
		hipFree(ofc->m_blurredFrameSDR[0]);
		hipFree(ofc->m_blurredFrameSDR[1]);
		hipFree(ofc->m_blurredFrameSDR[2]);
		hipFree(ofc->m_warpedFrame12SDR);
		hipFree(ofc->m_warpedFrame21SDR);
		hipFree(ofc->m_outputFrameSDR);
	}
	hipFree(ofc->m_offsetArray12);
	hipFree(ofc->m_offsetArray21);
	hipFree(ofc->m_blurredOffsetArray12[0]);
	hipFree(ofc->m_blurredOffsetArray21[0]);
	hipFree(ofc->m_blurredOffsetArray12[1]);
	hipFree(ofc->m_blurredOffsetArray21[1]);
	hipFree(ofc->m_statusArray);
	hipFree(ofc->m_summedUpDeltaArray);
	hipFree(ofc->m_lowestLayerArray);
	hipFree(ofc->m_hitCount12);
	hipFree(ofc->m_hitCount21);

	hipStreamDestroy(priv->m_csOFCStream1);
	hipStreamDestroy(priv->m_csOFCStream2);
	hipStreamDestroy(priv->m_csWarpStream1);
	hipStreamDestroy(priv->m_csWarpStream2);
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
void blurFrameArray(struct OpticalFlowCalc *ofc, const void* frame, void* blurredFrame, const int kernelSize, const bool directOutput) {
	struct priv *priv = (struct priv*)ofc->priv;
	// Early exit if kernel size is too small to blur
	if (kernelSize < 4) {
		hipMemcpy(blurredFrame, frame, (ofc->m_bIsHDR ? 2 : 1) * ofc->m_iDimY * ofc->m_iDimX, hipMemcpyDeviceToDevice);
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

	// Launch kernel based on HDR/SDR type
	if (ofc->m_bIsHDR) {
		blurFrameKernel<<<gridDim, threadDim, sharedMemSize, priv->m_csWarpStream1>>>(
			static_cast<const unsigned short*>(frame), static_cast<unsigned short*>(blurredFrame), kernelSize, cacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX);
		hipStreamSynchronize(priv->m_csWarpStream1);
	} else {
		blurFrameKernel<<<gridDim, threadDim, sharedMemSize, priv->m_csWarpStream1>>>(
			static_cast<const unsigned char*>(frame), static_cast<unsigned char*>(blurredFrame), kernelSize, cacheSize, boundsOffset, avgEntriesPerThread, remainder, lumStart, lumEnd, lumPixelCount, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX);
		hipStreamSynchronize(priv->m_csWarpStream1);
	}

	// Handle direct output if necessary
	if (directOutput) {
		if (ofc->m_bIsHDR) {
			hipMemcpy(ofc->m_outputFrameHDR, blurredFrame, 2 * ofc->m_iDimY * ofc->m_iDimX, hipMemcpyDeviceToDevice);
			hipMemcpy(ofc->m_outputFrameHDR + ofc->m_iDimY * ofc->m_iDimX, static_cast<const unsigned short*>(frame) + ofc->m_iDimY * ofc->m_iDimX, 2 * ((ofc->m_iDimY / 2) * ofc->m_iDimX), hipMemcpyDeviceToDevice);
		} else {
			hipMemcpy(ofc->m_outputFrameSDR, blurredFrame, ofc->m_iDimY * ofc->m_iDimX, hipMemcpyDeviceToDevice);
			hipMemcpy(ofc->m_outputFrameSDR + ofc->m_iDimY * ofc->m_iDimX, static_cast<const unsigned char*>(frame) + ofc->m_iDimY * ofc->m_iDimX, (ofc->m_iDimY / 2) * ofc->m_iDimX, hipMemcpyDeviceToDevice);
		}
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
	// P010 format
	if (ofc->m_bIsHDR && ofc->m_iFMT == HIP_FMT) {
		hipMemcpy(ofc->m_frameHDR[0], pInBuffer[0], ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToDevice);
		hipMemcpy(ofc->m_frameHDR[0] + ofc->m_iDimY * ofc->m_iDimX, pInBuffer[1], (ofc->m_iDimY / 2) * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToDevice);
	// YUV420P10 format
	} else if (ofc->m_bIsHDR) {
		hipMemcpy(ofc->m_frameHDR[0], pInBuffer[0], ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyHostToDevice);
		hipMemcpy(ofc->m_tempFrameHDR, pInBuffer[1], (ofc->m_iDimY / 2) * (ofc->m_iDimX / 2) * sizeof(unsigned short), hipMemcpyHostToDevice);
		hipMemcpy(ofc->m_tempFrameHDR + (ofc->m_iDimY / 2) * (ofc->m_iDimX / 2), pInBuffer[2], (ofc->m_iDimY / 2) * (ofc->m_iDimX / 2) * sizeof(unsigned short), hipMemcpyHostToDevice);
		convertYUV420PtoNV12Kernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[0], ofc->m_tempFrameHDR, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iDimY >> 1, ofc->m_iDimX >> 1, ofc->m_iChannelIdxOffset);
		hipStreamSynchronize(priv->m_csWarpStream1);
	// NV12 format
	} else if (ofc->m_iFMT == HIP_FMT) {
		hipMemcpy(ofc->m_frameSDR[0], pInBuffer[0], ofc->m_iDimY * ofc->m_iDimX, hipMemcpyDeviceToDevice);
		hipMemcpy(ofc->m_frameSDR[0] + ofc->m_iDimY * ofc->m_iDimX, pInBuffer[1], (ofc->m_iDimY / 2) * ofc->m_iDimX, hipMemcpyDeviceToDevice);
	// YUV420P format
	} else if (ofc->m_iFMT == YUV420P_FMT) {
		hipMemcpy(ofc->m_frameSDR[0], pInBuffer[0], ofc->m_iDimY * ofc->m_iDimX, hipMemcpyHostToDevice);
		hipMemcpy(ofc->m_tempFrameSDR, pInBuffer[1], (ofc->m_iDimY / 2) * (ofc->m_iDimX / 2), hipMemcpyHostToDevice);
		hipMemcpy(ofc->m_tempFrameSDR + (ofc->m_iDimY / 2) * (ofc->m_iDimX / 2), pInBuffer[2], (ofc->m_iDimY / 2) * (ofc->m_iDimX / 2), hipMemcpyHostToDevice);
		convertYUV420PtoNV12Kernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[0], ofc->m_tempFrameSDR, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iDimY >> 1, ofc->m_iDimX >> 1, ofc->m_iChannelIdxOffset);
		hipStreamSynchronize(priv->m_csWarpStream1);
	} else {
		printf("HopperRender does not support this video format: %d\n", ofc->m_iFMT);
		exit(1);
	}

	// Check for HIP Errors
	checkHIPError("updateFrame");
	
	// Blur the frame
	if (ofc->m_bIsHDR) {
		blurFrameArray(ofc, ofc->m_frameHDR[0], ofc->m_blurredFrameHDR[0], frameKernelSize, directOutput);
	} else {
		blurFrameArray(ofc, ofc->m_frameSDR[0], ofc->m_blurredFrameSDR[0], frameKernelSize, directOutput);
	}

	// Swap the frame buffers
	unsigned char* temp0 = ofc->m_frameSDR[0];
	ofc->m_frameSDR[0] = ofc->m_frameSDR[1];
	ofc->m_frameSDR[1] = ofc->m_frameSDR[2];
	ofc->m_frameSDR[2] = temp0;

	temp0 = ofc->m_blurredFrameSDR[0];
	ofc->m_blurredFrameSDR[0] = ofc->m_blurredFrameSDR[1];
	ofc->m_blurredFrameSDR[1] = ofc->m_blurredFrameSDR[2];
	ofc->m_blurredFrameSDR[2] = temp0;

	unsigned short* temp1 = ofc->m_frameHDR[0];
	ofc->m_frameHDR[0] = ofc->m_frameHDR[1];
	ofc->m_frameHDR[1] = ofc->m_frameHDR[2];
	ofc->m_frameHDR[2] = temp1;

	temp1 = ofc->m_blurredFrameHDR[0];
	ofc->m_blurredFrameHDR[0] = ofc->m_blurredFrameHDR[1];
	ofc->m_blurredFrameHDR[1] = ofc->m_blurredFrameHDR[2];
	ofc->m_blurredFrameHDR[2] = temp1;
}

/*
* Downloads the output frame from the GPU to the CPU
*
* @param ofc: Pointer to the optical flow calculator
* @param pOutBuffer: Pointer to the output buffer
*/
void downloadFrame(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer) {
	struct priv *priv = (struct priv*)ofc->priv;
	// P010 format
	if (ofc->m_bIsHDR && ofc->m_iFMT == HIP_FMT) {
		hipMemcpy(pOutBuffer[0], ofc->m_outputFrameHDR, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToDevice);
		hipMemcpy(pOutBuffer[1], ofc->m_outputFrameHDR + ofc->m_iDimY * ofc->m_iDimX, (ofc->m_iDimY >> 1) * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToDevice);
	// YUV420P10 format
	} else if (ofc->m_bIsHDR) {
		hipMemcpy(pOutBuffer[0], ofc->m_outputFrameHDR, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), hipMemcpyDeviceToHost);
		convertNV12toYUV420PKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(ofc->m_tempFrameHDR, ofc->m_outputFrameHDR, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iDimY >> 1, ofc->m_iDimX >> 1, ofc->m_iChannelIdxOffset, (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1));
		hipStreamSynchronize(priv->m_csWarpStream1);
		hipMemcpy(pOutBuffer[1], ofc->m_tempFrameHDR, (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1) * sizeof(unsigned short), hipMemcpyDeviceToHost);
		hipMemcpy(pOutBuffer[2], ofc->m_tempFrameHDR + (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1), (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1) * sizeof(unsigned short), hipMemcpyDeviceToHost);
	// NV12 format
	} else if (ofc->m_iFMT == HIP_FMT) {
		hipMemcpy(pOutBuffer[0], ofc->m_outputFrameSDR, ofc->m_iDimY * ofc->m_iDimX, hipMemcpyDeviceToDevice);
		hipMemcpy(pOutBuffer[1], ofc->m_outputFrameSDR + ofc->m_iDimY * ofc->m_iDimX, (ofc->m_iDimY >> 1) * ofc->m_iDimX, hipMemcpyDeviceToDevice);
	// YUV420P format
	} else if (ofc->m_iFMT == YUV420P_FMT) {
		hipMemcpy(pOutBuffer[0], ofc->m_outputFrameSDR, ofc->m_iDimY * ofc->m_iDimX, hipMemcpyDeviceToHost);
		convertNV12toYUV420PKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(ofc->m_tempFrameSDR, ofc->m_outputFrameSDR, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iDimY >> 1, ofc->m_iDimX >> 1, ofc->m_iChannelIdxOffset, (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1));
		hipStreamSynchronize(priv->m_csWarpStream1);
		hipMemcpy(pOutBuffer[1], ofc->m_tempFrameSDR, (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1), hipMemcpyDeviceToHost);
		hipMemcpy(pOutBuffer[2], ofc->m_tempFrameSDR + (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1), (ofc->m_iDimY >> 1) * (ofc->m_iDimX >> 1), hipMemcpyDeviceToHost);
	} else {
		printf("HopperRender does not support this video format: %d\n", ofc->m_iFMT);
		exit(1);
	}
	
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
	if (ofc->m_bIsHDR) {
		if (ofc->m_fBlackLevel == 0.0f && ofc->m_fWhiteLevel == 1023.0f) {
			hipMemcpy(ofc->m_outputFrameHDR, firstFrame ? ofc->m_frameHDR[2] : ofc->m_frameHDR[1], ofc->m_iDimY * ofc->m_iDimX * 3, hipMemcpyDeviceToDevice);
			downloadFrame(ofc, pOutBuffer);
		} else {
			processFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(firstFrame ? ofc->m_frameHDR[2] : ofc->m_frameHDR[1],
													ofc->m_outputFrameHDR,
													ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
			hipStreamSynchronize(priv->m_csWarpStream1);
			hipMemcpy(ofc->m_outputFrameHDR + ofc->m_iChannelIdxOffset, (firstFrame ? ofc->m_frameHDR[2] : ofc->m_frameHDR[1]) + ofc->m_iChannelIdxOffset, ofc->m_iDimY * (ofc->m_iDimX >> 1) * sizeof(unsigned short), hipMemcpyDeviceToDevice);
			downloadFrame(ofc, pOutBuffer);
		}
	} else {
		if (ofc->m_fBlackLevel == 0.0f && ofc->m_fWhiteLevel == 255.0f) {
			hipMemcpy(ofc->m_outputFrameSDR, firstFrame ? ofc->m_frameSDR[2] : ofc->m_frameSDR[1], ofc->m_iDimY * ofc->m_iDimX * 1.5, hipMemcpyDeviceToDevice);
			downloadFrame(ofc, pOutBuffer);
		} else {
			processFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(firstFrame ? ofc->m_frameSDR[2] : ofc->m_frameSDR[1],
													ofc->m_outputFrameSDR,
													ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
			hipStreamSynchronize(priv->m_csWarpStream1);
			hipMemcpy(ofc->m_outputFrameSDR + ofc->m_iChannelIdxOffset, (firstFrame ? ofc->m_frameSDR[2] : ofc->m_frameSDR[1]) + ofc->m_iChannelIdxOffset, ofc->m_iDimY * (ofc->m_iDimX >> 1), hipMemcpyDeviceToDevice);
			downloadFrame(ofc, pOutBuffer);
		}
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
	hipMemset(ofc->m_offsetArray12, 0, ofc->m_iLayerIdxOffset * sizeof(int));
	// Set layers 0-5 of the Y-Dir to 0
	hipMemset(ofc->m_offsetArray12 + ofc->m_iDirectionIdxOffset, 0, ofc->m_iDirectionIdxOffset * sizeof(int));
	// Set layers 1-4 of the X-Dir to -2,-1,1,2
	setInitialOffset<<<priv->m_lowGrid16x16x4, priv->m_threads16x16x1, 0, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iLayerIdxOffset);
	hipStreamSynchronize(priv->m_csOFCStream1);
	
	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (unsigned int iter = 0; iter < iNumIterations; iter++) {
		// Calculate the number of steps for this iteration executed to find the ideal offset (limits the maximum offset)
	    //iNumStepsPerIter = static_cast<unsigned int>(static_cast<double>(iNumSteps) - static_cast<double>(iter) * (static_cast<double>(iNumSteps) / static_cast<double>(iNumIterations)));
		iNumStepsPerIter = std::max(static_cast<int>(static_cast<double>(iNumSteps) * exp(-static_cast<double>(3 * iter) / static_cast<double>(iNumIterations))), 1);

		// Each step we adjust the offset array to find the ideal offset
		for (unsigned int step = 0; step < iNumStepsPerIter; step++) {
			// Reset the summed up delta array
			hipMemset(ofc->m_summedUpDeltaArray, 0, 5 * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(unsigned int));

			// 1. Calculate the image delta and sum up the deltas of each window
			if (ofc->m_bIsHDR) {
				calcDeltaSums<<<iter == 0 ? priv->m_lowGrid16x16x5 : priv->m_lowGrid8x8x5, iter == 0 ? priv->m_threads16x16x1 : priv->m_threads8x8x1, sharedMemSize, priv->m_csOFCStream1>>>(ofc->m_summedUpDeltaArray, 
																ofc->m_blurredFrameHDR[1],
                                                                ofc->m_blurredFrameHDR[2],
															    ofc->m_offsetArray12, ofc->m_iLayerIdxOffset, ofc->m_iDirectionIdxOffset,
																ofc->m_iDimY, ofc->m_iDimX, ofc->m_iLowDimY, ofc->m_iLowDimX, windowDim, ofc->m_cResolutionScalar);
			} else {
				calcDeltaSums<<<iter == 0 ? priv->m_lowGrid16x16x5 : priv->m_lowGrid8x8x5, iter == 0 ? priv->m_threads16x16x1 : priv->m_threads8x8x1, sharedMemSize, priv->m_csOFCStream1>>>(ofc->m_summedUpDeltaArray, 
																ofc->m_blurredFrameSDR[1],
                                                                ofc->m_blurredFrameSDR[2],
															    ofc->m_offsetArray12, ofc->m_iLayerIdxOffset, ofc->m_iDirectionIdxOffset,
																ofc->m_iDimY, ofc->m_iDimX, ofc->m_iLowDimY, ofc->m_iLowDimX, windowDim, ofc->m_cResolutionScalar);
			}
			
			hipStreamSynchronize(priv->m_csOFCStream1);

			// 2. Normalize the summed up delta array and find the best layer
			normalizeDeltaSums<<<priv->m_lowGrid8x8x1, priv->m_threads8x8x5, 0, priv->m_csOFCStream1>>>(ofc->m_summedUpDeltaArray, ofc->m_lowestLayerArray,
															   ofc->m_offsetArray12, windowDim, windowDim * windowDim,
															   ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset, ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX);
			hipStreamSynchronize(priv->m_csOFCStream1);

			// 3. Adjust the offset array based on the comparison results
			adjustOffsetArray<<<priv->m_lowGrid32x32x1, priv->m_threads32x32x1, 0, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_lowestLayerArray,
															  ofc->m_statusArray, windowDim, ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset,
															  ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX, step == iNumStepsPerIter - 1);
			hipStreamSynchronize(priv->m_csOFCStream1);
		}

		// 4. Adjust variables for the next iteration
		windowDim = std::max(windowDim >> 1, (unsigned int)1);
		sharedMemSize = 8 * 8 * sizeof(unsigned int);
		if (windowDim == 1) sharedMemSize = 0;

		// Reset the status array
		hipMemset(ofc->m_statusArray, 0, ofc->m_iLowDimY * ofc->m_iLowDimX);
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
	hipMemset(ofc->m_hitCount12, 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(int));
	hipMemset(ofc->m_hitCount21, 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(int));

	// #####################
	// ###### WARPING ######
	// #####################
	// Frame 1 to Frame 2
	if (outputMode != 1) {
		if (ofc->m_bIsHDR) {
			warpFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[0],
																									 ofc->m_blurredOffsetArray12[0],
																									 ofc->m_hitCount12,
																									 (outputMode < 2) ? ofc->m_outputFrameHDR : ofc->m_warpedFrame12HDR,
																									 frameScalar12,
																									 ofc->m_iLowDimY,
																									 ofc->m_iLowDimX,
																									 ofc->m_iDimY,
																									 ofc->m_iDimX, ofc->m_iRealDimX,
																									 ofc->m_cResolutionScalar,
																									 ofc->m_iLayerIdxOffset,
																									 ofc->m_iChannelIdxOffset);
		} else {
			warpFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[0],
																									 ofc->m_blurredOffsetArray12[0],
																									 ofc->m_hitCount12,
																									 (outputMode < 2) ? ofc->m_outputFrameSDR : ofc->m_warpedFrame12SDR,
																									 frameScalar12,
																									 ofc->m_iLowDimY,
																									 ofc->m_iLowDimX,
																									 ofc->m_iDimY,
																									 ofc->m_iDimX, ofc->m_iRealDimX,
																									 ofc->m_cResolutionScalar,
																									 ofc->m_iLayerIdxOffset,
																									 ofc->m_iChannelIdxOffset);
		}
		
	}
	// Frame 2 to Frame 1
	if (outputMode != 0) {
		if (ofc->m_bIsHDR) {
			warpFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream2>>>(ofc->m_frameHDR[1],
																									 ofc->m_blurredOffsetArray21[0],
																									 ofc->m_hitCount21,
																									 (outputMode < 2) ? ofc->m_outputFrameHDR : ofc->m_warpedFrame21HDR,
																									 frameScalar21,
																									 ofc->m_iLowDimY,
																									 ofc->m_iLowDimX,
																									 ofc->m_iDimY,
																									 ofc->m_iDimX, ofc->m_iRealDimX,
																									 ofc->m_cResolutionScalar,
																									 ofc->m_iLayerIdxOffset,
																									 ofc->m_iChannelIdxOffset);
		} else {
			warpFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream2>>>(ofc->m_frameSDR[1],
																									 ofc->m_blurredOffsetArray21[0],
																									 ofc->m_hitCount21,
																									 (outputMode < 2) ? ofc->m_outputFrameSDR : ofc->m_warpedFrame21SDR,
																									 frameScalar21,
																									 ofc->m_iLowDimY,
																									 ofc->m_iLowDimX,
																									 ofc->m_iDimY,
																									 ofc->m_iDimX, ofc->m_iRealDimX,
																									 ofc->m_cResolutionScalar,
																									 ofc->m_iLayerIdxOffset,
																									 ofc->m_iChannelIdxOffset);
		}
	}
	if (outputMode != 1) hipStreamSynchronize(priv->m_csWarpStream1);
	if (outputMode != 0) hipStreamSynchronize(priv->m_csWarpStream2);
	
	// ##############################
	// ###### ARTIFACT REMOVAL ######
	// ##############################
	// Frame 1 to Frame 2
	if (outputMode != 1) {
		if (ofc->m_bIsHDR) {
			artifactRemovalKernel<<<priv->m_grid8x8x1, priv->m_threads8x8x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[0],
																									   ofc->m_hitCount12,
																									   (outputMode < 2) ? ofc->m_outputFrameHDR : ofc->m_warpedFrame12HDR,
																									   ofc->m_iDimY,
																									   ofc->m_iDimX, ofc->m_iRealDimX,
																									   ofc->m_iChannelIdxOffset);
		} else {
			artifactRemovalKernel<<<priv->m_grid8x8x1, priv->m_threads8x8x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[0],
																									   ofc->m_hitCount12,
																									   (outputMode < 2) ? ofc->m_outputFrameSDR : ofc->m_warpedFrame12SDR,
																									   ofc->m_iDimY,
																									   ofc->m_iDimX, ofc->m_iRealDimX,
																									   ofc->m_iChannelIdxOffset);
		}
	}
	// Frame 2 to Frame 1
	if (outputMode != 0) {
		if (ofc->m_bIsHDR) {
			artifactRemovalKernel<<<priv->m_grid8x8x1, priv->m_threads8x8x2, 0, priv->m_csWarpStream2>>>(ofc->m_frameHDR[1],
																									   ofc->m_hitCount21,
																									   (outputMode < 2) ? ofc->m_outputFrameHDR : ofc->m_warpedFrame21HDR,
																									   ofc->m_iDimY,
																									   ofc->m_iDimX, ofc->m_iRealDimX,
																									   ofc->m_iChannelIdxOffset);
		} else {
			artifactRemovalKernel<<<priv->m_grid8x8x1, priv->m_threads8x8x2, 0, priv->m_csWarpStream2>>>(ofc->m_frameSDR[1],
																									   ofc->m_hitCount21,
																									   (outputMode < 2) ? ofc->m_outputFrameSDR : ofc->m_warpedFrame21SDR,
																									   ofc->m_iDimY,
																									   ofc->m_iDimX, ofc->m_iRealDimX,
																									   ofc->m_iChannelIdxOffset);
		}
	}
	if (outputMode != 1) hipStreamSynchronize(priv->m_csWarpStream1);
	if (outputMode != 0) hipStreamSynchronize(priv->m_csWarpStream2);

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
	if (ofc->m_bIsHDR) {
		blendFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_warpedFrame12HDR, ofc->m_warpedFrame21HDR,
												 ofc->m_outputFrameHDR, frame1Scalar, frame2Scalar,
	                                             ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX, ofc->m_iChannelIdxOffset, ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
	} else {
		blendFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_warpedFrame12SDR, ofc->m_warpedFrame21SDR,
												 ofc->m_outputFrameSDR, frame1Scalar, frame2Scalar,
	                                             ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX, ofc->m_iChannelIdxOffset, ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
	}

	hipStreamSynchronize(priv->m_csWarpStream1);

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
	if (ofc->m_bIsHDR) {
		insertFrameKernel<<<priv->m_halfGrid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[0],
												 ofc->m_outputFrameHDR,
	                                             ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX, ofc->m_iChannelIdxOffset,
												 ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
	} else {
		insertFrameKernel<<<priv->m_halfGrid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[0],
												 ofc->m_outputFrameSDR,
	                                             ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX, ofc->m_iChannelIdxOffset,
												 ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
	}
	hipStreamSynchronize(priv->m_csWarpStream1);

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
	const unsigned int halfDimX = ofc->m_iRealDimX >> 1;
	const unsigned int halfDimY = ofc->m_iDimY >> 1;

	if (ofc->m_bIsHDR) {
		if (frameCounter == 1) {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[2], ofc->m_frameHDR[2], 
														ofc->m_frameHDR[2],
														ofc->m_outputFrameHDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		} else if (frameCounter == 2) {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[1], ofc->m_frameHDR[1], 
														ofc->m_frameHDR[1],
														ofc->m_outputFrameHDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		} else if (frameCounter <= 3) {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[0], ofc->m_frameHDR[0], 
														ofc->m_frameHDR[0],
														ofc->m_outputFrameHDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		} else {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameHDR[0], ofc->m_warpedFrame12HDR, 
														ofc->m_warpedFrame21HDR,
														ofc->m_outputFrameHDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		}
	} else {
		if (frameCounter == 1) {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[2], ofc->m_frameSDR[2], 
														ofc->m_frameSDR[2],
														ofc->m_outputFrameSDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		} else if (frameCounter == 2) {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[1], ofc->m_frameSDR[1], 
														ofc->m_frameSDR[1],
														ofc->m_outputFrameSDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		} else if (frameCounter <= 3) {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[0], ofc->m_frameSDR[0], 
														ofc->m_frameSDR[0],
														ofc->m_outputFrameSDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		} else {
			sideBySideFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_frameSDR[0], ofc->m_warpedFrame12SDR, 
														ofc->m_warpedFrame21SDR,
														ofc->m_outputFrameSDR, frame1Scalar, frame2Scalar,
														ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,halfDimY, halfDimX, ofc->m_iChannelIdxOffset,
														ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal, ofc->m_iMiddleValue);
		}
	}
	hipStreamSynchronize(priv->m_csWarpStream1);

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
	if (ofc->m_bIsHDR) {
		convertFlowToHSVKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_blurredOffsetArray12[0], ofc->m_outputFrameHDR,
														ofc->m_frameHDR[1], blendScalar, ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,
														ofc->m_cResolutionScalar, ofc->m_iLayerIdxOffset, ofc->m_iChannelIdxOffset, ofc->m_iFMT == HIP_FMT);
	} else {
		convertFlowToHSVKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_blurredOffsetArray12[0], ofc->m_outputFrameSDR,
														ofc->m_frameSDR[1], blendScalar, ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,
														ofc->m_cResolutionScalar, ofc->m_iLayerIdxOffset, ofc->m_iChannelIdxOffset, ofc->m_iFMT == HIP_FMT);
	}
	
	hipStreamSynchronize(priv->m_csWarpStream1);

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
	if (ofc->m_bIsHDR) {
		convertFlowToGreyscaleKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_blurredOffsetArray12[0], ofc->m_outputFrameHDR,
														ofc->m_frameHDR[1], ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,
														ofc->m_cResolutionScalar, ofc->m_iLayerIdxOffset, ofc->m_iChannelIdxOffset, ofc->m_iFMT == HIP_FMT, (int)ofc->m_fMaxVal, ofc->m_iMiddleValue);
	} else {
		convertFlowToGreyscaleKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x2, 0, priv->m_csWarpStream1>>>(ofc->m_blurredOffsetArray12[0], ofc->m_outputFrameSDR,
														ofc->m_frameSDR[1], ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_iDimY, ofc->m_iDimX, ofc->m_iRealDimX,
														ofc->m_cResolutionScalar, ofc->m_iLayerIdxOffset, ofc->m_iChannelIdxOffset, ofc->m_iFMT == HIP_FMT, (int)ofc->m_fMaxVal, ofc->m_iMiddleValue);
	}
	
	hipStreamSynchronize(priv->m_csWarpStream1);

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
	hipMemset(ofc->m_offsetArray21, 0, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(int));

	// Launch kernel
	flipFlowKernel<<<priv->m_lowGrid16x16x1, priv->m_threads16x16x2, 0, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_offsetArray21,
												            ofc->m_iLowDimY, ofc->m_iLowDimX, ofc->m_cResolutionScalar, ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset);
	hipStreamSynchronize(priv->m_csOFCStream1);

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
		hipMemcpy(ofc->m_blurredOffsetArray12[1], ofc->m_offsetArray12, ofc->m_iLayerIdxOffset * sizeof(int), hipMemcpyDeviceToDevice);
		// Offset12 Y-Dir
		hipMemcpy(ofc->m_blurredOffsetArray12[1] + ofc->m_iLayerIdxOffset, ofc->m_offsetArray12 + ofc->m_iDirectionIdxOffset, ofc->m_iLayerIdxOffset * sizeof(int), hipMemcpyDeviceToDevice);
		// Offset21 X&Y-Dir
		hipMemcpy(ofc->m_blurredOffsetArray21[1], ofc->m_offsetArray21, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(int), hipMemcpyDeviceToDevice);
	} else {
		// Launch kernels
		blurFlowKernel<<<gridBF, threadsBF, sharedMemSize, priv->m_csOFCStream1>>>(ofc->m_offsetArray12, ofc->m_blurredOffsetArray12[1], ofc->m_iFlowBlurKernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, start, end, pixelCount, ofc->m_iNumLayers, ofc->m_iLowDimY, ofc->m_iLowDimX);
		blurFlowKernel<<<gridBF, threadsBF, sharedMemSize, priv->m_csOFCStream2>>>(ofc->m_offsetArray21, ofc->m_blurredOffsetArray21[1], ofc->m_iFlowBlurKernelSize, chacheSize, boundsOffset, avgEntriesPerThread, remainder, start, end, pixelCount, 1, ofc->m_iLowDimY, ofc->m_iLowDimX);
		//cleanFlowKernel<<<m_lowGrid16x16x1, m_threads16x16x2, 0, blurStream1>>>(m_offsetArray12, m_blurredOffsetArray12, m_iLowDimY, m_iLowDimX);
		//cleanFlowKernel<<<m_lowGrid16x16x1, m_threads16x16x2, 0, blurStream2>>>(m_offsetArray21, m_blurredOffsetArray21, m_iLowDimY, m_iLowDimX);

		// Synchronize streams to ensure completion
		hipStreamSynchronize(priv->m_csOFCStream1);
		hipStreamSynchronize(priv->m_csOFCStream2);
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
	// We don't save HDR images
	if (ofc->m_bIsHDR)
		return;

	// Copy the image array to the CPU
	size_t dataSize = 1.5 * ofc->m_iDimY * ofc->m_iDimX;
	hipMemcpy(ofc->m_imageArrayCPU, ofc->m_outputFrameSDR, dataSize, hipMemcpyDeviceToHost);

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
* Creates a new GPUArray of type unsigned char
*
* @param size: Number of entries in the array
*/
unsigned char* createGPUArrayUC(const size_t size) {
	// Allocate VRAM
	unsigned char* arrayPtrGPU;
	hipMalloc(&arrayPtrGPU, size);

	// Set all entries to 0
	hipMemset(arrayPtrGPU, 0, size);

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
	hipMalloc(&arrayPtrGPU, size * sizeof(unsigned short));

	// Set all entries to 0
	hipMemset(arrayPtrGPU, 0, size * sizeof(unsigned short));

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
	hipMalloc(&arrayPtrGPU, size * sizeof(int));

	// Set all entries to 0
	hipMemset(arrayPtrGPU, 0, size * sizeof(int));

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
	hipMalloc(&arrayPtrGPU, size * sizeof(unsigned int));

	// Set all entries to 0
	hipMemset(arrayPtrGPU, 0, size * sizeof(unsigned int));

	return arrayPtrGPU;
}

/*
* Initializes the optical flow calculator
*
* @param ofc: Pointer to the optical flow calculator
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param realDimX: The real width of the frame (not the stride width!)
* @param resolutionScalar: The scalar to reduce the resolution by
* @param flowBlurKernelSize: The size of the kernel to use for the flow blur
* @param isHDR: Whether the frames are in HDR format
* @param fmt: The format of the frames
*/
void initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int dimY, const int dimX, const int realDimX, unsigned char resolutionScalar, unsigned int flowBlurKernelSize, bool isHDR, int fmt) {
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

	// Video properties
	ofc->m_iDimX = dimX;
	ofc->m_iDimY = dimY;
	ofc->m_iRealDimX = realDimX;
	ofc->m_fBlackLevel = 0.0f;
	ofc->m_bIsHDR = isHDR;
	ofc->m_iFMT = fmt;
	// P010 format
	if (isHDR && fmt == HIP_FMT) {
		ofc->m_fMaxVal = 65535.0f;
		ofc->m_iMiddleValue = 32768;
	// YUV420P10 format
	} else if (isHDR) {
		ofc->m_fMaxVal = 1023.0f;
		ofc->m_iMiddleValue = 512;
	// YUV420P or NV12 format
	} else {
		ofc->m_fMaxVal = 255.0f;
		ofc->m_iMiddleValue = 128;
	}
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
	if (ofc->m_bIsHDR) {
		ofc->m_frameHDR[0] = createGPUArrayUS(1.5 * dimY * dimX);
		ofc->m_frameHDR[1] = createGPUArrayUS(1.5 * dimY * dimX);
		ofc->m_frameHDR[2] = createGPUArrayUS(1.5 * dimY * dimX);
		ofc->m_blurredFrameHDR[0] = createGPUArrayUS(dimY * dimX);
		ofc->m_blurredFrameHDR[1] = createGPUArrayUS(dimY * dimX);
		ofc->m_blurredFrameHDR[2] = createGPUArrayUS(dimY * dimX);
		ofc->m_warpedFrame12HDR = createGPUArrayUS(1.5 * dimY * dimX);
		ofc->m_warpedFrame21HDR = createGPUArrayUS(1.5 * dimY * dimX);
		ofc->m_outputFrameHDR = createGPUArrayUS(1.5 * dimY * dimX);
		ofc->m_tempFrameHDR = createGPUArrayUS((dimY / 2) * realDimX);
	} else {
		ofc->m_frameSDR[0] = createGPUArrayUC(1.5 * dimY * dimX);
		ofc->m_frameSDR[1] = createGPUArrayUC(1.5 * dimY * dimX);
		ofc->m_frameSDR[2] = createGPUArrayUC(1.5 * dimY * dimX);
		ofc->m_blurredFrameSDR[0] = createGPUArrayUC(dimY * dimX);
		ofc->m_blurredFrameSDR[1] = createGPUArrayUC(dimY * dimX);
		ofc->m_blurredFrameSDR[2] = createGPUArrayUC(dimY * dimX);
		ofc->m_warpedFrame12SDR = createGPUArrayUC(1.5 * dimY * dimX);
		ofc->m_warpedFrame21SDR = createGPUArrayUC(1.5 * dimY * dimX);
		ofc->m_outputFrameSDR = createGPUArrayUC(1.5 * dimY * dimX);
		ofc->m_tempFrameSDR = createGPUArrayUC((dimY / 2) * realDimX);
		ofc->m_imageArrayCPU = (unsigned char*)malloc(1.5 * dimY * dimX);
	}
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
	hipStreamCreate(&priv->m_csOFCStream1);
	hipStreamCreate(&priv->m_csOFCStream2);
	hipStreamCreate(&priv->m_csWarpStream1);
	hipStreamCreate(&priv->m_csWarpStream2);
}

/*
* Template instantiation
*/
template __global__ void convertNV12toYUV420PKernel(unsigned char* outputFrame, const unsigned char* inputFrame, const unsigned int dimY, 
										   const unsigned int dimX, const unsigned int halfDimY, 
										   const unsigned int halfDimX, const unsigned int channelIdxOffset, const unsigned int secondChannelIdxOffset);
template __global__ void convertNV12toYUV420PKernel(unsigned short* outputFrame, const unsigned short* inputFrame, const unsigned int dimY, 
										   const unsigned int dimX, const unsigned int halfDimY, 
										   const unsigned int halfDimX, const unsigned int channelIdxOffset, const unsigned int secondChannelIdxOffset);

template __global__ void convertYUV420PtoNV12Kernel(unsigned char* outputFrame, const unsigned char* inputFrame, const unsigned int dimY, 
										   const unsigned int dimX, const unsigned int halfDimY, 
										   const unsigned int halfDimX, const unsigned int channelIdxOffset);
template __global__ void convertYUV420PtoNV12Kernel(unsigned short* outputFrame, const unsigned short* inputFrame, const unsigned int dimY, 
										   const unsigned int dimX, const unsigned int halfDimY, 
										   const unsigned int halfDimX, const unsigned int channelIdxOffset);

template __global__ void processFrameKernel(const unsigned char* frame, unsigned char* outputFrame,
                                 const unsigned int dimY,
                                 const unsigned int dimX, const unsigned int realDimX, const float blackLevel, const float whiteLevel, const float maxVal);
template __global__ void processFrameKernel(const unsigned short* frame, unsigned short* outputFrame,
                                 const unsigned int dimY,
                                 const unsigned int dimX, const unsigned int realDimX, const float blackLevel, const float whiteLevel, const float maxVal);

template __global__ void blurFrameKernel(const unsigned char* frameArray, unsigned char* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const unsigned short dimY, const unsigned short dimX, const unsigned int realDimX);
template __global__ void blurFrameKernel(const unsigned short* frameArray, unsigned short* blurredFrameArray, 
								const unsigned char kernelSize, const unsigned char chacheSize, const unsigned char boundsOffset, 
								const unsigned char avgEntriesPerThread, const unsigned short remainder, const char lumStart,
								const unsigned char lumEnd, const unsigned short lumPixelCount, const unsigned short dimY, const unsigned short dimX, const unsigned int realDimX);

template __global__ void calcDeltaSums(unsigned int* summedUpDeltaArray, const unsigned char* frame1, const unsigned char* frame2,
							  const int* offsetArray, const unsigned int layerIdxOffset, const unsigned int directionIdxOffset,
						      const unsigned int dimY, const unsigned int dimX, const unsigned int lowDimY, const unsigned int lowDimX,
							  const unsigned int windowDim, const unsigned char resolutionScalar);
template __global__ void calcDeltaSums(unsigned int* summedUpDeltaArray, const unsigned short* frame1, const unsigned short* frame2,
							  const int* offsetArray, const unsigned int layerIdxOffset, const unsigned int directionIdxOffset,
						      const unsigned int dimY, const unsigned int dimX, const unsigned int lowDimY, const unsigned int lowDimX,
							  const unsigned int windowDim, const unsigned char resolutionScalar);

template __global__ void warpFrameKernel(const unsigned char* frame1, const int* offsetArray, int* hitCount,
								unsigned char* warpedFrame, const float frameScalar, const unsigned int lowDimY, const unsigned int lowDimX,
								const unsigned int dimY, const int dimX, const unsigned int realDimX, const unsigned char resolutionScalar,
								const unsigned int directionIdxOffset, const unsigned int channelIdxOffset);
template __global__ void warpFrameKernel(const unsigned short* frame1, const int* offsetArray, int* hitCount,
								unsigned short* warpedFrame, const float frameScalar, const unsigned int lowDimY, const unsigned int lowDimX,
								const unsigned int dimY, const int dimX, const unsigned int realDimX, const unsigned char resolutionScalar,
								const unsigned int directionIdxOffset, const unsigned int channelIdxOffset);

template __global__ void artifactRemovalKernel(const unsigned char* frame1, const int* hitCount, unsigned char* warpedFrame,
												 const unsigned int dimY, const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset);
template __global__ void artifactRemovalKernel(const unsigned short* frame1, const int* hitCount, unsigned short* warpedFrame,
												 const unsigned int dimY, const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset);

template __global__ void blendFrameKernel(const unsigned char* warpedFrame1, const unsigned char* warpedFrame2, unsigned char* outputFrame,
                                 const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                 const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset, const float blackLevel, const float whiteLevel, const float maxVal);
template __global__ void blendFrameKernel(const unsigned short* warpedFrame1, const unsigned short* warpedFrame2, unsigned short* outputFrame,
                                 const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                 const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset, const float blackLevel, const float whiteLevel, const float maxVal);

template __global__ void insertFrameKernel(const unsigned char* frame1, unsigned char* outputFrame, const unsigned int dimY,
                                  const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset, const float blackLevel, const float whiteLevel, const float maxVal);
template __global__ void insertFrameKernel(const unsigned short* frame1, unsigned short* outputFrame, const unsigned int dimY,
								  const unsigned int dimX, const unsigned int realDimX, const unsigned int channelIdxOffset, const float blackLevel, const float whiteLevel, const float maxVal);

template __global__ void sideBySideFrameKernel(const unsigned char* frame1, const unsigned char* warpedFrame1, const unsigned char* warpedFrame2, unsigned char* outputFrame, 
									  const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
                                      const unsigned int dimX, const unsigned int realDimX, const unsigned int halfDimY, 
									  const unsigned int halfDimX, const unsigned int channelIdxOffset,
									  const float blackLevel, const float whiteLevel, const float maxVal, const unsigned short middleValue);
template __global__ void sideBySideFrameKernel(const unsigned short* frame1, const unsigned short* warpedFrame1, const unsigned short* warpedFrame2, unsigned short* outputFrame,
									  const float frame1Scalar, const float frame2Scalar, const unsigned int dimY,
									  const unsigned int dimX, const unsigned int realDimX, const unsigned int halfDimY, 
									  const unsigned int halfDimX, const unsigned int channelIdxOffset,
									  const float blackLevel, const float whiteLevel, const float maxVal, const unsigned short middleValue);

template __global__ void convertFlowToHSVKernel(const int* flowArray, unsigned char* outputFrame, const unsigned char* frame1,
                                       const float blendScalar, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX, 
									   const unsigned int realDimX, const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset, const bool isP010);
template __global__ void convertFlowToHSVKernel(const int* flowArray, unsigned short* outputFrame, const unsigned short* frame1,
									   const float blendScalar, const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX,
									   const unsigned int realDimX, const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset, const bool isP010);

template __global__ void convertFlowToGreyscaleKernel(const int* flowArray, unsigned char* outputFrame, const unsigned char* frame1,
                                       const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX, 
									   const unsigned int realDimX, const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset, const bool isP010, const int maxVal, const unsigned short middleValue);
template __global__ void convertFlowToGreyscaleKernel(const int* flowArray, unsigned short* outputFrame, const unsigned short* frame1,
									   const unsigned int lowDimY, const unsigned int lowDimX, const unsigned int dimY, const unsigned int dimX,
									   const unsigned int realDimX, const unsigned char resolutionScalar, const unsigned int directionIdxOffset,
									   const unsigned int channelIdxOffset, const bool isP010, const int maxVal, const unsigned short middleValue);