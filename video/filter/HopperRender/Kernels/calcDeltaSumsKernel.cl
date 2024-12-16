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
__kernel void calcDeltaSumsKernel(__global unsigned int* summedUpDeltaArray, __global const unsigned short* frame1,
                                  __global const unsigned short* frame2, __global const short* offsetArray,
                                  __global const unsigned char* lowestLayerArray, const int directionIndexOffset,
                                  const int dimY, const int dimX, const int lowDimY, const int lowDimX,
                                  const int windowSize, const int resolutionScalar, const int isFirstIteration) {
    // Shared memory for the partial sums of the current block
    __local unsigned int partialSums[64];

    // Current entry to be computed by the thread
    int cx = get_global_id(0);
    int cy = get_global_id(1);
    cx = min(cx, lowDimX - 1);
    cy = min(cy, lowDimY - 1);
    const int cz = get_global_id(2);
    const int tIdx = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int layerOffset = cz * 2 * lowDimY * lowDimX;  // Offset to index the layer of the current thread
    const int scaledCx = cx << resolutionScalar;         // The X-Index of the current thread in the input frames
    const int scaledCy = cy << resolutionScalar;         // The Y-Index of the current thread in the input frames
    const int threadIndex2D = cy * lowDimX + cx;         // Standard thread index without Z-Dim
    unsigned int offsetBias = 0;                         // Bias to discourage unnecessary offset
    unsigned int neighborBias = 0;                       // Bias to discourage non-uniform flow

    // Retrieve the offset values for the current thread that are going to be tested
    short offsetX = offsetArray[layerOffset + threadIndex2D];
    short offsetY = offsetArray[directionIndexOffset + layerOffset + threadIndex2D];
    int newCx = scaledCx + offsetX;
    int newCy = scaledCy + offsetY;

    // Calculate the delta value for the current pixel
    unsigned int delta = abs((int)frame1[scaledCy * dimX + scaledCx] - (int)frame2[newCy * dimX + newCx]);

    // Retrieve the layer that contains the ideal offset values for the current window
    if (!isFirstIteration) {
        const int wx = (cx / windowSize) * windowSize;
        const int wy = (cy / windowSize) * windowSize;
        const unsigned char lowestLayer = lowestLayerArray[wy * lowDimX + wx];
        const int idealLayerOffset = (int)lowestLayer * 2 * lowDimY * lowDimX;

        // Retrieve the ideal offset values of the neighboring windows
        short neighborOffsetXDown = cy + windowSize < lowDimY ? offsetArray[idealLayerOffset + (cy + windowSize) * lowDimX + cx] : 0;
        short neighborOffsetYDown = cy + windowSize < lowDimY ? offsetArray[directionIndexOffset + idealLayerOffset + (cy + windowSize) * lowDimX + cx] : 0;
        short neighborOffsetXRight = cx + windowSize < lowDimX ? offsetArray[idealLayerOffset + cy * lowDimX + cx + windowSize] : 0;
        short neighborOffsetYRight = cx + windowSize < lowDimX ? offsetArray[directionIndexOffset + idealLayerOffset + cy * lowDimX + cx + windowSize] : 0;
        short neighborOffsetXLeft = cx - windowSize >= 0 ? offsetArray[idealLayerOffset + cy * lowDimX + cx - windowSize] : 0;
        short neighborOffsetYLeft = cx - windowSize >= 0 ? offsetArray[directionIndexOffset + idealLayerOffset + cy * lowDimX + cx - windowSize] : 0;
        short neighborOffsetXUp = cy - windowSize >= 0 ? offsetArray[idealLayerOffset + (cy - windowSize) * lowDimX + cx] : 0;
        short neighborOffsetYUp = cy - windowSize >= 0 ? offsetArray[directionIndexOffset + idealLayerOffset + (cy - windowSize) * lowDimX + cx] : 0;

        // Collect the offset and neighbor biases that will be used to discourage unnecessary offset and non-uniform flow
        offsetBias = abs(offsetX) + abs(offsetY);
        neighborBias = (abs(neighborOffsetXDown - offsetX) + abs(neighborOffsetYDown - offsetY) +
                        abs(neighborOffsetXRight - offsetX) + abs(neighborOffsetYRight - offsetY) +
                        abs(neighborOffsetXLeft - offsetX) + abs(neighborOffsetYLeft - offsetY) +
                        abs(neighborOffsetXUp - offsetX) + abs(neighborOffsetYUp - offsetY))
                       << 2;
    }

    if (windowSize == 1) {
        // Window size of 1x1
        summedUpDeltaArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] =
            (scaledCy < 0 || scaledCy >= dimY || scaledCx < 0 || scaledCx >= dimX || newCy < 0 || newCx < 0 ||
             newCy >= dimY || newCx >= dimX)
                ? 0
                : delta + offsetBias + neighborBias;
        return;
    } else {
        // All other window sizes
        partialSums[tIdx] = (scaledCy < 0 || scaledCy >= dimY || scaledCx < 0 || scaledCx >= dimX || newCy < 0 ||
                             newCx < 0 || newCy >= dimY || newCx >= dimX)
                                ? 0
                                : delta + offsetBias + neighborBias;
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
    if (windowSize >= 8) {
        // Window size of 8x8 or larger
        if (tIdx < 32) {
            warpReduce8x8(partialSums, tIdx);
        }
    } else if (windowSize == 4) {
        // Window size of 4x4
        if (get_local_id(1) < 2) {
            // Top 4x4 Blocks
            warpReduce4x4(partialSums, tIdx);
        } else if (get_local_id(1) >= 4 && get_local_id(1) < 6) {
            // Bottom 4x4 Blocks
            warpReduce4x4(partialSums, tIdx);
        }
    } else if (windowSize == 2) {
        // Window size of 2x2
        if ((get_local_id(1) & 1) == 0) {
            warpReduce2x2(partialSums, tIdx);
        }
    }

    // Sync all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sum up the results of all blocks
    if ((windowSize >= 8 && tIdx == 0) || (windowSize == 4 && (tIdx == 0 || tIdx == 4 || tIdx == 32 || tIdx == 36)) ||
        (windowSize == 2 && ((tIdx & 1) == 0 && (get_local_id(1) & 1) == 0))) {
        const int windowIndexX = cx / windowSize;
        const int windowIndexY = cy / windowSize;
        atomic_add(&summedUpDeltaArray[cz * lowDimY * lowDimX + (windowIndexY * windowSize) * lowDimX +
                                       (windowIndexX * windowSize)],
                   partialSums[tIdx]);
    }
}