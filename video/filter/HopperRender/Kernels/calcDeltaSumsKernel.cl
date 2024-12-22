// Helper function to get neighbor offset values
inline short getNeighborOffset(__global const short *offsetArray, int neighborIndexX, int neighborIndexY, 
                               int lowDimX, int lowDimY, int directionIndexOffset) {
    return (neighborIndexX >= 0 && neighborIndexX < lowDimX && neighborIndexY >= 0 && neighborIndexY < lowDimY)
        ? offsetArray[directionIndexOffset + neighborIndexY * lowDimX + neighborIndexX]
        : 0;
}

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
                                  const int directionIndexOffset, const int dimY, const int dimX, const int lowDimY,
                                  const int lowDimX, const int windowSize, const int searchWindowSize,
                                  const int resolutionScalar, const int isFirstIteration) {
    // Shared memory for the partial sums of the current block
    __local unsigned int partialSums[64];

    // Current entry to be computed by the thread
    int cx = get_global_id(0);
    int cy = get_global_id(1);
    cx = min(cx, lowDimX - 1);
    cy = min(cy, lowDimY - 1);
    const int cz = get_global_id(2);
    const int tIdx = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int scaledCx = cx << resolutionScalar;         // The X-Index of the current thread in the input frames
    const int scaledCy = cy << resolutionScalar;         // The Y-Index of the current thread in the input frames
    const int threadIndex2D = cy * lowDimX + cx;         // Standard thread index without Z-Dim
    unsigned int delta = 0;                              // The delta value of the current pixel
    unsigned int offsetBias = 0;
    unsigned int neighborBias1 = 0;
    unsigned int neighborBias2 = 0;
    short neighborOffsetX = 0;
    short neighborOffsetY = 0;
    unsigned short diffToNeighbor = 0;

    // Retrieve the offset values for the current thread that are going to be tested
    const short idealOffsetX = offsetArray[threadIndex2D];
    const short idealOffsetY = offsetArray[directionIndexOffset + threadIndex2D];
    const short relOffsetAdjustmentX = (cz % searchWindowSize) - (searchWindowSize / 2);
    const short relOffsetAdjustmentY = (cz / searchWindowSize) - (searchWindowSize / 2);
    const short offsetX = idealOffsetX + (relOffsetAdjustmentX * relOffsetAdjustmentX * (relOffsetAdjustmentX > 0 ? 1 : -1));
    const short offsetY = idealOffsetY + (relOffsetAdjustmentY * relOffsetAdjustmentY * (relOffsetAdjustmentY > 0 ? 1 : -1));
    const int newCx = scaledCx + offsetX;
    const int newCy = scaledCy + offsetY;

    // Calculate the delta value for the current pixel
    if (scaledCy < 0 || scaledCx < 0 || scaledCy >= dimY || scaledCx >= dimX) {
        delta = 32768;
    } else if (newCy < 0 || newCx < 0 || newCy >= dimY || newCx >= dimX) {
        delta = (abs_diff(frame1[scaledCy * dimX + scaledCx], frame2[min(max(newCy, 0), dimY - 1) * dimX + min(max(newCx, 0), dimX - 1)]) + 
                abs_diff(frame1[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1)], frame2[dimY * dimX + (min(max(newCy, 0), dimY - 1) >> 1) * dimX + (min(max(newCx, 0), dimX - 1) & ~1)]) + 
                abs_diff(frame1[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1) + 1], frame2[dimY * dimX + (min(max(newCy, 0), dimY - 1) >> 1) * dimX + (min(max(newCx, 0), dimX - 1) & ~1) + 1])) >> 2;
    } else {
        delta = abs_diff(frame1[scaledCy * dimX + scaledCx], frame2[newCy * dimX + newCx]) + 
                abs_diff(frame1[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1)], frame2[dimY * dimX + (newCy >> 1) * dimX + (newCx & ~1)]) + 
                abs_diff(frame1[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1) + 1], frame2[dimY * dimX + (newCy >> 1) * dimX + (newCx & ~1) + 1]);
    }

    // Calculate the offset bias
    offsetBias = abs(offsetX) + abs(offsetY);

    // Calculate the neighbor biases
    if (!isFirstIteration) {
        // Relative positions of neighbors
        const int neighborOffsets[8][2] = {
            {0, windowSize},   // Down
            {windowSize, 0},   // Right
            {-windowSize, 0},  // Left
            {0, -windowSize},  // Up
            {-windowSize, -windowSize}, // Top Left
            {windowSize, -windowSize},  // Top Right
            {-windowSize, windowSize},  // Bottom Left
            {windowSize, windowSize},   // Bottom Right
        };

        // Iterate over neighbors
        for (int i = 0; i < 8; ++i) {
            int neighborIndexX = cx + neighborOffsets[i][0];
            int neighborIndexY = cy + neighborOffsets[i][1];

            // Get the offset values of the current neighbor
            neighborOffsetX = getNeighborOffset(offsetArray, neighborIndexX, neighborIndexY, lowDimX, lowDimY, 0);
            neighborOffsetY = getNeighborOffset(offsetArray, neighborIndexX, neighborIndexY, lowDimX, lowDimY, directionIndexOffset);

            // Calculate the difference between the proposed offset and the neighbor's offset
            diffToNeighbor = abs_diff(neighborOffsetX, offsetX) + abs_diff(neighborOffsetY, offsetY);

            // Sum differences into appropriate groups
            if (i < 4) {
                neighborBias1 += diffToNeighbor; // Neighbors
            } else if (i < 8) {
                neighborBias2 += diffToNeighbor; // Diagonal neighbors
            }
        }

        // Scale biases
        neighborBias1 <<= 5;
        neighborBias2 <<= 5;
    }
    
    if (windowSize == 1) {
        // Window size of 1x1
        summedUpDeltaArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] = delta + offsetBias + neighborBias1 + neighborBias2;
        return;
    } else {
        // All other window sizes
        partialSums[tIdx] = delta + offsetBias + neighborBias1 + neighborBias2;
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