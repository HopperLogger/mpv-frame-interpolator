// Helper function to get neighbor offset values
inline short getNeighborOffset(__global const short* offsetArray, int neighborIndexX, int neighborIndexY, 
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
__kernel void calcDeltaSumsKernel(__global unsigned int* summedUpDeltaArray, __global const unsigned char* frame1,
                                  __global const unsigned char* frame2, __global const short* offsetArray,
                                  const int directionIndexOffset, const int dimY, const int dimX, const int lowDimY,
                                  const int lowDimX, const int windowSize, const int searchWindowSize,
                                  const int resolutionScalar, const int isFirstIteration, const int step) {
    // Shared memory for the partial sums of the current block
    __local unsigned int partialSums[64];

    // Current entry to be computed by the thread
    int cx = get_global_id(0);
    int cy = get_global_id(1);
    cx = min(cx, lowDimX - 1);
    cy = min(cy, lowDimY - 1);
    const int cz = get_global_id(2);
    const int tIdx = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int scaledCx = cx << resolutionScalar;               // The X-Index of the current thread in the input frames
    int scaledCy = cy << resolutionScalar;               // The Y-Index of the current thread in the input frames
    const int threadIndex2D = cy * lowDimX + cx;         // Standard thread index without Z-Dim
    unsigned int delta = 0;                              // The delta value of the current pixel
    unsigned int offsetBias = 0;                         // The bias of the current offset
    unsigned int neighborBias1 = 0;                      // The bias of the neighbors (up, down, left, right)
    unsigned int neighborBias2 = 0;                      // The bias of the diagonal neighbors
    short neighborOffsetX = 0;                           // The X-Offset of the current neighbor
    short neighborOffsetY = 0;                           // The Y-Offset of the current neighbor
    unsigned short diffToNeighbor = 0;                   // The difference of the current offset to the neighbor's offset

    // Retrieve the offset values for the current thread that are going to be tested
    const short idealOffsetX = offsetArray[threadIndex2D];
    const short idealOffsetY = offsetArray[directionIndexOffset + threadIndex2D];
    short relOffsetAdjustmentX = 0;
    short relOffsetAdjustmentY = 0;
    if (!(step & 1)) {
        relOffsetAdjustmentX = (cz % searchWindowSize) - (searchWindowSize / 2);
        relOffsetAdjustmentX = (relOffsetAdjustmentX * relOffsetAdjustmentX * (relOffsetAdjustmentX > 0 ? 1 : -1));
    } else {
        relOffsetAdjustmentY = (cz % searchWindowSize) - (searchWindowSize / 2);
        relOffsetAdjustmentY = (relOffsetAdjustmentY * relOffsetAdjustmentY * (relOffsetAdjustmentY > 0 ? 1 : -1));
    }
    const short offsetX = idealOffsetX + relOffsetAdjustmentX;
    const short offsetY = idealOffsetY + relOffsetAdjustmentY;
    int newCx = scaledCx + offsetX;
    int newCy = scaledCy + offsetY;

    // Check if we are out of bounds
    if (scaledCx < 0 || scaledCx >= dimX || scaledCy < 0 || scaledCy >= dimY) {
        delta = 0;
    } else {
        // Mirror the projected pixel if it is out of bounds
        if (newCx >= dimX) {
            newCx = dimX - (newCx - dimX + 1);
        } else if (newCx < 0) {
            newCx = -newCx - 1;
        }
        if (newCy >= dimY) {
            newCy = dimY - (newCy - dimY + 1);
        } else if (newCy < 0) {
            newCy = -newCy - 1;
        }

        // Calculate the delta value for the current pixel
        delta = abs_diff(frame1[newCy * dimX + newCx], frame2[scaledCy * dimX + scaledCx]) + 
                abs_diff(frame1[dimY * dimX + (newCy >> 1) * dimX + (newCx & ~1)], frame2[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1)]) + 
                abs_diff(frame1[dimY * dimX + (newCy >> 1) * dimX + (newCx & ~1) + 1], frame2[dimY * dimX + (scaledCy >> 1) * dimX + (scaledCx & ~1) + 1]);
    }

    // Calculate the offset bias
    if (!step) {
        offsetBias = abs(relOffsetAdjustmentX) == 0 ? 0 : max(abs(relOffsetAdjustmentX) >> 6, 1);
    } else {
        offsetBias = abs(relOffsetAdjustmentY) == 0 ? 0 : max(abs(relOffsetAdjustmentY) >> 6, 1);
    }

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
            if (!step) {
                neighborOffsetX = getNeighborOffset(offsetArray, neighborIndexX, neighborIndexY, lowDimX, lowDimY, 0);
            } else {
                neighborOffsetY = getNeighborOffset(offsetArray, neighborIndexX, neighborIndexY, lowDimX, lowDimY, directionIndexOffset);
            }

            // Calculate the difference between the proposed offset and the neighbor's offset
            if (!step) {
                diffToNeighbor = abs_diff(neighborOffsetX, offsetX);
            } else {
                diffToNeighbor = abs_diff(neighborOffsetY, offsetY);
            }

            // Sum differences into appropriate groups
            if (i < 4) {
                neighborBias1 += diffToNeighbor; // Neighbors
            } else if (i < 8) {
                neighborBias2 += diffToNeighbor; // Diagonal neighbors
            }
        }

        // Scale biases
        neighborBias1 >>= 3;
        neighborBias2 >>= 3;
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
        atomic_add(&summedUpDeltaArray[cz * lowDimY * lowDimX + (windowIndexY * windowSize) * lowDimX + (windowIndexX * windowSize)], partialSums[tIdx]);
    }
}