// Kernel that warps a frame according to the offset array
__kernel void warpFrameKernel(__global const unsigned short* sourceFrame, __global const short* offsetArray,
                              __global unsigned short* warpedFrame, const float frameScalar, const int lowDimX,
                              const int dimY, const int dimX, const int resolutionScalar,
                              const int directionIndexOffset, const int channelIndexOffset) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);

    if (cz == 0 && cy < dimY && cx < dimX) {
        // Y Channel
        const int scaledCx = cx >> resolutionScalar;  // The X-Index of the current thread in the offset array
        const int scaledCy = cy >> resolutionScalar;  // The Y-Index of the current thread in the offset array
        const int offsetX = (int)round((float)(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
        const int offsetY =
            (int)round((float)(offsetArray[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar);

        // Move a 5x5 neighborhood of the current pixel to mask the holes
        for (int nx = cx - 2; nx <= cx + 2; nx++) {
            for (int ny = cy - 2; ny <= cy + 2; ny++) {
                const int newCx = nx + offsetX;
                const int newCy = ny + offsetY;

                if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < dimX && ny >= 0 && ny < dimY && nx >= 0 && nx < dimX) {
                    warpedFrame[newCy * dimX + newCx] = sourceFrame[ny * dimX + nx];
                }
            }
        }
        
    } else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		// U/V-Channel
        const int scaledCx = (cx >> resolutionScalar) & ~1;  // The X-Index of the current thread in the offset array
        const int scaledCy = (cy >> resolutionScalar) << 1;  // The Y-Index of the current thread in the offset array
        const int offsetX = (int)round((float)(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
        const int offsetY =
            (int)round((float)(offsetArray[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar * 0.5);

        // Move a 5x3 neighborhood of the current pixel to mask the holes
        for (int nx = cx - 2; nx <= cx + 2; nx+=2) {
            for (int ny = cy - 1; ny <= cy + 1; ny++) {
                const int newCx = nx + offsetX;
                const int newCy = ny + offsetY;

                if (newCy >= 0 && newCy < (dimY >> 1) && newCx >= 0 && newCx < dimX && ny >= 0 && ny < (dimY >> 1) && nx >= 0 && nx < dimX) {
                    if (!(cx & 1)) {
                        // U-Channel
                        warpedFrame[channelIndexOffset + newCy * dimX + (newCx & ~1)] =
                            sourceFrame[channelIndexOffset + ny * dimX + (nx & ~1)];
                    } else {
                        // V-Channel
                        warpedFrame[channelIndexOffset + newCy * dimX + (newCx & ~1) + 1] =
                            sourceFrame[channelIndexOffset + ny * dimX + (nx & ~1) + 1];
                    }
                }
            }
        }
    }
}