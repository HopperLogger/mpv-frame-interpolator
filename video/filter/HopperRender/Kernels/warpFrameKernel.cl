// Kernel that warps a frame according to the offset array
__kernel void warpFrameKernel(__global const unsigned char* sourceFrame, __global const short* offsetArray,
                              __global unsigned char* warpedFrame, const float frameScalar, const int lowDimX,
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
        const int offsetY = (int)round((float)(offsetArray[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar);

        // Move a block of pixels to mask the holes
        const int newCx = max(min(cx + offsetX, dimX - 1), 0);
        const int newCy = max(min(cy + offsetY, dimY - 1), 0);

        warpedFrame[cy * dimX + cx] = sourceFrame[newCy * dimX + newCx];
        
    } else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		// U/V-Channel
        const int scaledCx = (cx >> resolutionScalar) & ~1;  // The X-Index of the current thread in the offset array
        const int scaledCy = (cy >> resolutionScalar) << 1;  // The Y-Index of the current thread in the offset array
        const int offsetX = (int)round((float)(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
        const int offsetY = (int)round((float)(offsetArray[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar * 0.5f);

        // Move a block of pixels to mask the holes
        const int newCx = max(min(cx + offsetX, dimX - 1), 0);
        const int newCy = max(min(cy + offsetY, (dimY >> 1) - 1), 0);

        warpedFrame[channelIndexOffset + cy * dimX + cx] = sourceFrame[channelIndexOffset + newCy * dimX + (newCx & ~1) + (cx & 1)];
    }
}