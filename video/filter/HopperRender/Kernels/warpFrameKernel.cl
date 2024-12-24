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
        const int newCx = cx + offsetX;
        const int newCy = cy + offsetY;

        // Check if the current pixel is inside the frame
        if (newCy >= 0 && newCy < dimY && newCx >= 0 && newCx < dimX) {
            warpedFrame[newCy * dimX + newCx] = sourceFrame[min(max(cy, 1), dimY - 2) * dimX + min(max(cx, 1), dimX - 2)];
        }
        
    } else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		// U/V-Channel
        const int scaledCx = (cx >> resolutionScalar) & ~1;  // The X-Index of the current thread in the offset array
        const int scaledCy = (cy >> resolutionScalar) << 1;  // The Y-Index of the current thread in the offset array
        const int offsetX = (int)round((float)(offsetArray[scaledCy * lowDimX + scaledCx]) * frameScalar);
        const int offsetY =
            (int)round((float)(offsetArray[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar * 0.5);

        const int newCx = cx + offsetX;
        const int newCy = cy + offsetY;

        // Check if the current pixel is inside the frame
        if (newCy >= 0 && newCy < (dimY >> 1) && newCx >= 0 && newCx < dimX) {
            if ((cx & 1) == 0) {
				// U-Channel
                warpedFrame[channelIndexOffset + newCy * dimX + (newCx & ~1)] =
                    sourceFrame[channelIndexOffset + min(max(cy, 1), dimY - 2) * dimX + (min(max(cx, 1), dimX - 2) & ~1)];

            } else {
				// V-Channel
                warpedFrame[channelIndexOffset + newCy * dimX + (newCx & ~1) + 1] =
                    sourceFrame[channelIndexOffset + min(max(cy, 1), dimY - 2) * dimX + (min(max(cx, 1), dimX - 2) & ~1) + 1];
            }
        }
    }
}