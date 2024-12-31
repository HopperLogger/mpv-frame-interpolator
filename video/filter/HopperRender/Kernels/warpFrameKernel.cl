unsigned char apply_levelsY(float value, float black_level, float white_level) {
    return fmax(fmin((value - black_level) / (white_level - black_level) * 255.0f, 255.0f), 0.0f);
}

unsigned char apply_levelsUV(float value, float white_level) {
    return fmax(fmin((value - 128.0f) / white_level * 255.0f + 128.0f, 255.0f), 0.0f);
}

// Helper function to mirror the coordinate if it is outside the bounds
int mirrorCoordinate(int pos, int dim) {
    if (pos >= dim - 1) {
        return (dim - 1) - (pos - dim + 2);
    } else if (pos < 1) {
        return -pos + 1;
    }
    return pos;
}

// Kernel that warps a frame according to the offset array
__kernel void warpFrameKernel(__global const unsigned char* sourceFrame12, __global const unsigned char* sourceFrame21,
                              __global const short* offsetArray12, __global const short* offsetArray21,
                              __global unsigned char* outputFrame, const float frameScalar12, const float frameScalar21,
                              const int lowDimX, const int dimY, const int dimX,
                              const int resolutionScalar, const int directionIndexOffset, const int channelIndexOffset,
                              const int frameOutputMode, const float black_level, const float white_level) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);
    const int verticalOffset = dimY >> 2;
    int adjCx = cx;
    int adjCy = cy;

    if (cz == 0 && cy < dimY && cx < dimX) {
        if (frameOutputMode == 5) { // SideBySide1
            const bool isInLeftSideY = cy >= verticalOffset && cy < (verticalOffset + (dimY >> 1)) && cx < (dimX >> 1);
        
            if (isInLeftSideY) { // Fill the left side of the output frame with the source frame
                outputFrame[cy * dimX + cx] = sourceFrame12[cy * dimX + cx];
                return;
            }
        } else if (frameOutputMode == 6) { // SideBySide2
            const bool isInLeftSideY = cy >= verticalOffset && cy < (verticalOffset + (dimY >> 1)) && cx < (dimX >> 1);
            const bool isInRightSideY = cy >= verticalOffset && cy < (verticalOffset + (dimY >> 1)) && cx >= (dimX >> 1) && cx < dimX;
        
            if (isInLeftSideY) { // Place the source frame in the left side of the output frame
                outputFrame[cy * dimX + cx] = sourceFrame12[((cy - verticalOffset) << 1) * dimX + (cx << 1)];
                return;
            } else if (isInRightSideY) { // Place the warped frame in the right side of the output frame
                adjCx = (cx - (dimX >> 1)) << 1;
                adjCy = (cy - verticalOffset) << 1;
            } else { // Fill the surrounding area with black
                outputFrame[cy * dimX + cx] = 0;
                return;
            }
        }

        // Get the current flow values
        const int scaledCx = adjCx >> resolutionScalar;  // The X-Index of the current thread in the offset array
        const int scaledCy = adjCy >> resolutionScalar;  // The Y-Index of the current thread in the offset array
        const int offsetX12 = (int)round((float)(offsetArray12[scaledCy * lowDimX + scaledCx]) * frameScalar12);
        const int offsetY12 = (int)round((float)(offsetArray12[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar12);
        const int offsetX21 = (int)round((float)(offsetArray21[scaledCy * lowDimX + scaledCx]) * frameScalar21);
        const int offsetY21 = (int)round((float)(offsetArray21[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar21);

        // Get the new pixel position
        const int newCx12 = mirrorCoordinate(adjCx + offsetX12, dimX);
        const int newCy12 = mirrorCoordinate(adjCy + offsetY12, dimY);
        const int newCx21 = mirrorCoordinate(adjCx + offsetX21, dimX);
        const int newCy21 = mirrorCoordinate(adjCy + offsetY21, dimY);

        // Move the origin pixel to the new position
        if (frameOutputMode == 0) { // WarpedFrame12
            outputFrame[cy * dimX + cx] = sourceFrame12[newCy12 * dimX + newCx12];
        } else if (frameOutputMode == 1) { // WarpedFrame21
            outputFrame[cy * dimX + cx] = sourceFrame21[newCy21 * dimX + newCx21];
        } else { // BlendedFrame
            outputFrame[cy * dimX + cx] = apply_levelsY(
                (float)sourceFrame12[newCy12 * dimX + newCx12] * frameScalar21 + (float)sourceFrame21[newCy21 * dimX + newCx21] * frameScalar12,
                black_level, white_level);
        }
        
    } else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
        if (frameOutputMode == 5) { // SideBySide1
            const bool isInLeftSideUV = cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx < (dimX >> 1);
        
            if (isInLeftSideUV) { // Fill the left side of the output frame with the source frame
                outputFrame[channelIndexOffset + cy * dimX + cx] = sourceFrame12[channelIndexOffset + cy * dimX + cx];
                return;
            }
        } else if (frameOutputMode == 6) { // SideBySide2
            const bool isInLeftSideUV = cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx < (dimX >> 1);
            const bool isInRightSideUV = cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx >= (dimX >> 1) && cx < dimX;
        
            if (isInLeftSideUV) { // Place the source frame in the left side of the output frame
                outputFrame[channelIndexOffset + cy * dimX + cx] = sourceFrame12[channelIndexOffset + ((cy - (verticalOffset >> 1)) << 1) * dimX + (cx << 1) + (cx & 1)];
                return;
            } else if (isInRightSideUV) { // Place the warped frame in the right side of the output frame
                adjCx = (cx - (dimX >> 1)) << 1;
                adjCy = (cy - (verticalOffset >> 1)) << 1;
            } else { // Fill the surrounding area with black
                outputFrame[channelIndexOffset + cy * dimX + cx] = 128;
                return;
            }
        }

		// Get the current flow values
        const int scaledCx = (adjCx >> resolutionScalar) & ~1;  // The X-Index of the current thread in the offset array
        const int scaledCy = (adjCy >> resolutionScalar) << 1;  // The Y-Index of the current thread in the offset array
        const int offsetX12 = (int)round((float)(offsetArray12[scaledCy * lowDimX + scaledCx]) * frameScalar12);
        const int offsetY12 = (int)round((float)(offsetArray12[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar12 * 0.5f);
        const int offsetX21 = (int)round((float)(offsetArray21[scaledCy * lowDimX + scaledCx]) * frameScalar21);
        const int offsetY21 = (int)round((float)(offsetArray21[directionIndexOffset + scaledCy * lowDimX + scaledCx]) * frameScalar21 * 0.5f);

        // Get the new pixel position
        const int newCx12 = mirrorCoordinate(adjCx + offsetX12, dimX);
        const int newCy12 = mirrorCoordinate(adjCy + offsetY12, (dimY >> 1));
        const int newCx21 = mirrorCoordinate(adjCx + offsetX21, dimX);
        const int newCy21 = mirrorCoordinate(adjCy + offsetY21, (dimY >> 1));

        // Move the origin pixel to the new position
        if (frameOutputMode == 0) { // WarpedFrame12
            outputFrame[channelIndexOffset + cy * dimX + cx] = sourceFrame12[channelIndexOffset + newCy12 * dimX + (newCx12 & ~1) + (cx & 1)];
        } else if (frameOutputMode == 1) { // WarpedFrame21
            outputFrame[channelIndexOffset + cy * dimX + cx] = sourceFrame21[channelIndexOffset + newCy21 * dimX + (newCx21 & ~1) + (cx & 1)];
        } else { // BlendedFrame
            outputFrame[channelIndexOffset + cy * dimX + cx] = apply_levelsUV(
                (float)sourceFrame12[channelIndexOffset + newCy12 * dimX + (newCx12 & ~1) + (cx & 1)] * frameScalar21 + (float)sourceFrame21[channelIndexOffset + newCy21 * dimX + (newCx21 & ~1) + (cx & 1)] * frameScalar12,
                white_level);
        }
    }
}