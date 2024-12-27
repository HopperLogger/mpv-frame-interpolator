// Kernel that places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
__kernel void sideBySide2Kernel(__global const unsigned short* sourceFrame,
                                __global const unsigned short* warpedFrame12,
                                __global const unsigned short* warpedFrame21, __global unsigned short* outputFrame,
                                const float frame1Scalar, const float frame2Scalar, const int dimY, const int dimX,
                                const int halfDimY, const int halfDimX, const int channelIndexOffset,
                                const float outputBlackLevel, const float outputWhiteLevel) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);
    const int verticalOffset = dimY >> 2;
    const bool isYChannel = cz == 0 && cy < dimY && cx < dimX;
    const bool isUVChannel = cz == 1 && cy < halfDimY && cx < dimX;
    const bool isVChannel = (cx & 1) == 1;
    const bool isInLeftSideY = cy >= verticalOffset && cy < (verticalOffset + halfDimY) && cx < halfDimX;
    const bool isInRightSideY = cy >= verticalOffset && cy < (verticalOffset + halfDimY) && cx >= halfDimX && cx < dimX;
    const bool isInLeftSideUV =
        cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx < halfDimX;
    const bool isInRightSideUV =
        cy >= (verticalOffset >> 1) && cy < ((verticalOffset >> 1) + (dimY >> 2)) && cx >= halfDimX && cx < dimX;
    float blendedFrameValue;

    // Early exit if thread indices are out of bounds
    if (cz > 1 || cy >= dimY || cx >= dimX || (cz == 1 && cy >= halfDimY)) return;

    // --- Blending ---
    if (isYChannel && isInRightSideY) {
		// Y Channel
        blendedFrameValue = ((float)(warpedFrame12[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) *
                                 frame1Scalar +
                             (float)(warpedFrame21[((cy - verticalOffset) << 1) * dimX + ((cx - halfDimX) << 1)]) *
                                 frame2Scalar);
    } else if (isUVChannel && isInRightSideUV) {
		// U/V Channels
        blendedFrameValue = ((float)(warpedFrame12[channelIndexOffset + 2 * (cy - (verticalOffset >> 1)) * dimX +
                                                   ((cx - halfDimX) << 1) + isVChannel]) *
                                 frame1Scalar +
                             (float)(warpedFrame21[channelIndexOffset + 2 * (cy - (verticalOffset >> 1)) * dimX +
                                                   ((cx - halfDimX) << 1) + isVChannel]) *
                                 frame2Scalar);
    }

    // --- Insertion ---
    if (isYChannel) {
        if (isInLeftSideY) {
			// Y Channel Left Side
            outputFrame[cy * dimX + cx] =
                fmax(fmin(((float)(sourceFrame[((cy - verticalOffset) << 1) * dimX + (cx << 1)]) - outputBlackLevel) /
                              (outputWhiteLevel - outputBlackLevel) * 65535.0f,
                          65535.0f),
                     0.0f);
        } else if (isInRightSideY) {
			// Y Channel Right Side
            outputFrame[cy * dimX + cx] = fmax(
                fmin(((float)blendedFrameValue - outputBlackLevel) / (outputWhiteLevel - outputBlackLevel) * 65535.0f,
                     65535.0f),
                0.0f);
        } else {
			// Y Channel Black Frame
            outputFrame[cy * dimX + cx] = 0;
        }
    } else if (isUVChannel) {
        if (isInLeftSideUV) {
			// UV Channels Left Side
            outputFrame[dimY * dimX + cy * dimX + cx] = (unsigned short)fmax(
            fmin(((float)sourceFrame[channelIndexOffset + ((cy - (verticalOffset >> 1)) << 1) * dimX + (cx << 1) + isVChannel] - 32768.0f) / outputWhiteLevel * 65535.0f + 32768.0f, 65535.0f), 0.0f);
        } else if (isInRightSideUV) {
			// UV Channels Right Side
            outputFrame[dimY * dimX + cy * dimX + cx] = (unsigned short)fmax(
            fmin((blendedFrameValue - 32768.0f) / outputWhiteLevel * 65535.0f + 32768.0f, 65535.0f), 0.0f);
        } else {
			// UV Channels Black Frame
            outputFrame[dimY * dimX + cy * dimX + cx] = 32768;
        }
    }
}