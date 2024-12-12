// Kernel that blends warpedFrame1 to warpedFrame2
__kernel void blendFrameKernel(__global const unsigned short* warpedFrameArray12,
                               __global const unsigned short* warpedFrameArray21, __global unsigned short* outputFrame,
                               const float frame1Scalar, const float frame2Scalar, const int dimY, const int dimX,
                               const int channelIndexOffset, const float outputBlackLevel,
                               const float outputWhiteLevel) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);
    float pixelValue;

    if (cz == 0 && cy < dimY && cx < dimX) {
		// Y Channel
        pixelValue = (float)warpedFrameArray12[cy * dimX + cx] * frame1Scalar +
                     (float)warpedFrameArray21[cy * dimX + cx] * frame2Scalar;
        outputFrame[cy * dimX + cx] = (unsigned short)fmax(
            fmin((pixelValue - outputBlackLevel) / (outputWhiteLevel - outputBlackLevel) * 65535.0f, 65535.0f), 0.0f);
    } else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		// U/V Channels
        pixelValue = (float)warpedFrameArray12[channelIndexOffset + cy * dimX + cx] * frame1Scalar +
                     (float)warpedFrameArray21[channelIndexOffset + cy * dimX + cx] * frame2Scalar;
        outputFrame[channelIndexOffset + cy * dimX + cx] = (unsigned short)pixelValue;
    }
}