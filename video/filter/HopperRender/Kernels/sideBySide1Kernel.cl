// Kernel that places half of frame 1 over the outputFrame
__kernel void sideBySide1Kernel(__global const unsigned char* sourceFrame, __global unsigned char* outputFrame,
                                const int dimY, const int dimX, const int channelIndexOffset,
                                const float outputBlackLevel, const float outputWhiteLevel) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);

    if (cz == 0 && cy < dimY && cx < (dimX >> 1)) { // Y Channel
        outputFrame[cy * dimX + cx] = fmax(fmin(((float)sourceFrame[cy * dimX + cx] - outputBlackLevel) / (outputWhiteLevel - outputBlackLevel) * 255.0f, 255.0f), 0.0f);
    } else if (cz == 1 && cy < (dimY >> 1) && cx < (dimX >> 1)) { // U/V Channels
        outputFrame[channelIndexOffset + cy * dimX + cx] = fmax(fmin(((float)sourceFrame[channelIndexOffset + cy * dimX + cx] - 128.0f) / outputWhiteLevel * 255.0f + 128.0f, 255.0f), 0.0f);
    }
}