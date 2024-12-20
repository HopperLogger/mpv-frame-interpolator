// Kernel that adjusts the offset array based on the comparison results
__kernel void adjustOffsetArrayKernel(__global short* offsetArray, __global const unsigned char* lowestLayerArray,
                                      const int windowSize, const int directionIndexOffset,
                                      const int searchWindowSize, const int lowDimY,
                                      const int lowDimX) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);
    const int threadIndex3D = cz * directionIndexOffset + cy * lowDimX + cx;  // Standard thread index without Z-Dim

    if (cy < lowDimY && cx < lowDimX) {
        // We only need the lowestLayer if we are still searching
        const int wx = (cx / windowSize) * windowSize;
        const int wy = (cy / windowSize) * windowSize;
        const unsigned char lowestLayer = lowestLayerArray[wy * lowDimX + wx];

        // Calculate the relative offset adjustment that was determined to be ideal
        if (cz == 0) {
            const short idealRelOffsetX = (lowestLayer % searchWindowSize) - (searchWindowSize / 2);
            offsetArray[threadIndex3D] += idealRelOffsetX << 1;
        } else {
            const short idealRelOffsetY = (lowestLayer / searchWindowSize) - (searchWindowSize / 2);
            offsetArray[threadIndex3D] += idealRelOffsetY << 1;
        }
    }
}