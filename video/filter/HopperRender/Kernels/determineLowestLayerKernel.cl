// Kernel that determines the offset layer with the lowest delta
__kernel void determineLowestLayerKernel(__global unsigned int* summedUpDeltaArray,
                                         __global unsigned char* lowestLayerArray, const int windowSize,
                                         const int searchWindowSize, const int lowDimY, const int lowDimX) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int threadIndex2D = cy * lowDimX + cx;  // Standard thread index without Z-Dim
    bool isWindowRepresent = cy % windowSize == 0 && cx % windowSize == 0;

    // Find the layer with the lowest value
    if (isWindowRepresent) {
        unsigned char lowestLayer = 0;

        for (int z = 1; z < searchWindowSize; ++z) {
            if (summedUpDeltaArray[z * lowDimY * lowDimX + threadIndex2D] <
                summedUpDeltaArray[lowestLayer * lowDimY * lowDimX + threadIndex2D]) {
                lowestLayer = z;
            }
        }

        lowestLayerArray[threadIndex2D] = lowestLayer;
    }
}