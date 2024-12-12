// Kernel that runs a tearing test on the GPU
__kernel void tearingTestKernel(__global unsigned short* outputFrame, const int dimY, const int dimX, const int posX) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);

    if (cz == 0 && cy < dimY && cx < dimX && cx >= posX && cx < posX + 10) {
        outputFrame[cy * dimX + cx] = 65535;
    } else if (cz == 0 && cy < dimY && cx < dimX) {
        outputFrame[cy * dimX + cx] = 0;
    } else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
        outputFrame[dimY * dimX + cy * dimX + cx] = 32768;
    }
}