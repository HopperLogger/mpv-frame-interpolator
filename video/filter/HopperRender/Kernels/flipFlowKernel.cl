// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__kernel void flipFlowKernel(__global const short* offsetArray12, __global short* offsetArray21, const int lowDimY,
                             const int lowDimX, const int resolutionScalar, const int directionIndexOffset) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);

    // Check if we are inside the flow array
    if (cy < lowDimY && cx < lowDimX) {
        // Get the current flow values
        const short x = offsetArray12[cy * lowDimX + cx];
        const short y = offsetArray12[directionIndexOffset + cy * lowDimX + cx];
        int newCx;
        int newCy;
        if (resolutionScalar) {
            newCx = cx + (int)round((float)x / pow(2.0f, resolutionScalar));
            newCy = cy + (int)round((float)y / pow(2.0f, resolutionScalar));
        } else {
            newCx = cx + x;
            newCy = cy + y;
        }
        // const int newCx = cx;
        // const int newCy = cy;

        // Project the flow values onto the flow array from frame 2 to frame 1
        if (cz == 0 && newCy < lowDimY && newCy >= 0 && newCx < lowDimX && newCx >= 0) {
			// X-Layer
            offsetArray21[newCy * lowDimX + newCx] = -x;
        } else if (cz == 1 && newCy < lowDimY && newCy >= 0 && newCx < lowDimX && newCx >= 0) {
			// Y-Layer
            offsetArray21[directionIndexOffset + newCy * lowDimX + newCx] = -y;
        }
    }
}