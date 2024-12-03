// Kernel that translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
__kernel void flipFlowKernel(__global const char* flowArray12,
							 __global char* flowArray21,
							 const int lowDimY,
							 const int lowDimX,
							 const int resolutionScalar,
							 const int directionIdxOffset) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);

	// Check if we are inside the flow array
	if (cy < lowDimY && cx < lowDimX) {
		// Get the current flow values
		const char x = flowArray12[cy * lowDimX + cx];
		const char y = flowArray12[directionIdxOffset + cy * lowDimX + cx];
		const int scaledX = x >> resolutionScalar;
		const int scaledY = y >> resolutionScalar;

		// Project the flow values onto the flow array from frame 2 to frame 1
		// X-Layer
		if (cz == 0 && (cy + scaledY) < lowDimY && (cy + scaledY) >= 0 && (cx + scaledX) < lowDimX && (cx + scaledX) >= 0) {
			flowArray21[(cy + scaledY) * lowDimX + cx + scaledX] = -x;
		// Y-Layer
		} else if (cz == 1 && (cy + scaledY) < lowDimY && (cy + scaledY) >= 0 && (cx + scaledX) < lowDimX && (cx + scaledX) >= 0) {
			flowArray21[directionIdxOffset + (cy + scaledY) * lowDimX + cx + scaledX] = -y;
		}
	}
}