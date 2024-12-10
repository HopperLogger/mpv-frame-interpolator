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
		//const char scaledX = round((float)x / pow(2.0f, resolutionScalar));
		//const char scaledY = round((float)y / pow(2.0f, resolutionScalar));
		//const int newCx = cx + scaledX;
		//const int newCy = cy + scaledY;
		const int newCx = cx;
		const int newCy = cy;

		// Project the flow values onto the flow array from frame 2 to frame 1
		// X-Layer
		if (cz == 0 && newCy < lowDimY && newCy >= 0 && newCx < lowDimX && newCx >= 0) {
			flowArray21[newCy * lowDimX + newCx] = -x;
		// Y-Layer
		} else if (cz == 1 && newCy < lowDimY && newCy >= 0 && newCx < lowDimX && newCx >= 0) {
			flowArray21[directionIdxOffset + newCy * lowDimX + newCx] = -y;
		}
	}
}