// Kernel that cleans a flow array
__kernel void cleanFlowKernel(__global const char* flowArray,
							  __global char* blurredFlowArray,
							  const int lowDimY,
							  const int lowDimX) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);

	char offsetX = flowArray[cy * lowDimX + cx];
	char offsetY = flowArray[lowDimY * lowDimX + cy * lowDimX + cx];

    if (abs(offsetY) <= 2 && abs(offsetX <= 2)) {
		for (int y = -6; y < 6; y++) {
			for (int x = -6; x < 6; x++) {
				if ((cy + y) < lowDimY && (cy + y) >= 0 && (cx + x) < lowDimX && (cx + x) >= 0) {
					blurredFlowArray[cz * lowDimY * lowDimX + (cy + y) * lowDimX + (cx + x)] = flowArray[cz * lowDimY * lowDimX + cy * lowDimX + cx];
				}
			}
		}
		//blurredFlowArray[cz * lowDimY * lowDimX + cy * lowDimX + cx] = 0;
	}
}