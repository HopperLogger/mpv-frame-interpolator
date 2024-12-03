// Kernel that runs a tearing test on the GPU
__kernel void tearingTestKernel(__global unsigned short* outputFrame,
								const int dimY,
								const int dimX,
								const int width,
								const int pos_x) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);

	if (cy < dimY && cx < dimX && cx >= pos_x && cx < pos_x + width) {
		outputFrame[cy * dimX + cx] = 65535;
	}
}