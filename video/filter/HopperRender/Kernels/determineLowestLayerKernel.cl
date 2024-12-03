// Kernel that determines the offset layer with the lowest delta
__kernel void determineLowestLayerKernel(__global double* normalizedDeltaArray,
										 __global unsigned short* globalLowestLayerArray,
										 const int windowDim,
										 const int numLayers,
										 const int lowDimY,
										 const int lowDimX) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int threadIndex2D = cy * lowDimX + cx; // Standard thread index without Z-Dim
	bool isWindowRepresent = cy % windowDim == 0 && cx % windowDim == 0;

	// Find the layer with the lowest value
	if (isWindowRepresent) {
		unsigned short lowestLayer = 0;

		for (int z = 1; z < numLayers; ++z) {
			if (normalizedDeltaArray[z * lowDimY * lowDimX + threadIndex2D] < 
				normalizedDeltaArray[lowestLayer * lowDimY * lowDimX + threadIndex2D]) {
				lowestLayer = z;
			}
		}

		globalLowestLayerArray[threadIndex2D] = lowestLayer;
	}
}