// Kernel that simply applies a shader to the frame and copies it to the output frame
__kernel void processFrameKernel(__global const unsigned short* frame,
								 __global unsigned short* outputFrame,
                                 const int dimY,
                                 const int dimX,
								 const float blackLevel,
								 const float whiteLevel,
								 const float maxVal) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);

	// Y Channel
	if (cy < dimY && cx < dimX) {
        outputFrame[cy * dimX + cx] = fmax(fmin(((float)frame[cy * dimX + cx] - blackLevel) / (whiteLevel - blackLevel) * maxVal, maxVal), 0.0f);
	}
}