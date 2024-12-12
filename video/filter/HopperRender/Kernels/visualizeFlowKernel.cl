// Kernel that creates an HSV flow image from the offset array
__kernel void visualizeFlowKernel(__global const char* offsetArray,
									 __global unsigned short* outputFrame,
									 __global const unsigned short* inputFrame,
									 const int lowDimY,
									 const int lowDimX,
									 const int dimY,
									 const int dimX,
									 const int resolutionScalar,
									 const int directionIndexOffset,
									 const int channelIndexOffset,
									 const int doBWOutput) {
	// Current entry to be computed by the thread
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int cz = get_global_id(2);
	const float blendScalar = 0.5f;

	const int scaledCx = cx >> resolutionScalar; // The X-Index of the current thread in the offset array
	const int scaledCy = cy >> resolutionScalar; // The Y-Index of the current thread in the offset array

	// Get the current flow values
	char x;
	char y;
	if (cz == 0 && cy < dimY && cx < dimX) {
		x = offsetArray[scaledCy * lowDimX + scaledCx];
		y = offsetArray[directionIndexOffset + scaledCy * lowDimX + scaledCx];
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX){
		x = offsetArray[(scaledCy << 1) * lowDimX + scaledCx];
		y = offsetArray[directionIndexOffset + (scaledCy << 1) * lowDimX + scaledCx];
	}

	// Used for color output
	struct RGB {
		unsigned char r, g, b;
	};
	struct RGB rgb;

	// Used for black and white output
	const unsigned short normFlow = min((abs(x) + abs(y)) << 10, 65535);

	// Color Output
	if (!doBWOutput) {
		// Calculate the angle in radians
		const float angle_rad = atan2((float)y, (float)x);

		// Convert radians to degrees
		float angle_deg = angle_rad * (180.0f / 3.14159265359f);

		// Ensure the angle is positive
		if (angle_deg < 0) {
			angle_deg += 360.0f;
		}

		// Normalize the angle to the range [0, 360]
		angle_deg = fmod(angle_deg, 360.0f);
		if (angle_deg < 0) {
			angle_deg += 360.0f;
		}

		// Map the angle to the hue value in the HSV model
		const float hue = angle_deg / 360.0f;

		// Convert HSV to RGB
		const int h_i = (int)(hue * 6.0f);
		const float f = hue * 6.0f - h_i;
		const float q = 1.0f - f;

		switch (h_i % 6) {
			case 0:
				rgb.r = 255;
				rgb.g = (unsigned char)(f * 255.0f);
				rgb.b = 0;
				break; // Red - Yellow
			case 1:
				rgb.r = (unsigned char)(q * 255.0f);
				rgb.g = 255;
				rgb.b = 0;
				break; // Yellow - Green
			case 2:
				rgb.r = 0;
				rgb.g = 255;
				rgb.b = (unsigned char)(f * 255.0f);
				break; // Green - Cyan
			case 3:
				rgb.r = 0;
				rgb.g = (unsigned char)(q * 255.0f);
				rgb.b = 255;
				break; // Cyan - Blue
			case 4:
				rgb.r = (unsigned char)(f * 255.0f);
				rgb.g = 0;
				rgb.b = 255;
				break; // Blue - Magenta
			case 5:
				rgb.r = 255;
				rgb.g = 0;
				rgb.b = (unsigned char)(q * 255.0f);
				break; // Magenta - Red
			default:
				rgb.r = 0;
				rgb.g = 0;
				rgb.b = 0;
				break;
		}

		// Adjust the color intensity based on the flow magnitude
		rgb.r = fmax(fmin((float)rgb.r / 255.0f * (abs(x) + abs(y)) * 4.0f, 255.0f), 0.0f);
		rgb.g = fmax(fmin((float)rgb.g / 255.0f * abs(y) * 8.0f, 255.0f), 0.0f);
		rgb.b = fmax(fmin((float)rgb.b / 255.0f * (abs(x) + abs(y)) * 4.0f, 255.0f), 0.0f);

		// Prevent random colors when there is no flow
		if (abs(x) < 1.0f && abs(y) < 1.0f) {
			rgb.r = 0;
			rgb.g = 0;
			rgb.b = 0;
		}
	}

	// Y Channel
	if (cz == 0 && cy < dimY && cx < dimX) {
		outputFrame[cy * dimX + cx] = doBWOutput ? normFlow : ((unsigned short)(
				(fmax(fmin(rgb.r * 0.299f + rgb.g * 0.587f + rgb.b * 0.114f, 255.0f), 0.0f)) * blendScalar) << 8) + 
				inputFrame[cy * dimX + cx] * (1.0f - blendScalar);
	// U/V Channels
	} else if (cz == 1 && cy < (dimY >> 1) && cx < dimX) {
		// U Channel
		if ((cx & 1) == 0) {
			outputFrame[channelIndexOffset + cy * dimX + (cx & ~1)] = doBWOutput ? 32768 : (unsigned short)(
						fmax(fmin(rgb.r * -0.168736f + rgb.g * -0.331264f + rgb.b * 0.5f + 128.0f, 255.0f), 0.0f)) << 8;
		// V Channel
		} else {
			outputFrame[channelIndexOffset + cy * dimX + (cx & ~1) + 1] = doBWOutput ? 32768 : (unsigned short)(
						fmax(fmin(rgb.r * 0.5f + rgb.g * -0.418688f + rgb.b * -0.081312f + 128.0f, 255.0f), 0.0f)) << 8;
		}
	}
}