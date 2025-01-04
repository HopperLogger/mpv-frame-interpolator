unsigned char apply_levelsY(float value, float black_level, float white_level) {
    return fmax(fmin((value - black_level) / (white_level - black_level) * 255.0f, 255.0f), 0.0f);
}

unsigned char apply_levelsUV(float value, float white_level) {
    return fmax(fmin((value - 128.0f) / white_level * 255.0f + 128.0f, 255.0f), 0.0f);
}

// Helper function to mirror the coordinate if it is outside the bounds
int mirrorCoordinate(int pos, int dim) {
    if (pos >= dim - 1) {
        return (dim - 1) - (pos - dim + 2);
    } else if (pos < 1) {
        return -pos + 1;
    }
    return pos;
}

// Generates a color for the current flow vector
unsigned char visualizeFlow(const short offsetX, const short offsetY, const unsigned char currPixel,
                            const int channel, const int doBWOutput) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int cz = get_global_id(2);

    // Used for color output
    struct RGB {
        unsigned char r, g, b;
    };
    struct RGB rgb;

    // Used for black and white output
    const unsigned char normFlow = min((abs(offsetX) + abs(offsetY)) << 2, 255);

    // Color Output
    if (!doBWOutput) {
        // Calculate the angle in radians
        const float angle_rad = atan2((float)offsetY, (float)offsetX);

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
                break;  // Red - Yellow
            case 1:
                rgb.r = (unsigned char)(q * 255.0f);
                rgb.g = 255;
                rgb.b = 0;
                break;  // Yellow - Green
            case 2:
                rgb.r = 0;
                rgb.g = 255;
                rgb.b = (unsigned char)(f * 255.0f);
                break;  // Green - Cyan
            case 3:
                rgb.r = 0;
                rgb.g = (unsigned char)(q * 255.0f);
                rgb.b = 255;
                break;  // Cyan - Blue
            case 4:
                rgb.r = (unsigned char)(f * 255.0f);
                rgb.g = 0;
                rgb.b = 255;
                break;  // Blue - Magenta
            case 5:
                rgb.r = 255;
                rgb.g = 0;
                rgb.b = (unsigned char)(q * 255.0f);
                break;  // Magenta - Red
            default:
                rgb.r = 0;
                rgb.g = 0;
                rgb.b = 0;
                break;
        }

        // Adjust the color intensity based on the flow magnitude
        rgb.r = fmax(fmin((float)rgb.r / 255.0f * (abs(offsetX) + abs(offsetY)) * 4.0f, 255.0f), 0.0f);
        rgb.g = fmax(fmin((float)rgb.g / 255.0f * abs(offsetY) * 8.0f, 255.0f), 0.0f);
        rgb.b = fmax(fmin((float)rgb.b / 255.0f * (abs(offsetX) + abs(offsetY)) * 4.0f, 255.0f), 0.0f);

        // Prevent random colors when there is no flow
        if (abs(offsetX) < 1.0f && abs(offsetY) < 1.0f) {
            rgb.r = 0;
            rgb.g = 0;
            rgb.b = 0;
        }
    }

    // Convert the RGB flow to YUV and return the appropriate channel
    if (channel == 0) { // Y Channel
        return doBWOutput ? normFlow : ((unsigned short)fmax(fmin(rgb.r * 0.299f + rgb.g * 0.587f + rgb.b * 0.114f, 255.0f), 0.0f) + currPixel) >> 1;
    } else if (channel == 1) { // U Channel
        return doBWOutput ? 128 : fmax(fmin(rgb.r * -0.168736f + rgb.g * -0.331264f + rgb.b * 0.5f + 128.0f, 255.0f), 0.0f);
    } else { // V Channel
        return doBWOutput ? 128 : fmax(fmin(rgb.r * 0.5f + rgb.g * -0.418688f + rgb.b * -0.081312f + 128.0f, 255.0f), 0.0f);
    }
}

// Kernel that warps a frame according to the offset array
__kernel void warpFrameKernel(__global const unsigned char* sourceFrame12, __global const unsigned char* sourceFrame21,
                              __global const short* offsetArray,
                              __global unsigned char* outputFrame, const float frameScalar12, const float frameScalar21,
                              const int lowDimX, const int dimY, const int dimX,
                              const int resolutionScalar, const int directionIndexOffset, const int channelIndexOffset,
                              const int frameOutputMode, const float black_level, const float white_level, const int cz) {
    // Current entry to be computed by the thread
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int verticalOffset = dimY >> 2;
    int adjCx = cx;
    int adjCy = cy;

    if (cy >= dimY && cx >= dimX) {
        return;
    }
    
    // SideBySide1 (Left side)
    if (frameOutputMode == 5 && cx < (dimX >> 1)) {
        outputFrame[cz * dimY * dimX + cy * dimX + cx] = sourceFrame12[cz * dimY * dimX + cy * dimX + cx];
        return;
    } else if (frameOutputMode == 6) { // SideBySide2
        const bool isInLeftSide = cy >= (verticalOffset >> cz) && cy < ((verticalOffset >> cz) + (dimY >> (1 + cz))) && cx < (dimX >> 1);
        const bool isInRightSide = cy >= (verticalOffset >> cz) && cy < ((verticalOffset >> cz) + (dimY >> (1 + cz))) && cx >= (dimX >> 1) && cx < dimX;
    
        if (isInLeftSide) { // Place the source frame in the left side of the output frame
            outputFrame[cz * dimY * dimX + cy * dimX + cx] = sourceFrame12[cz * dimY * dimX + ((cy - (verticalOffset >> cz)) << 1) * dimX + (cx << 1) + (cz ? (cx & 1) : 0)];
            return;
        } else if (isInRightSide) { // Place the warped frame in the right side of the output frame
            adjCx = (cx - (dimX >> 1)) << 1;
            adjCy = (cy - (verticalOffset >> cz)) << 1;
        } else { // Fill the surrounding area with black
            outputFrame[cz * dimY * dimX + cy * dimX + cx] = cz ? 128 : 0;
            return;
        }
    }

    // Get the current flow values
    const int scaledCx = cz ? (adjCx >> resolutionScalar) & ~1 : (adjCx >> resolutionScalar);  // The X-Index of the current thread in the offset array
    const int scaledCy = cz ? (adjCy >> resolutionScalar) << 1 : (adjCy >> resolutionScalar);  // The Y-Index of the current thread in the offset array
    const int offsetX = offsetArray[scaledCy * lowDimX + scaledCx];
    const int offsetY = offsetArray[directionIndexOffset + scaledCy * lowDimX + scaledCx];

    // GreyFlow
    if (frameOutputMode == 4) {
        outputFrame[cz * dimY * dimX + cy * dimX + cx] = cz ? 128 : visualizeFlow(offsetX, offsetY, 0, 0, 1);
        return;
    }

    // Get the new pixel position
    const int newCx12 = mirrorCoordinate(adjCx + (int)round((float)(offsetX) * frameScalar12), dimX);
    const int newCy12 = mirrorCoordinate(adjCy + (int)round((float)(offsetY) * frameScalar12 * (cz ? 0.5f : 1.0f)), cz ? (dimY >> 1) : dimY);
    const int newCx21 = mirrorCoordinate(adjCx - (int)round((float)(offsetX) * frameScalar21), dimX);
    const int newCy21 = mirrorCoordinate(adjCy - (int)round((float)(offsetY) * frameScalar21 * (cz ? 0.5f : 1.0f)), cz ? (dimY >> 1) : dimY);

    if (frameOutputMode == 0) { // WarpedFrame12
        outputFrame[cz * dimY * dimX + cy * dimX + cx] = sourceFrame12[cz * dimY * dimX + newCy12 * dimX + (newCx12 & (cz ? ~1 : ~0)) + (cx & (cz ? 1 : 0))];
    } else if (frameOutputMode == 1) { // WarpedFrame21
        outputFrame[cz * dimY * dimX + cy * dimX + cx] = sourceFrame21[cz * dimY * dimX + newCy21 * dimX + (newCx21 & (cz ? ~1 : ~0)) + (cx & (cz ? 1 : 0))];
    } else { // BlendedFrame
        unsigned char blendedValue = (float)sourceFrame12[cz * dimY * dimX + newCy12 * dimX + (newCx12 & (cz ? ~1 : ~0)) + (cx & (cz ? 1 : 0))] * frameScalar21 + 
                                        (float)sourceFrame21[cz * dimY * dimX + newCy21 * dimX + (newCx21 & (cz ? ~1 : ~0)) + (cx & (cz ? 1 : 0))] * frameScalar12;
        if (frameOutputMode == 3) { // HSVFlow
            blendedValue = visualizeFlow(-offsetX, -offsetY, blendedValue, cz + (cx & (cz ? 1 : 0)), frameOutputMode == 4);
        }
        outputFrame[cz * dimY * dimX + cy * dimX + cx] = cz ? apply_levelsUV(blendedValue, white_level) : apply_levelsY(blendedValue, black_level, white_level);
    }
}