#define BLOCK_SIZE 16
#define KERNEL_RADIUS 8

// Kernel that blurs a 2D image
__kernel void blurMaskKernel(__global const int* hitCount,
                             __global float* blurredMask,
                             const int height,
                             const int width) {
	// Shared memory for the tile, including halos for the blur
    __local float localTile[BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE + 2 * KERNEL_RADIUS];

    const float fac = 1.0f;

    // Thread and global indices
    int tx = get_local_id(0); // Local thread x index
    int ty = get_local_id(1); // Local thread y index
    int gx = get_global_id(0); // Global x index
    int gy = get_global_id(1); // Global y index

    // Width of the shared memory
    int localSizeX = get_local_size(0);
    int localSizeY = get_local_size(1);

    // Local memory coordinates for this thread
    int lx = tx + KERNEL_RADIUS;
    int ly = ty + KERNEL_RADIUS;

    // Calculate global index for this thread
	int globalIndex = gy * width + gx;

    // Load the main data into shared memory
    if (gx < width && gy < height) {
        localTile[ly][lx] = hitCount[globalIndex] != 1 ? fac : 0.0f;
    } else {
        localTile[ly][lx] = 0; // Padding for threads outside image bounds
    }

    // Load the halo regions
    // Top and bottom halo
    if (ty < KERNEL_RADIUS) {
        int haloYTop = gy - KERNEL_RADIUS;
        int haloYBottom = gy + localSizeY;
        localTile[ty][lx] = (haloYTop >= 0) ? (hitCount[haloYTop * width + gx] != 1 ? fac : 0.0f) : 0;
        localTile[ty + localSizeY + KERNEL_RADIUS][lx] = (haloYBottom < height) ? (hitCount[haloYBottom * width + gx] != 1 ? fac : 0.0f) : 0;
    }

    // Left and right halo
    if (tx < KERNEL_RADIUS) {
        int haloXLeft = gx - KERNEL_RADIUS;
        int haloXRight = gx + localSizeX;
        localTile[ly][tx] = (haloXLeft >= 0) ? (hitCount[gy * width + haloXLeft] != 1 ? fac : 0.0f) : 0;
        localTile[ly][tx + localSizeX + KERNEL_RADIUS] = (haloXRight < width) ? (hitCount[gy * width + haloXRight] != 1 ? fac : 0.0f) : 0;
    }

	// Corner halo
	if (tx < KERNEL_RADIUS && ty < KERNEL_RADIUS) {
		int haloXLeft = gx - KERNEL_RADIUS;
		int haloXRight = gx + localSizeX;
		int haloYTop = gy - KERNEL_RADIUS;
		int haloYBottom = gy + localSizeY;
		localTile[ty][tx] = (haloYTop >= 0 && haloXLeft >= 0) ? (hitCount[haloYTop * width + haloXLeft] != 1 ? fac : 0.0f) : 0; // Top Left square
		localTile[ty][tx + localSizeX + KERNEL_RADIUS] = (haloYTop >= 0 && haloXRight < width) ? (hitCount[haloYTop * width + haloXRight] != 1 ? fac : 0.0f) : 0; // Top Right square
		localTile[ty + localSizeY + KERNEL_RADIUS][tx] = (haloYBottom < height && haloXLeft >= 0) ? (hitCount[haloYBottom * width + haloXLeft] != 1 ? fac : 0.0f) : 0; // Bottom Left square
		localTile[ty + localSizeY + KERNEL_RADIUS][tx + localSizeX + KERNEL_RADIUS] = (haloYBottom < height && haloXRight < width) ? (hitCount[haloYBottom * width + haloXRight] != 1 ? fac : 0.0f) : 0; // Bottom Right square
	}

    // Wait for all threads to finish loading shared memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the blur operation
    if (gx < width && gy < height) {
        float sum = 0.0f;

        for (int ky = -KERNEL_RADIUS; ky < KERNEL_RADIUS; ++ky) {
            for (int kx = -KERNEL_RADIUS; kx < KERNEL_RADIUS; ++kx) {
                sum += localTile[ly + ky][lx + kx];
            }
        }

        // Average the sum
        int kernelSize = (2 * KERNEL_RADIUS) * (2 * KERNEL_RADIUS);
        blurredMask[globalIndex] = min(sum / (float)kernelSize, 1.0f);
    }
}