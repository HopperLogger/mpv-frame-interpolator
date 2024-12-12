#define BLOCK_SIZE 16
#define KERNEL_RADIUS 4

// Kernel that blurs a 2D image
__kernel void blurFrameKernel(__global const unsigned short* input,
                              __global unsigned short* output,
                              const int dimY,
                              const int dimX) {
	// Shared memory for the tile, including halos for the blur
    __local unsigned short localTile[BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE + 2 * KERNEL_RADIUS];

    // Thread and global indices
    int tx = get_local_id(0); // Local thread x index
    int ty = get_local_id(1); // Local thread y index
    int gx = get_global_id(0); // Global x index
    int gy = get_global_id(1); // Global y index

    // dimX of the shared memory
    int localSizeX = get_local_size(0);
    int localSizeY = get_local_size(1);

    // Local memory coordinates for this thread
    int lx = tx + KERNEL_RADIUS;
    int ly = ty + KERNEL_RADIUS;

    // Calculate global index for this thread
	int globalIndex = gy * dimX + gx;

    // Load the main data into shared memory
    if (gx < dimX && gy < dimY) {
        localTile[ly][lx] = input[globalIndex];
    } else {
        localTile[ly][lx] = 0; // Padding for threads outside image bounds
    }

    // Load the halo regions
    // Top and bottom halo
    if (ty < KERNEL_RADIUS) {
        int haloYTop = gy - KERNEL_RADIUS;
        int haloYBottom = gy + localSizeY;
        localTile[ty][lx] = (haloYTop >= 0) ? input[haloYTop * dimX + gx] : 0;
        localTile[ty + localSizeY + KERNEL_RADIUS][lx] = (haloYBottom < dimY) ? input[haloYBottom * dimX + gx] : 0;
    }

    // Left and right halo
    if (tx < KERNEL_RADIUS) {
        int haloXLeft = gx - KERNEL_RADIUS;
        int haloXRight = gx + localSizeX;
        localTile[ly][tx] = (haloXLeft >= 0) ? input[gy * dimX + haloXLeft] : 0;
        localTile[ly][tx + localSizeX + KERNEL_RADIUS] = (haloXRight < dimX) ? input[gy * dimX + haloXRight] : 0;
    }

	// Corner halo
	if (tx < KERNEL_RADIUS && ty < KERNEL_RADIUS) {
		int haloXLeft = gx - KERNEL_RADIUS;
		int haloXRight = gx + localSizeX;
		int haloYTop = gy - KERNEL_RADIUS;
		int haloYBottom = gy + localSizeY;
		localTile[ty][tx] = (haloYTop >= 0 && haloXLeft >= 0) ? input[haloYTop * dimX + haloXLeft] : 0; // Top Left square
		localTile[ty][tx + localSizeX + KERNEL_RADIUS] = (haloYTop >= 0 && haloXRight < dimX) ? input[haloYTop * dimX + haloXRight] : 0; // Top Right square
		localTile[ty + localSizeY + KERNEL_RADIUS][tx] = (haloYBottom < dimY && haloXLeft >= 0) ? input[haloYBottom * dimX + haloXLeft] : 0; // Bottom Left square
		localTile[ty + localSizeY + KERNEL_RADIUS][tx + localSizeX + KERNEL_RADIUS] = (haloYBottom < dimY && haloXRight < dimX) ? input[haloYBottom * dimX + haloXRight] : 0; // Bottom Right square
	}

    // Wait for all threads to finish loading shared memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the blur operation
    if (gx < dimX && gy < dimY) {
        int sum = 0;

        for (int ky = -KERNEL_RADIUS; ky < KERNEL_RADIUS; ++ky) {
            for (int kx = -KERNEL_RADIUS; kx < KERNEL_RADIUS; ++kx) {
                sum += localTile[ly + ky][lx + kx];
            }
        }

        // Average the sum
        int kernelSize = (2 * KERNEL_RADIUS) * (2 * KERNEL_RADIUS);
        output[globalIndex] = (unsigned short)(sum / kernelSize);
    }
}