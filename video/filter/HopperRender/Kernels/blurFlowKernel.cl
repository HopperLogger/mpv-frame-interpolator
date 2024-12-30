#define BLOCK_SIZE 16
#define KERNEL_RADIUS 4

// Helper function to mirror the coordinate if it is outside the bounds
int mirrorCoordinate(int pos, int dim) {
    if (pos >= dim) {
        return dim - (pos - dim + 1);
    } else if (pos < 0) {
        return -pos - 1;
    }
    return pos;
}

// Kernel that blurs a flow array
__kernel void blurFlowKernel(__global const short* offsetArray12, __global const short* offsetArray21,
                             __global short* blurredOffsetArray12, __global short* blurredOffsetArray21, const int dimY,
                             const int dimX) {
    // Thread and global indices
    int tx = get_local_id(0);   // Local thread x index
    int ty = get_local_id(1);   // Local thread y index
    int gx = get_global_id(0);  // Global x index
    int gy = get_global_id(1);  // Global y index
    int gz = get_global_id(2);  // Global z index (layer index)
    const bool is12 = gz < 2;
    if (!is12) gz -= 2;
    __global const short* input = is12 ? offsetArray12 : offsetArray21;
    __global short* output = is12 ? blurredOffsetArray12 : blurredOffsetArray21;

    // If the kernel radius is 0, just copy the value
    if (KERNEL_RADIUS < 1) {
        output[gz * dimX * dimY + gy * dimX + gx] = input[gz * dimX * dimY + gy * dimX + gx];
        return;
    }

    // Shared memory for the tile, including halos for the blur
    __local short localTile[BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE + 2 * KERNEL_RADIUS];

    // dimX of the shared memory
    int localSizeX = get_local_size(0);
    int localSizeY = get_local_size(1);

    // Local memory coordinates for this thread
    int lx = tx + KERNEL_RADIUS;
    int ly = ty + KERNEL_RADIUS;

    // Load the main data into shared memory
    localTile[ly][lx] = input[gz * dimX * dimY + mirrorCoordinate(gy, dimY) * dimX + mirrorCoordinate(gx, dimX)];

    // Load the halo regions
    // Top and bottom halo
    if (ty < KERNEL_RADIUS) {
        int haloYTop = mirrorCoordinate(gy - KERNEL_RADIUS, dimY);
        int haloYBottom = mirrorCoordinate(gy + localSizeY, dimY);
        localTile[ty][lx] = input[gz * dimX * dimY + haloYTop * dimX + mirrorCoordinate(gx, dimX)];
        localTile[ty + localSizeY + KERNEL_RADIUS][lx] = input[gz * dimX * dimY + haloYBottom * dimX + mirrorCoordinate(gx, dimX)];
    }

    // Left and right halo
    if (tx < KERNEL_RADIUS) {
        int haloXLeft = mirrorCoordinate(gx - KERNEL_RADIUS, dimX);
        int haloXRight = mirrorCoordinate(gx + localSizeX, dimX);
        localTile[ly][tx] = input[gz * dimX * dimY + mirrorCoordinate(gy, dimY) * dimX + haloXLeft];
        localTile[ly][tx + localSizeX + KERNEL_RADIUS] = input[gz * dimX * dimY + mirrorCoordinate(gy, dimY) * dimX + haloXRight];
    }

    // Corner halo
    if (tx < KERNEL_RADIUS && ty < KERNEL_RADIUS) {
        int haloXLeft = mirrorCoordinate(gx - KERNEL_RADIUS, dimX);
        int haloXRight = mirrorCoordinate(gx + localSizeX, dimX);
        int haloYTop = mirrorCoordinate(gy - KERNEL_RADIUS, dimY);
        int haloYBottom = mirrorCoordinate(gy + localSizeY, dimY);
        localTile[ty][tx] = input[gz * dimX * dimY + haloYTop * dimX + haloXLeft];  // Top Left square
        localTile[ty][tx + localSizeX + KERNEL_RADIUS] = input[gz * dimX * dimY + haloYTop * dimX + haloXRight];  // Top Right square
        localTile[ty + localSizeY + KERNEL_RADIUS][tx] = input[gz * dimX * dimY + haloYBottom * dimX + haloXLeft];  // Bottom Left square
        localTile[ty + localSizeY + KERNEL_RADIUS][tx + localSizeX + KERNEL_RADIUS] = input[gz * dimX * dimY + haloYBottom * dimX + haloXRight];  // Bottom Right square
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
        output[gz * dimX * dimY + gy * dimX + gx] = (short)(sum / kernelSize);
    }
}