#define INITIAL_RESOLUTION_SCALAR 2 // The initial resolution scalar (0: Full resolution, 1: Half resolution, 2: Quarter resolution, 3: Eighth resolution, 4: Sixteenth resolution, ...)
#define FRAME_BLUR_KERNEL_SIZE 16 // The kernel size used to blur the source frames
#define FLOW_BLUR_KERNEL_SIZE 8 // The kernel size used to blur the flow arrays
#define INITIAL_SEARCH_RADIUS 8 // The initial radius in which the optical flow calculation will search for the best match
#define NUM_ITERATIONS 11 // Number of iterations to use in the optical flow calculation (0: As many as possible)
#define AUTO_FRAME_SCALE 1 // Whether to automatically reduce/increase the calculation resolution depending on performance (0: Disabled, 1: Enabled)
#define AUTO_SEARCH_RADIUS_ADJUST 1 // Whether to automatically reduce/increase the number of calculation steps depending on performance (0: Disabled, 1: Enabled)
#define MIN_SEARCH_RADIUS 5 // The minimum number of calculation steps (if below this, resolution will be decreased or calculation disabled)
#define MAX_SEARCH_RADIUS 11 // The maximum number of calculation steps (if reached, resolution will be increased or steps will be kept at this number)
#define LOG_PERFORMANCE 0 // Whether or not to print debug messages regarding calculation performance (0: Disabled, 1: Enabled)
#define MAX_NUM_BUFFERED_IMG 50 // The maximum number of buffered images allowed to be in the image pool
#define DUMP_IMAGES 0 // Whether or not to dump the images to disk (0: Disabled, 1: Enabled)