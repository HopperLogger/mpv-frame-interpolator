// Quality Adjustments
#define MAX_RES_SCALAR 2 // The maximum resolution scalar used to calculate the optical flow (0: Full resolution, 1: Half resolution, 2: Quarter resolution, 3: Eighth resolution, 4: Sixteenth resolution, ...)

#define NUM_ITERATIONS 0 // How many times the window size used in the ofc calculation will be halved. This controls how precise the optical flow gets. (0: As often as possible)

#define MIN_SEARCH_RADIUS 5  // The minimum window size used in the ofc calculation
#define MAX_SEARCH_RADIUS 16 // The maximum window size used in the ofc calculation

#define MAX_NUM_STEPS 1 // The maximum number of times to repeat each iteration to find the best offset for each window (allows farther movement to be detected)

// Performance Adjustments
#define AUTO_FRAME_SCALE 1          // Whether to automatically reduce/increase the calculation resolution depending on performance (0: Disabled, 1: Enabled)
#define AUTO_SEARCH_RADIUS_ADJUST 1 // Whether to automatically reduce/increase the number of calculation steps and window size depending on performance (0: Disabled, 1: Enabled)

#define UPPER_PERF_BUFFER 1.4 // The upper performance buffer, i.e. calc_time * upper_buffer > frame_time triggers quality reduction
#define LOWER_PERF_BUFFER 1.6 // The lower performance buffer, i.e. calc_time * lower_buffer < frame_time triggers quality improvement

// Debugging
#define DUMP_IMAGES 0          // Whether or not to dump the warped frames to the 'dump' folder (0: Disabled, 1: Enabled)
#define INC_APP_IND 1          // Whether or not to include the AppIndicator (0: Disabled, 1: Enabled)
#define SAVE_STATS 0           // Whether or not to save the ofc Calc Times to a log file (0: Disabled, 1: Enabled)
#define FLOW_BLUR_ENABLED 1    // Whether or not to blur the flow arrays (0: Disabled, 1: Enabled)