// Quality Adjustments
#define MAX_CALC_RES 270 // The maximum resolution used to calculate the optical flow

#define NUM_ITERATIONS 0 // How many times the window size used in the ofc calculation will be halved. This controls how precise the optical flow gets. (0: As often as possible)

#define MIN_SEARCH_RADIUS 5  // The minimum window size used in the ofc calculation
#define MAX_SEARCH_RADIUS 16 // The maximum window size used in the ofc calculation

// Performance Adjustments
#define AUTO_SEARCH_RADIUS_ADJUST 1 // Whether to automatically reduce/increase the number of calculation steps and window size depending on performance (0: Disabled, 1: Enabled)

#define UPPER_PERF_BUFFER 1.4 // The upper performance buffer, i.e. calc_time * upper_buffer > frame_time triggers quality reduction
#define LOWER_PERF_BUFFER 1.6 // The lower performance buffer, i.e. calc_time * lower_buffer < frame_time triggers quality improvement

// Debugging
#define INC_APP_IND 1 // Whether or not to include the AppIndicator (0: Disabled, 1: Enabled)
#define SAVE_STATS 0  // Whether or not to save the ofc Calc Times to a log file (0: Disabled, 1: Enabled)