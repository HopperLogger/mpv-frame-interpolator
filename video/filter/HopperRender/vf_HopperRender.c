#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>

#include "config.h"
#include "filters/f_autoconvert.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "opticalFlowCalc.h"
#include "video/mp_image_pool.h"

#if INC_APP_IND
typedef struct {
    pid_t pid;
    int pipe_fd[2];
} ThreadData;
#endif

#if DUMP_IMAGES
static int dump_iter = 0;
#endif

typedef enum FrameOutput {
    WarpedFrame12,
    WarpedFrame21,
    BlendedFrame,
    HSVFlow,
    GreyFlow,
    SideBySide1,
    SideBySide2,
    TearingTest
} FrameOutput;

typedef enum InterpolationState { Deactivated, NotNeeded, Active, TooSlow } InterpolationState;

struct HopperRender_opts {
    int frame_output;
};

struct priv {
    // HopperRender options
    struct HopperRender_opts *opts;

    // Autoconverter
    struct mp_autoconvert *conv;

    // Thread data
#if INC_APP_IND
    ThreadData
        m_tdAppIndicatorThreadData;  // Data for the AppIndicator thread used to communicate with the status widget
    int m_iAppIndicatorFileDesc;     // The file descriptor for the AppIndicator status widget
#endif
    pthread_t m_ptOFCThreadID;  // The thread ID of the optical flow calculation thread

    // Settings
    FrameOutput frameOutputMode;  // What frame output to use
    bool isFilterInitialized;     // Whether or not the filter has been initialized
    double targetFrameTime;       // The target presentation time stamp (PTS) of the video

    // Video info
    struct mp_image *referenceImage;  // The reference image used for the optical flow calculation

    // Timings
    double currentOutputPTS;  // The current presentation time stamp (PTS) of the actual playback
    double sourceFPS;         // The fps of the source video
    double playbackSpeed;     // The speed of the playback
    double sourceFrameTime;   // The current time between source frames (1 / sourceFPS)
    bool resync;              // Whether or not the filter should resync

    // Optical flow calculation
    struct OpticalFlowCalc *ofc;     // Optical flow calculator struct
    double blendingScalar;           // The scalar used to determine the position between frame1 and frame2

    // Frame output
    struct mp_image_pool *imagePool;  // The software image pool used to store the source frames
    int interpolatedFrameNum;         // The current interpolated frame number
    int sourceFrameNum;  // Frame counter (relative! i.e. number of source frames received since last seek or playback
                         // start)
    int numIntFrames;    // Number of interpolated frames for every source frame

    // Performance and activation status
    InterpolationState
        interpolationState;         // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
    double warpCalcDurations[10];  // The durations of the warp calculations
    double currTotalCalcDuration;   // The total duration of the current frame calculation
};

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define ERR_CHECK(cond, func, f)           \
    if (cond) {                            \
        MP_ERR(f, "Error in %s\n", func);  \
        vf_HopperRender_uninit(f);         \
        mp_filter_internal_mark_failed(f); \
        return;                            \
    }

// Prototypes
#if INC_APP_IND
void *vf_HopperRender_launch_AppIndicator_script(void *arg);

/*
 * Terminates the AppIndicator widget
 *
 * @param data: The thread data
 */
static void vf_HopperRender_terminate_AppIndicator_script(ThreadData *data) {
    if (data->pid > 0) {
        // Send SIGINT signal to the process to terminate it
        if (kill(data->pid, SIGINT) == -1) {
            perror("kill");
        } else {
            // Wait for the process to terminate
            waitpid(data->pid, NULL, 0);
        }
        close(data->pipe_fd[0]);  // Close read end
    }
}

/*
 * Processes the commands received from the AppIndicator
 *
 * @param priv: The video filter private data
 */
static void vf_HopperRender_process_AppIndicator_command(struct priv *priv) {
    // Read the command from the pipe
    int code = -1;
    char buffer[256];
    ssize_t bytesRead;
    while ((bytesRead = read(priv->m_tdAppIndicatorThreadData.pipe_fd[0], buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytesRead] = '\0';
        if (buffer[0] != '\n') {
            code = atoi(buffer);
            break;
        }
    }

    // Process the command
    switch (code) {
        // **** Activation Status ****
        case 0:
            if (priv->interpolationState != Deactivated) {
                priv->interpolationState = Deactivated;
                priv->interpolatedFrameNum = 0;
            } else {
                priv->interpolationState = Active;
            }
            break;
        // **** Frame Output ****
        case 1:
            priv->frameOutputMode = WarpedFrame12;
            break;
        case 2:
            priv->frameOutputMode = WarpedFrame21;
            break;
        case 3:
            priv->frameOutputMode = BlendedFrame;
            break;
        case 4:
            priv->frameOutputMode = HSVFlow;
            break;
        case 5:
            priv->frameOutputMode = GreyFlow;
            break;
        case 7:
            priv->frameOutputMode = SideBySide1;
            break;
        case 8:
            priv->frameOutputMode = SideBySide2;
            break;
        case 9:
            priv->frameOutputMode = TearingTest;
            break;
        default:
            // Black and White Levels
            if (code >= 100 && code <= 355) {
                priv->ofc->outputBlackLevel = (float)((code - 100) * 256);
            } else if (code >= 400 && code <= 655) {
                priv->ofc->outputWhiteLevel = (float)((code - 400) * 256);
            }
            break;
    }
}

/*
 * Sends the current info to the AppIndicator status widget
 *
 * @param priv: The video filter private data
 */
static void vf_HopperRender_update_AppIndicator_widget(struct priv *priv) {
    char buffer2[512];
    memset(buffer2, 0, sizeof(buffer2));
    int offset =
        snprintf(buffer2, sizeof(buffer2),
                 "Search Radius: %d\nCalc Res: %dx%d\nTarget Time: %06.2f ms (%.1f fps)\nFrame Time: "
                 "%06.2f ms (%.3f fps | %.2fx)\nCalc Time: %06.2f ms (%.0f fps > %.3f fps)",
                 priv->ofc->opticalFlowSearchRadius,
                 priv->ofc->frameWidth >> priv->ofc->opticalFlowResScalar,
                 priv->ofc->frameHeight >> priv->ofc->opticalFlowResScalar, priv->targetFrameTime * 1000.0,
                 1.0 / priv->targetFrameTime, priv->sourceFrameTime * 1000.0, 1.0 / priv->sourceFrameTime,
                 priv->playbackSpeed, priv->currTotalCalcDuration * 1000.0, 1.0 / priv->currTotalCalcDuration, 1.0 / priv->sourceFrameTime);

    for (int i = 0; i < 10; i++) {
        if (i < min(priv->numIntFrames, 10)) {
            offset += snprintf(buffer2 + offset, sizeof(buffer2) - offset, "\nWarp%d: %06.2f ms", i,
                               priv->warpCalcDurations[i] * 1000.0);
        } else {
            offset += snprintf(buffer2 + offset, sizeof(buffer2) - offset, "\n");
        }
    }
    if (write(priv->m_iAppIndicatorFileDesc, buffer2, sizeof(buffer2)) == -1) {
        perror("write");
        close(priv->m_iAppIndicatorFileDesc);
        exit(EXIT_FAILURE);
    }
}

/*
 * Start the AppIndicator widget
 *
 * @param arg: The thread data
 */
void *vf_HopperRender_launch_AppIndicator_script(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    if (pipe(data->pipe_fd) == -1) {
        perror("pipe");
        pthread_exit(NULL);
    }

    data->pid = fork();
    if (data->pid == 0) {
        // Child process
        close(data->pipe_fd[0]);                // Close read end
        dup2(data->pipe_fd[1], STDOUT_FILENO);  // Redirect stdout to pipe
        dup2(data->pipe_fd[1], STDERR_FILENO);  // Redirect stderr to pipe
        close(data->pipe_fd[1]);                // Close original write end

        // Get the home directory from the environment
        const char *home = getenv("HOME");
        if (home == NULL) {
            // Handle error if HOME is not set
            fprintf(stderr, "HOME environment variable is not set.\n");
            return NULL;
        }

        // Create a buffer to store the full path
        char full_path[512];
        snprintf(full_path, sizeof(full_path),
                 "%s/mpv-build/mpv/video/filter/HopperRender/HopperRenderSettingsApplet.py", home);

        // Use the constructed path in execlp
        execlp("python3", "python3", full_path, (char *)NULL);

        // If execlp returns, there was an error
        perror("execlp");
        exit(EXIT_FAILURE);
    } else if (data->pid < 0) {
        // Fork failed
        perror("fork");
        pthread_exit(NULL);
    } else {
        // Parent process
        close(data->pipe_fd[1]);  // Close write end

        // Set the read end of the pipe to non-blocking mode
        int flags = fcntl(data->pipe_fd[0], F_GETFL, 0);
        if (flags == -1) {
            perror("fcntl(F_GETFL)");
            pthread_exit(NULL);
        }
        if (fcntl(data->pipe_fd[0], F_SETFL, flags | O_NONBLOCK) == -1) {
            perror("fcntl(F_SETFL)");
            pthread_exit(NULL);
        }
    }
    return NULL;
}
#endif

/*
 * Frees the video filter.
 *
 * @param f: The video filter instance
 */
static void vf_HopperRender_uninit(struct mp_filter *f) {
    struct priv *priv = f->priv;
    if (priv->isFilterInitialized) {
        freeOFC(priv->ofc);
        mp_image_unrefp(&priv->referenceImage);
    }
    mp_image_pool_clear(priv->imagePool);
    free(priv->ofc);
    talloc_free(priv->opts);
#if INC_APP_IND
    close(priv->m_iAppIndicatorFileDesc);
    vf_HopperRender_terminate_AppIndicator_script(&priv->m_tdAppIndicatorThreadData);
#endif
}

/*
 * Adjust optical flow calculation settings for optimal performance and quality
 *
 * @param f: The video filter instance
 */
static void vf_HopperRender_auto_adjust_settings(struct mp_filter *f) {
    struct priv *priv = f->priv;

// Send the stats to the AppIndicator status widget
#if INC_APP_IND
    vf_HopperRender_update_AppIndicator_widget(priv);
#endif

// Save the stats to a file
#if SAVE_STATS
    FILE *file = fopen("ofclog.txt", "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    char buffer[512];
    memset(buffer, 0, sizeof(buffer));
    sprintf(buffer, "%f\n", priv->currTotalCalcDuration);
    fprintf(file, "%s", buffer);
    fclose(file);
#endif

#if !DUMP_IMAGES && AUTO_SEARCH_RADIUS_ADJUST
    // Check if we were too slow or have leftover capacity
    if ((priv->currTotalCalcDuration * UPPER_PERF_BUFFER) > priv->sourceFrameTime) {
        if (priv->ofc->opticalFlowSearchRadius > MIN_SEARCH_RADIUS) {
            // Decrease the number of steps to reduce calculation time
            priv->ofc->opticalFlowSearchRadius--;
            adjustSearchRadius(priv->ofc, priv->ofc->opticalFlowSearchRadius);
        } else {
            // Disable Interpolation if we are too slow
            priv->interpolationState = TooSlow;
        }

    } else if ((priv->currTotalCalcDuration * LOWER_PERF_BUFFER) < priv->sourceFrameTime) {
        // Increase the frame scalar if we have enough leftover capacity
        if (priv->ofc->opticalFlowSearchRadius < MAX_SEARCH_RADIUS) {
            priv->ofc->opticalFlowSearchRadius++;
            adjustSearchRadius(priv->ofc, priv->ofc->opticalFlowSearchRadius);
        }
    }
#endif
}

/*
 * Coordinates the optical flow calc thread and generates the interpolated frames
 *
 * @param f: The video filter instance
 * @param outputPlanes: The output planes where the interpolated frame will be stored
 */
static void vf_HopperRender_interpolate_frame(struct mp_filter *f, unsigned char **outputPlanes) {
    struct priv *priv = f->priv;

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);

    // Warp frames
    if (priv->frameOutputMode != HSVFlow) {
        ERR_CHECK(
            warpFrames(priv->ofc, priv->blendingScalar, priv->frameOutputMode, (int)(priv->interpolatedFrameNum == 0)),
            "warpFrames", f);
    }

    // Blend the frames together
    if (priv->frameOutputMode == BlendedFrame || priv->frameOutputMode == SideBySide1) {
        ERR_CHECK(blendFrames(priv->ofc, priv->blendingScalar), "blendFrames", f);
    }

    // Special output modes
    if (priv->frameOutputMode == HSVFlow) {
        ERR_CHECK(visualizeFlow(priv->ofc, 0), "visualizeFlow", f);
    } else if (priv->frameOutputMode == GreyFlow) {
        ERR_CHECK(visualizeFlow(priv->ofc, 1), "visualizeFlowGrey", f);
    } else if (priv->frameOutputMode == SideBySide1) {
        ERR_CHECK(sideBySide1(priv->ofc), "sideBySide1", f);
    } else if (priv->frameOutputMode == SideBySide2) {
        ERR_CHECK(sideBySide2(priv->ofc, priv->blendingScalar, priv->sourceFrameNum), "sideBySide2", f);
    } else if (priv->frameOutputMode == TearingTest) {
        ERR_CHECK(tearingTest(priv->ofc), "tearingTest", f);
    }

// Save the result to a file
#if DUMP_IMAGES
    // Get the home directory from the environment
    const char *home = getenv("HOME");
    if (home == NULL) {
        MP_ERR(f, "HOME environment variable is not set.\n");
        mp_filter_internal_mark_failed(f);
        return;
    }
    char path[32];
    snprintf(path, sizeof(path), "%s/dump/%d.bin", home, dump_iter);
    dump_iter++;
    ERR_CHECK(saveImage(priv->ofc, path), "saveImage", f);
#endif

    // Download the result to the output buffer
    ERR_CHECK(downloadFrame(priv->ofc, priv->ofc->outputFrameArray, outputPlanes), "downloadFrame", f);
    gettimeofday(&endTime, NULL);
    if (priv->interpolatedFrameNum < 10) priv->warpCalcDurations[priv->interpolatedFrameNum] =
        (endTime.tv_sec - startTime.tv_sec) + ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);
    priv->currTotalCalcDuration += (endTime.tv_sec - startTime.tv_sec) + ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);

    priv->blendingScalar += priv->targetFrameTime / priv->sourceFrameTime;
    if (priv->blendingScalar >= 1.0) {
        priv->blendingScalar -= 1.0;
    }
}

// Helper function to truncate a double to a certain number of decimal places
static double truncate_to_floor(double value, int decimal_places) {
    double multiplier = pow(10.0, decimal_places);
    return floor(value * multiplier) / multiplier;
}

/*
 * Delivers the intermediate frames to the output pin.
 *
 * @param f: The video filter instance
 */
static void vf_HopperRender_process_intermediate_frame(struct mp_filter *f) {
    struct priv *priv = f->priv;

    struct mp_image *img =
        mp_image_pool_get(priv->imagePool, IMGFMT_P010, priv->ofc->frameWidth, priv->ofc->frameHeight);
    mp_image_copy_attributes(img, priv->referenceImage);

    // Generate the interpolated frame
    vf_HopperRender_interpolate_frame(f, img->planes);

    // Update playback timestamp
    priv->currentOutputPTS += truncate_to_floor(priv->targetFrameTime * priv->playbackSpeed, 3);
    img->pts = priv->currentOutputPTS;

    // Determine if we need to process the next intermediate frame
    if (priv->interpolatedFrameNum < priv->numIntFrames - 1) {
        priv->interpolatedFrameNum += 1;
        mp_filter_internal_mark_progress(f);
    } else {
        priv->interpolatedFrameNum = 0;
    }

    // Deliver the interpolated frame
    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, img));
}

/*
 * Processes and delivers a new source frame.
 *
 * @param f: The video filter instance
 */
static void vf_HopperRender_process_new_source_frame(struct mp_filter *f) {
    struct priv *priv = f->priv;

    // Read the new source frame
    struct mp_frame frame = mp_pin_out_read(priv->conv->f->pins[1]);
    struct mp_image *img = frame.data;

    // Detect if the frame is an end of frame
    if (mp_frame_is_signaling(frame)) {
        goto output;
    }

    // Retrieve the source frame time
    if (img->nominal_fps > 0.0) {
        priv->sourceFPS = img->nominal_fps;
    }
    priv->sourceFrameTime = 1.0 / (priv->sourceFPS * priv->playbackSpeed);

    // If the source is already at or above 60 FPS, we don't need interpolation
    if (priv->sourceFrameTime <= priv->targetFrameTime) {
        priv->interpolationState = NotNeeded;
        goto output;
    } else if (priv->interpolationState == NotNeeded) {
        priv->interpolationState = Active;
    } else if (priv->interpolationState != Active) {
        goto output;
    }

    // Initialize the filter if needed
    if (!priv->isFilterInitialized) {
        priv->referenceImage = mp_image_new_ref(img);
        ERR_CHECK(initOpticalFlowCalc(priv->ofc, img->h, img->w), "initOpticalFlowCalc", f);
        priv->isFilterInitialized = true;
    }

    // Update timestamps and source information
    priv->sourceFrameNum += 1;
    if (priv->sourceFrameNum <= 3 || priv->resync) {
        priv->currentOutputPTS =
            img->pts;  // The first three frames we take the original PTS (see output: 1.0, 2.0, 3.0, 3.1, ...)
    } else {
        // The rest of the frames we increase in 60fps steps
        priv->currentOutputPTS += truncate_to_floor(priv->targetFrameTime * priv->playbackSpeed, 3);  
        img->pts = priv->currentOutputPTS;
    }

    // Calculate the number of interpolated frames
    priv->numIntFrames = (int)max(ceil((1.0 - priv->blendingScalar) / (priv->targetFrameTime / priv->sourceFrameTime)), 1.0);
    
    // Adjust the settings to process everything fast enough
    vf_HopperRender_auto_adjust_settings(f);

    // Upload the source frame to the GPU buffers
    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    ERR_CHECK(updateFrame(priv->ofc, img->planes), "updateFrame", f);

    // Calculate the optical flow
    if (priv->sourceFrameNum >= 2 && priv->frameOutputMode != TearingTest) {
        bool needsFlipping = (priv->frameOutputMode == WarpedFrame21 || priv->frameOutputMode == BlendedFrame ||
                                priv->frameOutputMode == SideBySide1 || priv->frameOutputMode == SideBySide2);

        ERR_CHECK(calculateOpticalFlow(priv->ofc) || 
                 (needsFlipping && flipFlow(priv->ofc)) ||
                  blurFlowArrays(priv->ofc), "OFC failed", f);
    }
    gettimeofday(&endTime, NULL);
    priv->currTotalCalcDuration = (endTime.tv_sec - startTime.tv_sec) + ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);

    // Interpolate the frames
    if ((priv->sourceFrameNum >= 3 || priv->frameOutputMode == SideBySide2)) {
        vf_HopperRender_interpolate_frame(f, img->planes);
        if (priv->numIntFrames > 1) {
            priv->interpolatedFrameNum = 1;
            mp_filter_internal_mark_progress(f);
        }
    } else if (priv->sourceFrameNum >= 2) {
        // First frame is directly output, therefore we don't change the output buffer
        // After the first frame, we always output the previous frame
        ERR_CHECK(downloadFrame(priv->ofc, priv->ofc->inputFrameArray[1], img->planes), "processFrame", f);
    }

output:
    mp_pin_in_write(f->ppins[1], frame);
}

/*
 * Main filter process function. Called on every new source frame.
 *
 * @param f: The video filter instance
 */
static void vf_HopperRender_process(struct mp_filter *f) {
    struct priv *priv = f->priv;

    // Convert the incoming frames using the autoconvert filter (Any -> P010)
    if (mp_pin_can_transfer_data(priv->conv->f->pins[0], f->ppins[0])) {
        mp_pin_in_write(priv->conv->f->pins[0], mp_pin_out_read(f->ppins[0]));
    }

    // Process intermediate frames
    if (priv->interpolatedFrameNum > 0 && mp_pin_in_needs_data(f->ppins[1])) {
        vf_HopperRender_process_intermediate_frame(f);
        return;
    }

    // Process a new source frame
    if (priv->interpolatedFrameNum == 0 && mp_pin_can_transfer_data(f->ppins[1], priv->conv->f->pins[1])) {
#if INC_APP_IND
        vf_HopperRender_process_AppIndicator_command(priv);
#endif
        vf_HopperRender_process_new_source_frame(f);
    }
}

/*
 * Processes commands send to the video filter (by mpv).
 *
 * @param f: The video filter instance
 * @param cmd: The command to process
 */
static bool vf_HopperRender_command(struct mp_filter *f, struct mp_filter_command *cmd) {
    struct priv *priv = f->priv;

    // Speed change event
    if (cmd->type == MP_FILTER_COMMAND_TEXT && priv->playbackSpeed != cmd->speed) {
        priv->playbackSpeed = cmd->speed;
        priv->resync = true;
    }

    // Reset the state either because we could now be fast enough to interpolate
    if (priv->interpolationState != Deactivated) {
        priv->interpolationState = Active;
    }
    return true;
}

/*
 * Resets the video filter on seek
 *
 * @param f: The video filter instance
 */
static void vf_HopperRender_reset(struct mp_filter *f) {
    struct priv *priv = f->priv;
    priv->sourceFrameNum = 0;
    priv->interpolatedFrameNum = 0;
    priv->blendingScalar = 0.0;
}

// Filter definition
static const struct mp_filter_info vf_HopperRender_filter = {.name = "HopperRender",
                                                             .process = vf_HopperRender_process,
                                                             .priv_size = sizeof(struct priv),
                                                             .reset = vf_HopperRender_reset,
                                                             .destroy = vf_HopperRender_uninit,
                                                             .command = vf_HopperRender_command};

/*
 * Creates the video filter and initializes the private data.
 *
 * @param parent: The parent filter
 * @param options: The filter options
 */
static struct mp_filter *vf_HopperRender_create(struct mp_filter *parent, void *options) {
    // Validate the defines
    if (MAX_CALC_RES < 64) {
        MP_ERR(parent, "MAX_CALC_RES must be at least 64.\n");
        return NULL;
    }
    if (NUM_ITERATIONS < 0) {
        MP_ERR(parent,
               "NUM_ITERATIONS must be a positive number. Set it to 0 to automatically use the maximum number of "
               "iterations.\n");
        return NULL;
    }
    if (MIN_SEARCH_RADIUS < 2) {
        MP_ERR(parent, "MIN_SEARCH_RADIUS must be at least 2.\n");
        return NULL;
    }
    if (MAX_SEARCH_RADIUS < 2) {
        MP_ERR(parent, "MAX_SEARCH_RADIUS must be at least 2.\n");
        return NULL;
    }
    if (MAX_SEARCH_RADIUS > 256) {
        MP_ERR(parent, "MAX_SEARCH_RADIUS must be at most 256.\n");
        return NULL;
    }
    if (MAX_SEARCH_RADIUS < MIN_SEARCH_RADIUS) {
        MP_ERR(parent, "MAX_SEARCH_RADIUS must be greater than or equal to MIN_SEARCH_RADIUS.\n");
        return NULL;
    }
    if (UPPER_PERF_BUFFER < 1.0) {
        MP_ERR(parent, "UPPER_PERF_BUFFER must be at least 1.0.\n");
        return NULL;
    }
    if (LOWER_PERF_BUFFER < 1.0) {
        MP_ERR(parent, "LOWER_PERF_BUFFER must be at least 1.0.\n");
        return NULL;
    }
    if (LOWER_PERF_BUFFER < UPPER_PERF_BUFFER) {
        MP_ERR(parent, "UPPER_PERF_BUFFER must be less than or equal to LOWER_PERF_BUFFER.\n");
        return NULL;
    }

    // Create the video filter
    struct mp_filter *f = mp_filter_create(parent, &vf_HopperRender_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }
    struct priv *priv = f->priv;
    priv->opts = talloc_steal(priv, options);

#if INC_APP_IND
    // Create the fifo for the AppIndicator widget if it doesn't exist
    char *fifo = "/tmp/hopperrender";
    if (access(fifo, F_OK) == -1) {
        if (mkfifo(fifo, 0666) != 0) {
            MP_ERR(f, "Failed to create pipe for HopperRender.\n");
            talloc_free(options);
            return NULL;
        }
    }

    // Launch the AppIndicator widget
    pthread_t thread;
    priv->m_iAppIndicatorFileDesc = -1;
    pthread_create(&thread, NULL, vf_HopperRender_launch_AppIndicator_script, &priv->m_tdAppIndicatorThreadData);
    pthread_detach(thread);
    sleep(1);
    priv->m_iAppIndicatorFileDesc = open(fifo, O_WRONLY);
    if (priv->m_iAppIndicatorFileDesc == -1) {
        MP_ERR(f, "Failed to open AppIndicator FIFO.\n");
        talloc_free(options);
        return NULL;
    }
#endif

    // Connect the input and output pins
    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    // Initialize the autoconvert filter
    priv->conv = mp_autoconvert_create(f);
    if (!priv->conv) {
        talloc_free(options);
        return NULL;
    }
    mp_autoconvert_add_imgfmt(priv->conv, IMGFMT_P010, 0);

    // Thread data
    priv->m_ptOFCThreadID = 0;

    // Settings
    priv->frameOutputMode = (FrameOutput)priv->opts->frame_output;
    priv->isFilterInitialized = false;
    double display_fps = 60.0;
#if !DUMP_IMAGES
    struct mp_stream_info *info = mp_filter_find_stream_info(f);
    if (info) {
        if (info->get_display_fps) display_fps = info->get_display_fps(info);  // Set the target FPS to the display FPS
    }
#endif
    priv->targetFrameTime = 1.0 / display_fps;

    // Video info
    priv->referenceImage = calloc(1, sizeof(struct mp_image));

    // Timings
    priv->currentOutputPTS = 0.0;
    priv->sourceFPS = 24000.0 / 1001.0;  // Default to 23.976 FPS
    priv->playbackSpeed = 1.0;
    priv->sourceFrameTime = 1001.0 / 24000.0;
    priv->resync = false;

    // Optical Flow calculation
    priv->blendingScalar = 0.0;

    // Frame output
    priv->imagePool = mp_image_pool_new(NULL);
    priv->interpolatedFrameNum = 0;
    priv->sourceFrameNum = 0;
    priv->numIntFrames = 1;

    // Performance and activation status
    priv->interpolationState = Active;
    priv->currTotalCalcDuration = 0.0;

    // Optical Flow calculator
    priv->ofc = calloc(1, sizeof(struct OpticalFlowCalc));

    return f;
}

#define OPT_BASE_STRUCT struct HopperRender_opts
static const m_option_t vf_opts_fields[] = {{"FrameOutput", OPT_INT(frame_output), M_RANGE(0, 7), OPTDEF_INT(2)}};

// Filter entry
const struct mp_user_filter_entry vf_HopperRender = {.desc = {.description = "Optical-Flow Frame Interpolation",
                                                              .name = "HopperRender",
                                                              .priv_size = sizeof(struct priv),
                                                              .options = vf_opts_fields},
                                                     .create = vf_HopperRender_create};
