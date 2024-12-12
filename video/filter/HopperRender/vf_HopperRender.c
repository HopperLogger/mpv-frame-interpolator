#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

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
    BlurredFrames,
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
    double currentSourcePTS;  // The current presentation time stamp (PTS) of the video
    double currentOutputPTS;  // The current presentation time stamp (PTS) of the actual playback
    double sourceFPS;         // The fps of the source video
    double playbackSpeed;     // The speed of the playback
    double sourceFrameTime;   // The current time between source frames (1 / sourceFPS)

    // Optical flow calculation
    struct OpticalFlowCalc *ofc;     // Optical flow calculator struct
    double blendingScalar;           // The scalar used to determine the position between frame1 and frame2
    volatile bool isOFCBusy;         // Whether or not the optical flow calculation is currently running
    bool hasOFCFailed;               // Whether or not the optical flow calculation has failed
    int performanceAdjustmentDelay;  // The number of frames to not adjust the performance after a change

    // Frame output
    struct mp_image_pool *imagePool;  // The software image pool used to store the source frames
    int interpolatedFrameNum;         // The current interpolated frame number
    int sourceFrameNum;  // Frame counter (relative! i.e. number of source frames received since last seek or playback
                         // start)
    int numIntFrames;    // Number of interpolated frames for every source frame

    // Performance and activation status
    int numTooSlow;  // The number of times the interpolation has been too slow
    InterpolationState
        interpolationState;       // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
    struct timeval startTimeOFC;  // The start time of the optical flow calculation
    struct timeval endTimeOFC;    // The end time of the optical flow calculation
    struct timeval startTimeWarping[100];  // The start times of the warp calculations
    struct timeval endTimeWarping[100];    // The end times of the warp calculations
    double opticalFlowCalcDuration;        // The duration of the optical flow calculation
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
void *vf_HopperRender_optical_flow_calc_thread(void *arg);

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
                priv->ofc->opticalFlowCalcShouldTerminate = true;
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
        case 6:
            priv->frameOutputMode = BlurredFrames;
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
        // **** Shaders ****
        case 10:
            priv->ofc->outputBlackLevel = 0.0f;
            priv->ofc->outputWhiteLevel = 220.0f;
            priv->ofc->outputBlackLevel *= 256.0f;
            priv->ofc->outputWhiteLevel *= 256.0f;
            break;
        case 11:
            priv->ofc->outputBlackLevel = 16.0f;
            priv->ofc->outputWhiteLevel = 219.0f;
            priv->ofc->outputBlackLevel *= 256.0f;
            priv->ofc->outputWhiteLevel *= 256.0f;
            break;
        case 12:
            priv->ofc->outputBlackLevel = 10.0f;
            priv->ofc->outputWhiteLevel = 225.0f;
            priv->ofc->outputBlackLevel *= 256.0f;
            priv->ofc->outputWhiteLevel *= 256.0f;
            break;
        case 13:
            priv->ofc->outputBlackLevel = 0.0f;
            priv->ofc->outputWhiteLevel = 255.0f;
            priv->ofc->outputBlackLevel *= 256.0f;
            priv->ofc->outputWhiteLevel *= 256.0f;
            break;
        default:
            break;
    }
}

/*
 * Sends the current info to the AppIndicator status widget
 *
 * @param priv: The video filter private data
 */
static void vf_HopperRender_update_AppIndicator_widget(struct priv *priv, double warpCalcDurations[100],
                                                       double currTotalWarpDuration) {
    char buffer2[512];
    memset(buffer2, 0, sizeof(buffer2));
    int offset =
        snprintf(buffer2, sizeof(buffer2),
                 "Num Steps: %d\nSearch Radius: %d\nCalc Res: %dx%d\nTarget Time: %06.2f ms (%.1f fps)\nFrame Time: "
                 "%06.2f ms (%.3f fps | %.2fx)\nOfc: %06.2f ms (%.0f fps)\nWarp Time: %06.2f ms (%.0f fps)",
                 priv->ofc->opticalFlowSteps, priv->ofc->opticalFlowSearchRadius,
                 priv->ofc->frameWidth >> priv->ofc->opticalFlowResScalar,
                 priv->ofc->frameHeight >> priv->ofc->opticalFlowResScalar, priv->targetFrameTime * 1000.0,
                 1.0 / priv->targetFrameTime, priv->sourceFrameTime * 1000.0, 1.0 / priv->sourceFrameTime,
                 priv->playbackSpeed, priv->opticalFlowCalcDuration * 1000.0, 1.0 / priv->opticalFlowCalcDuration,
                 currTotalWarpDuration * 1000.0, 1.0 / currTotalWarpDuration);

    for (int i = 0; i < 10; i++) {
        if (i < min(priv->numIntFrames, 10)) {
            offset += snprintf(buffer2 + offset, sizeof(buffer2) - offset, "\nWarp%d: %06.2f ms", i,
                               warpCalcDurations[i] * 1000.0);
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
    mp_image_pool_clear(priv->imagePool);
    if (priv->isFilterInitialized) {
        mp_image_unrefp(&priv->referenceImage);
        priv->ofc->opticalFlowCalcShouldTerminate = true;
        pthread_kill(priv->m_ptOFCThreadID, SIGTERM);
        pthread_join(priv->m_ptOFCThreadID, NULL);
        freeOFC(priv->ofc);
    }
    free(priv->ofc);
    talloc_free(priv->opts);
#if INC_APP_IND
    close(priv->m_iAppIndicatorFileDesc);
    vf_HopperRender_terminate_AppIndicator_script(&priv->m_tdAppIndicatorThreadData);
#endif
}

#if !DUMP_IMAGES
/*
 * Applies the new resolution scalar to the optical flow calculator and clears the offset arrays
 *
 * @param f: The video filter instance
 */
static void vf_HopperRender_reinit_ofc(struct mp_filter *f) {
    struct priv *priv = f->priv;
    priv->ofc->opticalFlowSearchRadius = priv->ofc->opticalFlowSearchRadius;
    // Here we just adjust all the variables that are affected by the new resolution scalar
    if (priv->interpolationState == TooSlow) priv->interpolationState = Active;
    priv->ofc->opticalFlowResScalar = priv->ofc->opticalFlowResScalar;
    priv->ofc->opticalFlowFrameWidth = priv->ofc->frameWidth >> priv->ofc->opticalFlowResScalar;
    priv->ofc->opticalFlowFrameHeight = priv->ofc->frameHeight >> priv->ofc->opticalFlowResScalar;
    priv->ofc->directionIndexOffset = priv->ofc->opticalFlowFrameHeight * priv->ofc->opticalFlowFrameWidth;
    priv->ofc->layerIndexOffset = 2 * priv->ofc->opticalFlowFrameHeight * priv->ofc->opticalFlowFrameWidth;
    ERR_CHECK(setKernelParameters(priv->ofc), "reinit", f);
}
#endif

/*
 * Adjust optical flow calculation settings for optimal performance and quality
 *
 * @param f: The video filter instance
 * @param OFCisDone: Whether or not the optical flow calculation finished on time
 */
static void vf_HopperRender_auto_adjust_settings(struct mp_filter *f, const bool ofcFinishedOnTime) {
    struct priv *priv = f->priv;

    // Calculate the calculation durations
    double warpCalcDurations[100];
    double currTotalWarpDuration = 0.0;
    for (int i = 0; i < min(priv->numIntFrames, 100); i++) {
        warpCalcDurations[i] = (priv->endTimeWarping[i].tv_sec - priv->startTimeWarping[i].tv_sec) +
                               ((priv->endTimeWarping[i].tv_usec - priv->startTimeWarping[i].tv_usec) / 1000000.0);
        currTotalWarpDuration += warpCalcDurations[i];
    }
    double currTotalCalcDuration = max(priv->opticalFlowCalcDuration, currTotalWarpDuration);

// Send the stats to the AppIndicator status widget
#if INC_APP_IND
    vf_HopperRender_update_AppIndicator_widget(priv, warpCalcDurations, currTotalWarpDuration);
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
    sprintf(buffer, "%f\n", priv->opticalFlowCalcDuration);
    fprintf(file, "%s", buffer);
    fclose(file);
#endif

#if !DUMP_IMAGES
    /*
     * Calculation took too long (OFC had to be interupted)
     */
    if (!ofcFinishedOnTime) {
        // OFC interruption is critical, so we reduce the resolution
        if (AUTO_FRAME_SCALE && priv->ofc->opticalFlowResScalar < 5) {
            priv->ofc->opticalFlowResScalar += 1;
            vf_HopperRender_reinit_ofc(f);
            priv->performanceAdjustmentDelay = 3;
        }
        return;
    }

    // Adjust the performance buffer delay
    if (priv->performanceAdjustmentDelay > 0) {
        priv->performanceAdjustmentDelay -= 1;
        return;
    }

    if ((currTotalCalcDuration * UPPER_PERF_BUFFER) > priv->sourceFrameTime) {
        /*
         * Calculation took longer than the threshold
         */
        if (AUTO_SEARCH_RADIUS_ADJUST && priv->ofc->opticalFlowSearchRadius > MIN_SEARCH_RADIUS) {
            // Decrease the number of steps to reduce calculation time
            priv->ofc->opticalFlowSearchRadius = max(priv->ofc->opticalFlowSearchRadius - 1, MIN_SEARCH_RADIUS);
            priv->ofc->opticalFlowSearchRadius = priv->ofc->opticalFlowSearchRadius;
            adjustSearchRadius(priv->ofc, priv->ofc->opticalFlowSearchRadius);

        } else if (AUTO_SEARCH_RADIUS_ADJUST && priv->ofc->opticalFlowSteps > 1) {
            // We can't reduce the search radius any further, so we reduce the number of steps instead
            priv->ofc->opticalFlowSteps = max(priv->ofc->opticalFlowSteps - 1, 1);

        } else if (AUTO_FRAME_SCALE && priv->ofc->opticalFlowResScalar < 5) {
            // We can't reduce the number of steps any further, so we reduce the resolution divider instead
            // To avoid adjustments, we only adjust the resolution divider if we have been too slow for a while
            priv->numTooSlow += 1;
            if (priv->numTooSlow > 1) {
                priv->numTooSlow = 0;
                priv->ofc->opticalFlowResScalar += 1;
                vf_HopperRender_reinit_ofc(f);
            }
        }

        priv->performanceAdjustmentDelay = 3;

        // Disable Interpolation if we are too slow
        if ((AUTO_FRAME_SCALE || AUTO_SEARCH_RADIUS_ADJUST) && ((currTotalCalcDuration * 1.05) > priv->sourceFrameTime))
            priv->interpolationState = TooSlow;

    } else if ((currTotalCalcDuration * LOWER_PERF_BUFFER) < priv->sourceFrameTime) {
        /*
         * We have left over capacity
         */
        // Increase the frame scalar if we have enough leftover capacity
        if (AUTO_FRAME_SCALE && priv->ofc->opticalFlowResScalar > priv->ofc->opticalFlowMinResScalar &&
            priv->ofc->opticalFlowSearchRadius >= MAX_SEARCH_RADIUS) {
            priv->numTooSlow = 0;
            priv->ofc->opticalFlowResScalar -= 1;
            priv->ofc->opticalFlowSearchRadius = MIN_SEARCH_RADIUS;
            vf_HopperRender_reinit_ofc(f);
        } else if (AUTO_SEARCH_RADIUS_ADJUST && priv->ofc->opticalFlowSearchRadius < MAX_SEARCH_RADIUS) {
            priv->ofc->opticalFlowSearchRadius = min(priv->ofc->opticalFlowSearchRadius + 1, MAX_SEARCH_RADIUS);
            priv->ofc->opticalFlowSearchRadius = priv->ofc->opticalFlowSearchRadius;
            adjustSearchRadius(priv->ofc, priv->ofc->opticalFlowSearchRadius);
        } else if (AUTO_SEARCH_RADIUS_ADJUST && priv->ofc->opticalFlowSteps < MAX_NUM_STEPS) {
            priv->ofc->opticalFlowSteps = priv->ofc->opticalFlowSteps + 1;
            priv->ofc->opticalFlowSearchRadius = MIN_SEARCH_RADIUS;
            priv->ofc->opticalFlowSearchRadius = priv->ofc->opticalFlowSearchRadius;
            adjustSearchRadius(priv->ofc, priv->ofc->opticalFlowSearchRadius);
        }
        priv->performanceAdjustmentDelay = 3;
    }
#endif
}

/*
 * The optical flow calculation thread
 *
 * @param arg: The video filter private data (struct priv)
 */
void *vf_HopperRender_optical_flow_calc_thread(void *arg) {
    sigset_t set;
    int sig;
    struct priv *priv = (struct priv *)arg;

    // Block signals SIGUSR1 and SIGTERM in the thread
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, NULL);

    // Loop until SIGUSR1 or SIGTERM is received
    while (1) {
        sigwait(&set, &sig);

        if (sig == SIGUSR1) {
            gettimeofday(&priv->startTimeOFC, NULL);
            priv->isOFCBusy = true;

            bool needsFlipping = (priv->frameOutputMode == WarpedFrame21 || priv->frameOutputMode == BlendedFrame ||
                                  priv->frameOutputMode == SideBySide1 || priv->frameOutputMode == SideBySide2);

            if (calculateOpticalFlow(priv->ofc) || (needsFlipping && flipFlow(priv->ofc)) ||
                blurFlowArrays(priv->ofc) ||
                (priv->ofc->opticalFlowCalcShouldTerminate && setKernelParameters(priv->ofc))) {
                priv->hasOFCFailed = true;
            }

            gettimeofday(&priv->endTimeOFC, NULL);
            priv->opticalFlowCalcDuration = (priv->endTimeOFC.tv_sec - priv->startTimeOFC.tv_sec) +
                                            ((priv->endTimeOFC.tv_usec - priv->startTimeOFC.tv_usec) / 1000000.0);
            priv->isOFCBusy = false;
            priv->ofc->opticalFlowCalcShouldTerminate = false;
        } else if (sig == SIGTERM) {
            priv->isOFCBusy = false;
            priv->hasOFCFailed = true;
            priv->ofc->opticalFlowCalcShouldTerminate = false;
            pthread_exit(NULL);
        }
    }
}

/*
 * Initializes the video filter.
 *
 * @param f: The video filter instance
 * @param dimY: The height of the video
 * @param dimX: The stride width of the video
 *
 * @return: The result of the configuration
 */
static void vf_HopperRender_init(struct mp_filter *f, int frameHeight, int frameWidth) {
    struct priv *priv = f->priv;

    // Initialize the optical flow calculator
    ERR_CHECK(initOpticalFlowCalc(priv->ofc, frameHeight, frameWidth), "initOpticalFlowCalc", f);

    // Create the optical flow calc thread
    ERR_CHECK(pthread_create(&priv->m_ptOFCThreadID, NULL, vf_HopperRender_optical_flow_calc_thread, priv),
              "pthread_create", f);
}

/*
 * Coordinates the optical flow calc thread and generates the interpolated frames
 *
 * @param f: The video filter instance
 * @param planes: The planes of the output frame
 */
static void vf_HopperRender_interpolate_frame(struct mp_filter *f, unsigned char **outputPlanes) {
    struct priv *priv = f->priv;

    if (priv->interpolatedFrameNum < 100) gettimeofday(&priv->startTimeWarping[priv->interpolatedFrameNum], NULL);

    // Warp frames
    if (priv->frameOutputMode != HSVFlow && priv->frameOutputMode != BlurredFrames) {
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
    if (priv->interpolatedFrameNum < 100) gettimeofday(&priv->endTimeWarping[priv->interpolatedFrameNum], NULL);

    priv->blendingScalar += priv->targetFrameTime / priv->sourceFrameTime;
    if (priv->blendingScalar >= 1.0) {
        priv->blendingScalar -= 1.0;
    }
}

/*
 * Delivers the intermediate frames to the output pin.
 *
 * @param f: The video filter instance
 * @param fScalar: The scalar used to interpolate the frames
 */
static void vf_HopperRender_process_intermediate_frame(struct mp_filter *f) {
    struct priv *priv = f->priv;

    struct mp_image *img =
        mp_image_pool_get(priv->imagePool, IMGFMT_P010, priv->ofc->frameWidth, priv->ofc->frameHeight);
    mp_image_copy_attributes(img, priv->referenceImage);
    struct mp_frame frame = {.type = MP_FRAME_VIDEO, .data = img};

    // Generate the interpolated frame
    vf_HopperRender_interpolate_frame(f, img->planes);

    // Update playback timestamp
    priv->currentOutputPTS += priv->targetFrameTime * priv->playbackSpeed;
    img->pts = priv->currentOutputPTS;

    // Determine if we need to process the next intermediate frame
    if (priv->interpolatedFrameNum < priv->numIntFrames - 1) {
        priv->interpolatedFrameNum += 1;
        mp_filter_internal_mark_progress(f);
    } else {
        priv->interpolatedFrameNum = 0;
    }

    // Deliver the interpolated frame
    mp_pin_in_write(f->ppins[1], frame);
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

    // If the source is already at or above 60 FPS, we don't need interpolation
    priv->sourceFPS = img->nominal_fps;
    priv->sourceFrameTime = 1.0 / (priv->sourceFPS * priv->playbackSpeed);
    if (priv->sourceFrameTime <= priv->targetFrameTime) {
        priv->interpolationState = NotNeeded;
        mp_pin_in_write(f->ppins[1], frame);
        return;
    }

    // Detect if the frame is an end of frame
    if (mp_frame_is_signaling(frame)) {
        mp_pin_in_write(f->ppins[1], frame);
        return;
    }

    // Initialize the filter if needed
    if (!priv->isFilterInitialized) {
        priv->referenceImage = mp_image_new_ref(img);
        vf_HopperRender_init(f, img->h, img->w);
        priv->isFilterInitialized = true;
    }

    // Update timestamps and source information
    priv->sourceFrameNum += 1;
    priv->currentSourcePTS = img->pts;
    if (priv->sourceFrameNum <= 3 || priv->interpolationState != Active) {
        priv->currentOutputPTS =
            img->pts;  // The first three frames we take the original PTS (see output: 1.0, 2.0, 3.0, 3.1, ...)
    } else {
        priv->currentOutputPTS +=
            priv->targetFrameTime * priv->playbackSpeed;  // The rest of the frames we increase in 60fps steps
    }
    img->pts = priv->currentOutputPTS;
    if (priv->interpolationState == Active) {
        priv->numIntFrames =
            (int)floor((1.0 - priv->blendingScalar) / (priv->targetFrameTime / priv->sourceFrameTime)) + 1;
    } else {
        priv->numIntFrames = 1;
    }

    // Check if the OFC is still busy
    bool OFCisDone = true;
    double currCalcTime = 0.0;
    while (priv->isOFCBusy) {
        if (OFCisDone) {
            gettimeofday(&priv->endTimeOFC, NULL);
            currCalcTime = (priv->endTimeOFC.tv_sec - priv->startTimeOFC.tv_sec) +
                           ((priv->endTimeOFC.tv_usec - priv->startTimeOFC.tv_usec) / 1000000.0);
            if (currCalcTime > priv->sourceFrameTime * 0.95) {
                OFCisDone = false;
                MP_WARN(f, "OFC was too slow!\n");
                priv->ofc->opticalFlowCalcShouldTerminate = true;  // Interrupts the OFC calculator
            }
        }
    }

    // Check if the OFC failed
    ERR_CHECK(priv->hasOFCFailed, "OFC failed", f);

    // Adjust the settings to process everything fast enough
    vf_HopperRender_auto_adjust_settings(f, OFCisDone);

    // Upload the source frame to the GPU buffers and blur it for OFC calculation
    ERR_CHECK(updateFrame(priv->ofc, img->planes, priv->frameOutputMode == BlurredFrames), "updateFrame", f);

    // Calculate the optical flow
    if (priv->interpolationState == Active && priv->sourceFrameNum >= 2 && priv->frameOutputMode != BlurredFrames &&
        priv->frameOutputMode != TearingTest) {
        pthread_kill(priv->m_ptOFCThreadID, SIGUSR1);
    }

    // Interpolate the frames
    if (priv->interpolationState == Active && (priv->sourceFrameNum >= 3 || priv->frameOutputMode == SideBySide2) &&
        priv->numIntFrames > 1) {
        vf_HopperRender_interpolate_frame(f, img->planes);
        priv->interpolatedFrameNum = 1;
        mp_filter_internal_mark_progress(f);
    } else if (priv->sourceFrameNum >= 2) {
        // First frame is directly output, therefore we don't change the output buffer
        // After the first frame, we always output the previous frame
        ERR_CHECK(downloadFrame(priv->ofc, priv->ofc->inputFrameArray[1], img->planes), "processFrame", f);
    }

    // Deliver the source frame
    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, img));
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
    }

    // If the source is already at or above 60 FPS, we don't need interpolation
    if (priv->sourceFrameTime <= priv->targetFrameTime) {
        priv->interpolationState = NotNeeded;
    } else if (priv->interpolationState != Deactivated) {
        // Reset the state either because the fps/speed change now requires interpolation, or we could now be fast
        // enough to interpolate
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
    priv->ofc->opticalFlowCalcShouldTerminate = true;
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
    if (MAX_RES_SCALAR < 0 || MAX_RES_SCALAR > 5) {
        MP_ERR(parent, "MAX_RES_SCALAR must be between 0 and 5.\n");
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
    if (MAX_SEARCH_RADIUS > 16) {
        MP_ERR(parent, "MAX_SEARCH_RADIUS must be at most 16.\n");
        return NULL;
    }
    if (MAX_SEARCH_RADIUS < MIN_SEARCH_RADIUS) {
        MP_ERR(parent, "MAX_SEARCH_RADIUS must be greater than or equal to MIN_SEARCH_RADIUS.\n");
        return NULL;
    }
    if (MAX_NUM_STEPS < 1) {
        MP_ERR(parent, "MAX_NUM_STEPS must be at least 1.\n");
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
    priv->currentSourcePTS = 0.0;
    priv->currentOutputPTS = 0.0;
    priv->sourceFPS = 24000.0 / 1001.0;  // Default to 23.976 FPS
    priv->playbackSpeed = 1.0;
    priv->sourceFrameTime = 1001.0 / 24000.0;

    // Optical Flow calculation
    priv->blendingScalar = 0.0;
    priv->isOFCBusy = false;
    priv->hasOFCFailed = false;
    priv->performanceAdjustmentDelay = 0;

    // Frame output
    priv->imagePool = mp_image_pool_new(NULL);
    priv->interpolatedFrameNum = 0;
    priv->sourceFrameNum = 0;
    priv->numIntFrames = 1;

    // Performance and activation status
    priv->numTooSlow = 0;
    priv->interpolationState = Active;
    priv->opticalFlowCalcDuration = 0.0;

    // Optical Flow calculator
    priv->ofc = calloc(1, sizeof(struct OpticalFlowCalc));

    return f;
}

#define OPT_BASE_STRUCT struct HopperRender_opts
static const m_option_t vf_opts_fields[] = {{"FrameOutput", OPT_INT(frame_output), M_RANGE(0, 8), OPTDEF_INT(2)}};

// Filter entry
const struct mp_user_filter_entry vf_HopperRender = {.desc = {.description = "Optical-Flow Frame Interpolation",
                                                              .name = "HopperRender",
                                                              .priv_size = sizeof(struct priv),
                                                              .options = vf_opts_fields},
                                                     .create = vf_HopperRender_create};
