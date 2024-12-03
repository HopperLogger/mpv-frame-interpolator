#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "common/msg.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "video/mp_image_pool.h"
#include "filters/f_autoconvert.h"
#include "opticalFlowCalc.h"

#define INITIAL_RESOLUTION_SCALAR 2 // The initial resolution scalar (0: Full resolution, 1: Half resolution, 2: Quarter resolution, 3: Eighth resolution, 4: Sixteenth resolution, ...)
#define NUM_ITERATIONS 11 // Number of iterations to use in the optical flow calculation (0: As many as possible)
#define AUTO_FRAME_SCALE 0 // Whether to automatically reduce/increase the calculation resolution depending on performance (0: Disabled, 1: Enabled)
#define LOG_PERFORMANCE 0 // Whether or not to print debug messages regarding calculation performance (0: Disabled, 1: Enabled)
#define MAX_NUM_BUFFERED_IMG 50 // The maximum number of buffered images allowed to be in the image pool
#define DUMP_IMAGES 0 // Whether or not to dump the images to disk (0: Disabled, 1: Enabled)

typedef struct {
    pid_t pid;
    int pipe_fd[2];
} ThreadData;

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

typedef enum InterpolationState {
    Deactivated,
	NotNeeded,
	Active,
	TooSlow
} InterpolationState;

struct priv {
	// Autoconverter
	struct mp_autoconvert *conv;

    // Thread data
    ThreadData m_tdAppIndicatorThreadData; // Data for the AppIndicator thread used to communicate with the status widget
	int m_iAppIndicatorFileDesc; // The file descriptor for the AppIndicator status widget
	pthread_t m_ptOFCThreadID; // The thread ID of the optical flow calculation thread

    // Settings
	FrameOutput m_foFrameOutput; // What frame output to use
	int m_iNumIterations; // Number of iterations to use in the optical flow calculation (0: As many as possible)
	bool m_bInitialized; // Whether or not the filter has been initialized
	double m_dTargetPTS; // The target presentation time stamp (PTS) of the video
	
	// Video info
	int m_iDimX; // The width of the frame
	int m_iDimY; // The height of the frame
	struct mp_image *m_miRefImage; // The reference image used for the optical flow calculation

	// Timings
	double m_dCurrSourcePTS; // The current presentation time stamp (PTS) of the video
	double m_dCurrPlaybackPTS; // The current presentation time stamp (PTS) of the actual playback
	double m_dSourceFPS; // The fps of the source video
    double m_dPlaybackSpeed; // The speed of the playback
	double m_dSourceFrameTime; // The current time between source frames (1 / m_dSourceFPS)
	
	// Optical flow calculation
	struct OpticalFlowCalc* ofc; // Optical flow calculator struct
	int m_cResolutionScalar; // Determines which resolution scalar will be used for the optical flow calculation (0: Full resolution, 1: Half resolution, 2: Quarter resolution, 3: Eighth resolution, 4: Sixteenth resolution, ...)
	double m_dScalar; // The scalar used to determine the position between frame1 and frame2
	volatile bool m_bOFCBusy; // Whether or not the optical flow calculation is currently running
	bool m_bOFCFailed; // Whether or not the optical flow calculation has failed

	// Frame output
	struct mp_image_pool *m_miSWPool; // The software image pool used to store the source frames
	int m_iIntFrameNum; // The current interpolated frame number
	int m_iFrameCounter; // Frame counter (relative! i.e. number of source frames received since last seek or playback start)
	int m_iNumIntFrames; // Number of interpolated frames for every source frame

	// Performance and activation status
	int m_cNumTimesTooSlow; // The number of times the interpolation has been too slow
	InterpolationState m_isInterpolationState; // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
	struct timeval m_teOFCCalcStart; // The start time of the optical flow calculation
	struct timeval m_teOFCCalcEnd; // The end time of the optical flow calculation
	struct timeval m_teWarpCalcStart[100]; // The start times of the warp calculations
	struct timeval m_teWarpCalcEnd[100]; // The end times of the warp calculations
	double m_dOFCCalcDuration; // The duration of the optical flow calculation
};

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define ERR_CHECK(cond, func, f) if (cond) { MP_ERR(f, "Error in %s\n", func); vf_HopperRender_uninit(f); mp_filter_internal_mark_failed(f); return; }

// Prototypes
void *vf_HopperRender_optical_flow_calc_thread(void *arg);
void *vf_HopperRender_launch_AppIndicator_script(void *arg);

/*
* Terminates the AppIndicator widget
*
* @param data: The thread data
*/
static void vf_HopperRender_terminate_AppIndicator_script(ThreadData *data)
{
    if (data->pid > 0) {
        // Send SIGINT signal to the process to terminate it
        if (kill(data->pid, SIGINT) == -1) {
            perror("kill");
        } else {
            // Wait for the process to terminate
            waitpid(data->pid, NULL, 0);
        }
        close(data->pipe_fd[0]); // Close read end
    }
}

/*
* Frees the video filter.
*
* @param f: The video filter instance
*/
static void vf_HopperRender_uninit(struct mp_filter *f)
{
    struct priv *priv = f->priv;
	mp_image_unrefp(&priv->m_miRefImage);
	mp_image_pool_clear(priv->m_miSWPool);
	close(priv->m_iAppIndicatorFileDesc);
	if (priv->m_bInitialized) {
		priv->ofc->m_bOFCTerminate = true;
		pthread_kill(priv->m_ptOFCThreadID, SIGTERM);
    	pthread_join(priv->m_ptOFCThreadID, NULL);
		freeOFC(priv->ofc);
	}
	free(priv->ofc);
	vf_HopperRender_terminate_AppIndicator_script(&priv->m_tdAppIndicatorThreadData);
}

/*
* Processes the commands received from the AppIndicator
*
* @param priv: The video filter private data
*/
static void vf_HopperRender_process_AppIndicator_command(struct priv *priv)
{
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
			if (priv->m_isInterpolationState != Deactivated) {
				priv->m_isInterpolationState = Deactivated;
				priv->m_iIntFrameNum = 0;
				priv->ofc->m_bOFCTerminate = true;
			} else {
				priv->m_isInterpolationState = Active;
			}
			break;
		// **** Frame Output ****
		case 1:
			priv->m_foFrameOutput = WarpedFrame12;
			break;
		case 2:
			priv->m_foFrameOutput = WarpedFrame21;
			break;
		case 3:
			priv->m_foFrameOutput = BlendedFrame;
			break;
		case 4:
			priv->m_foFrameOutput = HSVFlow;
			break;
		case 5:
			priv->m_foFrameOutput = GreyFlow;
			break;
		case 6:
			priv->m_foFrameOutput = BlurredFrames;
			break;
		case 7:
			priv->m_foFrameOutput = SideBySide1;
			break;
		case 8:
			priv->m_foFrameOutput = SideBySide2;
			break;
		case 9:
			priv->m_foFrameOutput = TearingTest;
			break;
		// **** Shaders ****
		case 10:
			priv->ofc->m_fBlackLevel = 0.0f;
			priv->ofc->m_fWhiteLevel = 220.0f;
			priv->ofc->m_fBlackLevel *= 256.0f;
			priv->ofc->m_fWhiteLevel *= 256.0f;
			break;
		case 11:
			priv->ofc->m_fBlackLevel = 16.0f;
			priv->ofc->m_fWhiteLevel = 219.0f;
			priv->ofc->m_fBlackLevel *= 256.0f;
			priv->ofc->m_fWhiteLevel *= 256.0f;
			break;
		case 12:
			priv->ofc->m_fBlackLevel = 10.0f;
			priv->ofc->m_fWhiteLevel = 225.0f;
			priv->ofc->m_fBlackLevel *= 256.0f;
			priv->ofc->m_fWhiteLevel *= 256.0f;
			break;
		case 13:
			priv->ofc->m_fBlackLevel = 0.0f;
			priv->ofc->m_fWhiteLevel = 255.0f;
			priv->ofc->m_fBlackLevel *= 256.0f;
			priv->ofc->m_fWhiteLevel *= 256.0f;
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
static void vf_HopperRender_update_AppIndicator_widget(struct priv *priv, double warpCalcDurations[100], double currTotalWarpDuration)
{
	char buffer2[512];
	memset(buffer2, 0, sizeof(buffer2));
    int offset = snprintf(buffer2, sizeof(buffer2), 
                          "Calc Res: %dx%d\nTarget Time: %06.2f ms (%.1f fps)\nFrame Time: %06.2f ms (%.3f fps | %.2fx)\nOfc: %06.2f ms (%.0f fps)\nWarp Time: %06.2f ms (%.0f fps)",
                          priv->m_iDimX >> priv->m_cResolutionScalar, priv->m_iDimY >> priv->m_cResolutionScalar, priv->m_dTargetPTS * 1000.0, 1.0 / priv->m_dTargetPTS,
						  priv->m_dSourceFrameTime * 1000.0, 1.0 / priv->m_dSourceFrameTime, priv->m_dPlaybackSpeed, priv->m_dOFCCalcDuration * 1000.0, 1.0 / priv->m_dOFCCalcDuration, currTotalWarpDuration * 1000.0, 1.0 / currTotalWarpDuration);

    for (int i = 0; i < 10; i++) {
		if (i < min(priv->m_iNumIntFrames, 10)) {
        	offset += snprintf(buffer2 + offset, sizeof(buffer2) - offset, "\nWarp%d: %06.2f ms", i, warpCalcDurations[i] * 1000.0);
		} else {
			offset += snprintf(buffer2 + offset, sizeof(buffer2) - offset, "\n");
		}
    }
	if (write(priv->m_iAppIndicatorFileDesc, buffer2, sizeof(buffer2)) == -1) {
		perror("write");
		close(priv->m_iAppIndicatorFileDesc);
		exit(EXIT_FAILURE);
	}

	// Save the stats to a file
/* 	FILE *file = fopen("/home/julian/ofclog.txt", "a");
	if (file == NULL) {
		perror("Error opening file");
		return;
	}
	char buffer[512];
	memset(buffer, 0, sizeof(buffer));
	sprintf(buffer, "%f\n", priv->m_dOFCCalcDuration);
	fprintf(file, "%s", buffer);
	fclose(file); */
}

/*
* Applies the new resolution scalar to the optical flow calculator and clears the offset arrays
*
* @param f: The video filter instance
*/
static void vf_HopperRender_reinit_ofc(struct mp_filter *f)
{
	struct priv *priv = f->priv;
	// Here we just adjust all the variables that are affected by the new resolution scalar
	if (priv->m_isInterpolationState == TooSlow) priv->m_isInterpolationState = Active;
	priv->ofc->m_cResolutionScalar = priv->m_cResolutionScalar;
	priv->ofc->m_iLowDimX = priv->m_iDimX >> priv->m_cResolutionScalar;
	priv->ofc->m_iLowDimY = priv->m_iDimY >> priv->m_cResolutionScalar;
	priv->ofc->m_iDirectionIdxOffset = priv->ofc->m_iLowDimY * priv->ofc->m_iLowDimX;
	priv->ofc->m_iLayerIdxOffset = 2 * priv->ofc->m_iLowDimY * priv->ofc->m_iLowDimX;
	ERR_CHECK(setKernelParameters(priv->ofc), "reinit", f);
}

/*
* Adjust optical flow calculation settings for optimal performance and quality
*
* @param f: The video filter instance
* @param OFCisDone: Whether or not the optical flow calculation finished on time
*/
static void vf_HopperRender_auto_adjust_settings(struct mp_filter *f, const bool OFCisDone)
{
	struct priv *priv = f->priv;

	// Calculate the calculation durations
	double warpCalcDurations[100];
	double currTotalWarpDuration = 0.0;
	for (int i = 0; i < min(priv->m_iNumIntFrames, 100); i++) {
		warpCalcDurations[i] = (priv->m_teWarpCalcEnd[i].tv_sec - priv->m_teWarpCalcStart[i].tv_sec) + ((priv->m_teWarpCalcEnd[i].tv_usec - priv->m_teWarpCalcStart[i].tv_usec) / 1000000.0);
		currTotalWarpDuration += warpCalcDurations[i];
	}
	double currTotalCalcDuration = max(priv->m_dOFCCalcDuration, currTotalWarpDuration);
	
	// Send the stats to the AppIndicator status widget
	vf_HopperRender_update_AppIndicator_widget(priv, warpCalcDurations, currTotalWarpDuration);

	/*
	* Calculation took too long (OFC had to be interupted)
	*/
	if (!OFCisDone) {
		// OFC interruption is critical, so we reduce the resolution
		if (AUTO_FRAME_SCALE && priv->m_cResolutionScalar < 5) {
			priv->m_cResolutionScalar += 1;
			vf_HopperRender_reinit_ofc(f);
		}
		return;

	/*
	* Calculation took longer than the threshold
	*/
	} else if ((currTotalCalcDuration + currTotalCalcDuration * 0.2) > priv->m_dSourceFrameTime) {
		if (LOG_PERFORMANCE) MP_TRACE(f, "Calculation took too long %.3f sec AVG SFT: %.3f RES SCALAR: %d\n", currTotalCalcDuration, priv->m_dSourceFrameTime, priv->m_cResolutionScalar);

		// We can't reduce the number of steps any further, so we reduce the resolution divider instead
		if (AUTO_FRAME_SCALE && priv->m_cResolutionScalar < 5) {
			// To avoid unnecessary adjustments, we only adjust the resolution divider if we have been too slow for a while
			priv->m_cNumTimesTooSlow += 1;
			if (priv->m_cNumTimesTooSlow > 1) {
				priv->m_cNumTimesTooSlow = 0;
				priv->m_cResolutionScalar += 1;
				vf_HopperRender_reinit_ofc(f);
			}
			return;
		}

		// Disable Interpolation if we are too slow
		if (AUTO_FRAME_SCALE && ((currTotalCalcDuration + currTotalCalcDuration * 0.05) > priv->m_dSourceFrameTime)) priv->m_isInterpolationState = TooSlow;

	/*
	* We have left over capacity
	*/
	} else if ((currTotalCalcDuration + currTotalCalcDuration * 0.4) < priv->m_dSourceFrameTime) {
		if (LOG_PERFORMANCE) MP_TRACE(f, "Calculation has capacity %.3f sec AVG SFT: %.3f RES SCALAR: %d\n", currTotalCalcDuration, priv->m_dSourceFrameTime, priv->m_cResolutionScalar);

		// Increase the frame scalar if we have enough leftover capacity
		if (AUTO_FRAME_SCALE && priv->m_cResolutionScalar > 2) {
			priv->m_cNumTimesTooSlow = 0;
			priv->m_cResolutionScalar -= 1;
			vf_HopperRender_reinit_ofc(f);
		}

	/*
	* Calculation took as long as it should
	*/
	} else {
		if (LOG_PERFORMANCE) MP_TRACE(f, "Calculation took %.3f sec AVG SFT: %.3f RES SCALAR: %d\n", currTotalCalcDuration, priv->m_dSourceFrameTime, priv->m_cResolutionScalar);
	}
}

/*
* The optical flow calculation thread
*
* @param arg: The video filter private data (struct priv)
*/
void *vf_HopperRender_optical_flow_calc_thread(void *arg)
{
    sigset_t set;
    int sig;
	struct priv *priv = (struct priv*)arg;
    
    // Block signals SIGUSR1 and SIGTERM in the thread
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, NULL);
    
    // Loop until SIGUSR1 or SIGTERM is received
    while (1) {
        sigwait(&set, &sig);
        if (sig == SIGUSR1) {
			priv->m_bOFCBusy = true;

			// Calculate the optical flow (frame 1 to frame 2)
			if (calculateOpticalFlow(priv->ofc, priv->m_iNumIterations)) goto fail;

			// Flip the flow array to get the flow from frame 2 to frame 1
			if (priv->m_foFrameOutput == WarpedFrame21 || 
				priv->m_foFrameOutput == BlendedFrame || 
				priv->m_foFrameOutput == SideBySide1 || 
				priv->m_foFrameOutput == SideBySide2) {
				if (flipFlow(priv->ofc)) goto fail;
			}

			// Blur the flow arrays
			if (blurFlowArrays(priv->ofc)) goto fail;

			// Clear the flow arrays if we aborted calculation
			if (priv->ofc->m_bOFCTerminate) {
				if (setKernelParameters(priv->ofc)) goto fail;
			}

			// Collect the calculation duration
			gettimeofday(&priv->m_teOFCCalcEnd, NULL);
			priv->m_dOFCCalcDuration = (priv->m_teOFCCalcEnd.tv_sec - priv->m_teOFCCalcStart.tv_sec) + ((priv->m_teOFCCalcEnd.tv_usec - priv->m_teOFCCalcStart.tv_usec) / 1000000.0);

			priv->m_bOFCBusy = false;
			priv->ofc->m_bOFCTerminate = false;

        } else if (sig == SIGTERM) {
			fail:
			priv->m_bOFCBusy = false;
			priv->m_bOFCFailed = true;
			priv->ofc->m_bOFCTerminate = false;
			pthread_exit(NULL);
            break;
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
static void vf_HopperRender_init(struct mp_filter *f, int dimY, int dimX)
{
	struct priv *priv = f->priv;

	priv->m_iDimX = dimX;
	priv->m_iDimY = dimY;

	// Initialize the optical flow calculator
	ERR_CHECK(initOpticalFlowCalc(priv->ofc, dimY, dimX, priv->m_cResolutionScalar), "initOpticalFlowCalc", f);
	
	// Create the optical flow calc thread
	ERR_CHECK(pthread_create(&priv->m_ptOFCThreadID, NULL, vf_HopperRender_optical_flow_calc_thread, priv), "pthread_create", f);
}

/*
* Coordinates the optical flow calc thread and generates the interpolated frames
*
* @param f: The video filter instance
* @param planes: The planes of the output frame
*/
static int dump_iter = 0;
static void vf_HopperRender_interpolate_frame(struct mp_filter *f, unsigned char** planes)
{
	struct priv *priv = f->priv;
	
	if (priv->m_iIntFrameNum < 100) gettimeofday(&priv->m_teWarpCalcStart[priv->m_iIntFrameNum], NULL);
	
	// Warp frames
	if (priv->m_foFrameOutput != HSVFlow && 
		priv->m_foFrameOutput != BlurredFrames) {
		ERR_CHECK(warpFrames(priv->ofc, priv->m_dScalar, priv->m_foFrameOutput), "warpFrames", f);
	}
	
	// Blend the frames together
	if (priv->m_foFrameOutput == BlendedFrame || 
		priv->m_foFrameOutput == SideBySide1) {
		ERR_CHECK(blendFrames(priv->ofc, priv->m_dScalar), "blendFrames", f);
	}
	
	// Draw the flow as an HSV image
	if (priv->m_foFrameOutput == HSVFlow) {
		ERR_CHECK(drawFlowAsHSV(priv->ofc, 0.5f), "drawFlowAsHSV", f);
	// Draw the flow as a grayscale image
	} else if (priv->m_foFrameOutput == GreyFlow) {
		ERR_CHECK(drawFlowAsGrayscale(priv->ofc), "drawFlowAsGrayscale", f);
	// Show side by side comparison
	} else if (priv->m_foFrameOutput == SideBySide1) {
		ERR_CHECK(insertFrame(priv->ofc), "insertFrame", f);
	} else if (priv->m_foFrameOutput == SideBySide2) {
	    ERR_CHECK(sideBySideFrame(priv->ofc, priv->m_dScalar, priv->m_iFrameCounter), "sideBySideFrame", f);
	} else if (priv->m_foFrameOutput == TearingTest) {
		ERR_CHECK(tearingTest(priv->ofc), "tearingTest", f);
	}

	// Save the result to a file
	if (DUMP_IMAGES) {
		// Get the home directory from the environment
		const char* home = getenv("HOME");
		if (home == NULL) {
			MP_ERR(f, "HOME environment variable is not set.\n");
			mp_filter_internal_mark_failed(f);
			return;
		}
		char path[32];
		snprintf(path, sizeof(path), "%s/dump/%d.bin", home, dump_iter);
		dump_iter++;
		ERR_CHECK(saveImage(priv->ofc, path), "saveImage", f);
	}
	
	// Download the result to the output buffer
	ERR_CHECK(downloadFrame(priv->ofc, priv->ofc->m_outputFrame, planes), "downloadFrame", f);
	if (priv->m_iIntFrameNum < 100) gettimeofday(&priv->m_teWarpCalcEnd[priv->m_iIntFrameNum], NULL);

	priv->m_dScalar += priv->m_dTargetPTS / priv->m_dSourceFrameTime;
	if (priv->m_dScalar >= 1.0) {
		priv->m_dScalar -= 1.0;
	}
}

/*
* Delivers the intermediate frames to the output pin.
*
* @param f: The video filter instance
* @param fScalar: The scalar used to interpolate the frames
*/
static void vf_HopperRender_process_intermediate_frame(struct mp_filter *f)
{
	struct priv *priv = f->priv;

	struct mp_image *img = mp_image_pool_get(priv->m_miSWPool, IMGFMT_P010, priv->m_iDimX, priv->m_iDimY);
	mp_image_copy_attributes(img, priv->m_miRefImage);
	struct mp_frame frame = {.type = MP_FRAME_VIDEO, .data = img};
	
    // Generate the interpolated frame
    vf_HopperRender_interpolate_frame(f, img->planes);
	
    // Update playback timestamp
    priv->m_dCurrPlaybackPTS += priv->m_dTargetPTS * priv->m_dPlaybackSpeed;
    img->pts = priv->m_dCurrPlaybackPTS;

    // Determine if we need to process the next intermediate frame
	if (priv->m_iIntFrameNum < priv->m_iNumIntFrames - 1) {
		priv->m_iIntFrameNum += 1;
		mp_filter_internal_mark_progress(f);
	} else {
		priv->m_iIntFrameNum = 0;
	}
	
    // Deliver the interpolated frame
    mp_pin_in_write(f->ppins[1], frame);
}

/*
* Processes and delivers a new source frame.
*
* @param f: The video filter instance
*/
static void vf_HopperRender_process_new_source_frame(struct mp_filter *f)
{
	struct priv *priv = f->priv;

    // Read the new source frame
    struct mp_frame frame = mp_pin_out_read(priv->conv->f->pins[1]);
    struct mp_image *img = frame.data;
	
	// Detect if the frame is an end of frame
	if (mp_frame_is_signaling(frame)) {
        mp_pin_in_write(f->ppins[1], frame);
        return;
    }

    // Initialize the filter if needed
    if (!priv->m_bInitialized) {
		priv->m_dSourceFPS = img->nominal_fps;
		priv->m_miRefImage = mp_image_new_ref(img);
        vf_HopperRender_init(f, img->h, img->w);
        priv->m_bInitialized = true;
    }

    // Update timestamps and source information
	priv->m_iFrameCounter += 1;
	priv->m_dCurrSourcePTS = img->pts;
    if (priv->m_iFrameCounter <= 3 || priv->m_isInterpolationState != Active) {
        priv->m_dCurrPlaybackPTS = img->pts; // The first three frames we take the original PTS (see output: 1.0, 2.0, 3.0, 3.1, ...)
    } else {
        priv->m_dCurrPlaybackPTS += priv->m_dTargetPTS * priv->m_dPlaybackSpeed; // The rest of the frames we increase in 60fps steps
    }
	img->pts = priv->m_dCurrPlaybackPTS;
    priv->m_dSourceFrameTime = 1.0 / (priv->m_dSourceFPS * priv->m_dPlaybackSpeed);
	if (priv->m_isInterpolationState == Active) {
		priv->m_iNumIntFrames = (int)floor((1.0 - priv->m_dScalar) / (priv->m_dTargetPTS / priv->m_dSourceFrameTime)) + 1;
	} else {
		priv->m_iNumIntFrames = 1;
	}

	// Check if the OFC is still busy
	bool OFCisDone = true;
	while (priv->m_bOFCBusy) {
		if (OFCisDone) {
			OFCisDone = false;
			MP_WARN(f, "OFC was too slow!\n");
			priv->ofc->m_bOFCTerminate = true; // Interrupts the OFC calculator
		}
	}

	// Check if the OFC failed
	ERR_CHECK(priv->m_bOFCFailed, "OFC failed", f);

	// Adjust the settings to process everything fast enough
	vf_HopperRender_auto_adjust_settings(f, OFCisDone);

    // Upload the source frame to the GPU buffers and blur it for OFC calculation
    gettimeofday(&priv->m_teOFCCalcStart, NULL);
    ERR_CHECK(updateFrame(priv->ofc, img->planes, priv->m_foFrameOutput == BlurredFrames), "updateFrame", f);

	// Calculate the optical flow
	if (priv->m_isInterpolationState == Active && priv->m_iFrameCounter >= 2 && priv->m_foFrameOutput != BlurredFrames && priv->m_foFrameOutput != TearingTest) {
		pthread_kill(priv->m_ptOFCThreadID, SIGUSR1);
	}

	// Interpolate the frames
    if (priv->m_isInterpolationState == Active && (priv->m_iFrameCounter >= 3 || priv->m_foFrameOutput == SideBySide2) && priv->m_iNumIntFrames > 1) {
		vf_HopperRender_interpolate_frame(f, img->planes);
        priv->m_iIntFrameNum = 1;
        mp_filter_internal_mark_progress(f);
    } else {
        ERR_CHECK(processFrame(priv->ofc, img->planes, priv->m_iFrameCounter), "processFrame", f);
    }

    // Deliver the source frame
    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, img));
}

/*
* Main filter process function. Called on every new source frame.
*
* @param f: The video filter instance
*/
static void vf_HopperRender_process(struct mp_filter *f)
{
    struct priv *priv = f->priv;

	// Convert the incoming frames using the autoconvert filter (Any -> P010)
	if (mp_pin_can_transfer_data(priv->conv->f->pins[0], f->ppins[0])) {
        mp_pin_in_write(priv->conv->f->pins[0], mp_pin_out_read(f->ppins[0]));
	}

    // Process intermediate frames
    if (priv->m_iIntFrameNum > 0 && mp_pin_in_needs_data(f->ppins[1])) {
        vf_HopperRender_process_intermediate_frame(f);
        return;
    }

    // Process a new source frame
    if (priv->m_iIntFrameNum == 0 && mp_pin_can_transfer_data(f->ppins[1], priv->conv->f->pins[1])) {
		vf_HopperRender_process_AppIndicator_command(priv);
        vf_HopperRender_process_new_source_frame(f);
	}
}

/*
* Start the AppIndicator widget
*
* @param arg: The thread data
*/
void *vf_HopperRender_launch_AppIndicator_script(void *arg)
{
	ThreadData *data = (ThreadData *)arg;
    if (pipe(data->pipe_fd) == -1) {
        perror("pipe");
        pthread_exit(NULL);
    }

    data->pid = fork();
    if (data->pid == 0) {
        // Child process
        close(data->pipe_fd[0]); // Close read end
        dup2(data->pipe_fd[1], STDOUT_FILENO); // Redirect stdout to pipe
        dup2(data->pipe_fd[1], STDERR_FILENO); // Redirect stderr to pipe
        close(data->pipe_fd[1]); // Close original write end

        // Get the home directory from the environment
		const char* home = getenv("HOME");
		if (home == NULL) {
			// Handle error if HOME is not set
			fprintf(stderr, "HOME environment variable is not set.\n");
			return NULL;
		}

		// Create a buffer to store the full path
		char full_path[512];
		snprintf(full_path, sizeof(full_path), "%s/mpv-build/mpv/video/filter/HopperRender/HopperRenderSettingsApplet.py", home);

		// Use the constructed path in execlp
		execlp("python3", "python3", full_path, (char*)NULL);

		// If execlp returns, there was an error
		perror("execlp");
        exit(EXIT_FAILURE);
    } else if (data->pid < 0) {
        // Fork failed
        perror("fork");
        pthread_exit(NULL);
    } else {
        // Parent process
        close(data->pipe_fd[1]); // Close write end

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

/*
* Processes commands send to the video filter (by mpv).
*
* @param f: The video filter instance
* @param cmd: The command to process
*/
static bool vf_HopperRender_command(struct mp_filter *f, struct mp_filter_command *cmd)
{
	struct priv *priv = f->priv;

	// Speed change event
	if (cmd->type == MP_FILTER_COMMAND_TEXT && priv->m_dPlaybackSpeed != cmd->speed) {
		priv->m_dPlaybackSpeed = cmd->speed;
		priv->m_dSourceFrameTime = 1.0 / (priv->m_dSourceFPS * priv->m_dPlaybackSpeed);
		priv->m_iIntFrameNum = 0;
		priv->m_dScalar = 0.0;
		priv->m_iFrameCounter = 0;
	}

	// If the source is already at or above 60 FPS, we don't need interpolation
	if (priv->m_dSourceFrameTime <= priv->m_dTargetPTS) {
		priv->m_isInterpolationState = NotNeeded;
	// Reset the state either because the fps/speed change now requires interpolation, or we could now be fast enough to interpolate
	} else if (priv->m_isInterpolationState != Deactivated) {
		priv->m_isInterpolationState = Active;
	}
	return true;
}

/*
* Resets the video filter on seek
*
* @param f: The video filter instance
*/
static void vf_HopperRender_reset(struct mp_filter *f)
{
    struct priv *priv = f->priv;
    priv->m_iFrameCounter = 0;
	priv->m_iIntFrameNum = 0;
	priv->m_dScalar = 0.0;
	priv->ofc->m_bOFCTerminate = true;
}

// Filter definition
static const struct mp_filter_info vf_HopperRender_filter = {
    .name = "HopperRender",
    .process = vf_HopperRender_process,
    .priv_size = sizeof(struct priv),
	.reset = vf_HopperRender_reset,
    .destroy = vf_HopperRender_uninit,
	.command = vf_HopperRender_command
};

/*
* Creates the video filter and initializes the private data.
*
* @param parent: The parent filter
* @param options: The filter options
*/
static struct mp_filter *vf_HopperRender_create(struct mp_filter *parent, void *options)
{
	// Create the video filter
    struct mp_filter *f = mp_filter_create(parent, &vf_HopperRender_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }
	struct priv *priv = f->priv;

	// Create the fifo for the AppIndicator widget if it doesn't exist
	char* fifo = "/tmp/hopperrender";
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
	priv->m_foFrameOutput = BlendedFrame;
	priv->m_iNumIterations = NUM_ITERATIONS;
	priv->m_bInitialized = false;
	struct mp_stream_info *info = mp_filter_find_stream_info(f);
    double display_fps = 60.0;
    if (info) {
        if (info->get_display_fps)
            display_fps = info->get_display_fps(info); // Set the target FPS to the display FPS
    }
	priv->m_dTargetPTS = 1.0 / display_fps;

	// Video info
	priv->m_iDimX = 1920;
	priv->m_iDimY = 1080;
	priv->m_miRefImage = calloc(1, sizeof(struct mp_image));

	// Timings
	priv->m_dCurrSourcePTS = 0.0;
	priv->m_dCurrPlaybackPTS = 0.0;
    priv->m_dSourceFPS = 24000.0 / 1001.0; // Default to 23.976 FPS
    priv->m_dPlaybackSpeed = 1.0;
	priv->m_dSourceFrameTime = 1001.0 / 24000.0;
	
	// Optical Flow calculation
	priv->m_cResolutionScalar = INITIAL_RESOLUTION_SCALAR;
	priv->m_dScalar = 0.0;
	priv->m_bOFCBusy = false;
	priv->m_bOFCFailed = false;

	// Frame output
	priv->m_miSWPool = mp_image_pool_new(NULL);
	priv->m_iIntFrameNum = 0;
	priv->m_iFrameCounter = 0;
	priv->m_iNumIntFrames = 1;

	// Performance and activation status
	priv->m_cNumTimesTooSlow = 0;
	priv->m_isInterpolationState = Active;
	priv->m_dOFCCalcDuration = 0.0;

	// Optical Flow calculator
	priv->ofc = calloc(1, sizeof(struct OpticalFlowCalc));

    return f;
}

// Filter entry
const struct mp_user_filter_entry vf_HopperRender = {
    .desc = {
        .description = "Optical-Flow Frame Interpolation",
        .name = "HopperRender",
        .priv_size = sizeof(struct priv)
    },
    .create = vf_HopperRender_create
};