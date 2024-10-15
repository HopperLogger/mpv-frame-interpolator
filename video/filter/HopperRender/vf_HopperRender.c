#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
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
#include <math.h>
#include <sys/stat.h>

#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "video/mp_image_pool.h"
#include "opticalFlowCalc.cuh"

#define PTS60FPS 0.01666666666666666666666666666667 // The target time between frames for 60 FPS video
#define INITIAL_RESOLUTION_SCALAR 2 // The initial resolution scalar (0: Full resolution, 1: Half resolution, 2: Quarter resolution, 3: Eighth resolution, 4: Sixteenth resolution, ...)
#define INITIAL_NUM_STEPS 10 // The initial number of steps executed to find the ideal offset (limits the maximum offset distance per iteration)
#define FRAME_BLUR_KERNEL_SIZE 16 // The size of the blur kernel used to blur the source frames before calculating the optical flow
#define FLOW_BLUR_KERNEL_SIZE 32 // The size of the blur kernel used to blur the offset calculated by the optical flow
#define NUM_ITERATIONS 0 // Number of iterations to use in the optical flow calculation (0: As many as possible)
#define SPREAD_CORES 0 // Whether or not to spread the threads over the available cores (0: Disabled, 1: SPREAD, 2: SINGLE)
#define TEST_MODE 0 // Enables or disables automatic settings change to allow accurate performance testing (0: Disabled, 1: Enabled)
#define AUTO_FRAME_SCALE 0 // Whether to automatically reduce/increase the calculation resolution depending on performance (0: Disabled, 1: Enabled)
#define AUTO_STEPS_ADJUST 0 // Whether to automatically reduce/increase the number of calculation steps depending on performance (0: Disabled, 1: Enabled)
#define LOG_PERFORMANCE 0 // Whether or not to print debug messages regarding calculation performance (0: Disabled, 1: Enabled)
#define MIN_NUM_STEPS 4 // The minimum number of calculation steps (if below this, resolution will be decreased or calculation disabled)
#define MAX_NUM_STEPS 15 // The maximum number of calculation steps (if reached, resolution will be increased or steps will be kept at this number)
#define MAX_NUM_BUFFERED_IMG 50 // The maximum number of buffered images allowed to be in the image pool

typedef struct {
    pid_t pid;
    int pipe_fd[2];
} ThreadData;

typedef enum FrameOutput {
    WarpedFrame12,
    WarpedFrame21,
	BlendedFrame,
	HSprivlow,
	GreyFlow,
	BlurredFrames,
	SideBySide1,
	SideBySide2
} FrameOutput;

typedef enum InterpolationState {
    Deactivated,
	NotNeeded,
	Active,
	TooSlow
} InterpolationState;

struct priv {
    // Thread data
    ThreadData m_tdAppIndicatorThreadData; // Data for the AppIndicator thread used to communicate with the status widget
	int m_iAppIndicatorFileDesc; // The file descriptor for the AppIndicator status widget
	pthread_t m_ptOFCThreadID; // The thread ID of the optical flow calculation thread

    // Settings
	FrameOutput m_foFrameOutput; // What frame output to use
	unsigned int m_iNumIterations; // Number of iterations to use in the optical flow calculation (0: As many as possible)
	unsigned int m_iFrameBlurKernelSize; // The size of the blur kernel used to blur the source frames before calculating the optical flow
	unsigned int m_iFlowBlurKernelSize; // The size of the blur kernel used to blur the offset calculated by the optical flow
	bool m_bInitialized; // Whether or not the filter has been initialized
	double m_dTargetPTS; // The target presentation time stamp (PTS) of the video
	
	// Video info
	unsigned int m_iDimX; // The width of the frame
	unsigned int m_iDimY; // The height of the frame
	bool m_bIsHDR; // Whether or not the video is HDR
	enum mp_imgfmt m_fmt; // The format of the video
	struct mp_image *m_miRefImage; // The reference image used for the optical flow calculation

	// Timings
	double m_dCurrSourcePTS; // The current presentation time stamp (PTS) of the video
	double m_dCurrPlaybackPTS; // The current presentation time stamp (PTS) of the actual playback
	double m_dSourceFPS; // The fps of the source video
    double m_dPlaybackSpeed; // The speed of the playback
	double m_dSourceFrameTime; // The current time between source frames (1 / m_dSourceFPS)
	
	// Optical flow calculation
	struct OpticalFlowCalc* ofc; // Optical flow calculator struct
	unsigned char m_cResolutionScalar; // Determines which resolution scalar will be used for the optical flow calculation (0: Full resolution, 1: Half resolution, 2: Quarter resolution, 3: Eighth resolution, 4: Sixteenth resolution, ...)
	unsigned int m_iNumSteps; // Number of steps executed to find the ideal offset (limits the maximum offset distance per iteration)

	// Frame output
	struct mp_image_pool *m_miSWPool; // The software image pool used to store the source frames
	struct mp_image **m_miHWPool; // The hardware image pool used to store the source frames
	size_t m_iHWPoolSize; // The number of frames in the hardware image pool
	int m_iIntFrameNum; // The current interpolated frame number
	int m_iFrameCounter; // Frame counter (relative! i.e. number of source frames received since last seek or playback start)
	int m_iNumIntFrames; // Number of interpolated frames for every source frame

	// Performance and activation status
	unsigned char m_cNumTimesTooSlow; // The number of times the interpolation has been too slow
	InterpolationState m_isInterpolationState; // The state of the filter (0: Deactivated, 1: Not Needed, 2: Active, 3: Too Slow)
	struct timeval m_teOFCCalcStart; // The start time of the optical flow calculation
	struct timeval m_teOFCCalcEnd; // The end time of the optical flow calculation
	struct timeval m_teWarpCalcStart[100]; // The start times of the warp calculations
	struct timeval m_teWarpCalcEnd[100]; // The end times of the warp calculations
	double m_dOFCCalcDuration; // The duration of the optical flow calculation
};

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

/*
* Applies the new resolution scalar to the optical flow calculator
*
* @param priv: The video filter private data
* @param newResolutionScalar: The new resolution scalar to apply
*/
static void vf_HopperRender_adjust_frame_scalar(struct priv *priv, const unsigned char newResolutionScalar) {
	// Here we just adjust all the variables that are affected by the new resolution scalar
	priv->m_cResolutionScalar = newResolutionScalar;
	priv->ofc->m_iFlowBlurKernelSize = priv->m_iFlowBlurKernelSize >> priv->m_cResolutionScalar;
	if (priv->m_isInterpolationState == TooSlow) priv->m_isInterpolationState = Active;
	//priv->m_iNumSteps = MIN_NUM_STEPS;
	priv->ofc->m_cResolutionScalar = priv->m_cResolutionScalar;
	priv->ofc->m_iLowDimX = priv->m_iDimX >> priv->m_cResolutionScalar;
	priv->ofc->m_iLowDimY = priv->m_iDimY >> priv->m_cResolutionScalar;
	priv->ofc->m_iDirectionIdxOffset = priv->ofc->m_iNumLayers * priv->ofc->m_iLowDimY * priv->ofc->m_iLowDimX;
	priv->ofc->m_iLayerIdxOffset = priv->ofc->m_iLowDimY * priv->ofc->m_iLowDimX;
	priv->ofc->m_lowGrid32x32x1.x = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimX) / 32.0), 1.0));
	priv->ofc->m_lowGrid32x32x1.y = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimY) / 32.0), 1.0));
	priv->ofc->m_lowGrid16x16x5.x = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimX) / 16.0), 1.0));
	priv->ofc->m_lowGrid16x16x5.y = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimY) / 16.0), 1.0));
	priv->ofc->m_lowGrid16x16x4.x = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimX) / 16.0), 1.0));
	priv->ofc->m_lowGrid16x16x4.y = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimY) / 16.0), 1.0));
	priv->ofc->m_lowGrid16x16x1.x = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimX) / 16.0), 1.0));
	priv->ofc->m_lowGrid16x16x1.y = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimY) / 16.0), 1.0));
	priv->ofc->m_lowGrid8x8x5.x = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimX) / 8.0), 1.0));
	priv->ofc->m_lowGrid8x8x5.y = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimY) / 8.0), 1.0));
	priv->ofc->m_lowGrid8x8x1.x = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimX) / 8.0), 1.0));
	priv->ofc->m_lowGrid8x8x1.y = (int)(fmax(ceil((double)(priv->ofc->m_iLowDimY) / 8.0), 1.0));
}

/*
* Processes the commands received from the AppIndicator
*
* @param priv: The video filter private data
* @param code: The received command code
*/
void vf_HopperRender_process_AppIndicator_command(struct priv *priv, int code) {
	switch (code) {
		// **** Activation Status ****
		case 0:
			if (priv->m_isInterpolationState != Deactivated) {
				priv->m_isInterpolationState = Deactivated;
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
			priv->m_foFrameOutput = HSprivlow;
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
		// **** Shaders ****
		case 9:
			priv->ofc->m_fBlackLevel = 0.0f;
			priv->ofc->m_fWhiteLevel = 220.0f;
			if (priv->m_bIsHDR && priv->m_fmt == IMGFMT_CUDA) {
				priv->ofc->m_fBlackLevel *= 256.0f;
				priv->ofc->m_fWhiteLevel *= 256.0f;
			} else if (priv->m_bIsHDR) {
				priv->ofc->m_fBlackLevel *= 4.0f;
				priv->ofc->m_fWhiteLevel *= 4.0f;
			}
			break;
		case 10:
			priv->ofc->m_fBlackLevel = 16.0f;
			priv->ofc->m_fWhiteLevel = 219.0f;
			if (priv->m_bIsHDR && priv->m_fmt == IMGFMT_CUDA) {
				priv->ofc->m_fBlackLevel *= 256.0f;
				priv->ofc->m_fWhiteLevel *= 256.0f;
			} else if (priv->m_bIsHDR) {
				priv->ofc->m_fBlackLevel *= 4.0f;
				priv->ofc->m_fWhiteLevel *= 4.0f;
			}
			break;
		case 11:
			priv->ofc->m_fBlackLevel = 10.0f;
			priv->ofc->m_fWhiteLevel = 225.0f;
			if (priv->m_bIsHDR && priv->m_fmt == IMGFMT_CUDA) {
				priv->ofc->m_fBlackLevel *= 256.0f;
				priv->ofc->m_fWhiteLevel *= 256.0f;
			} else if (priv->m_bIsHDR) {
				priv->ofc->m_fBlackLevel *= 4.0f;
				priv->ofc->m_fWhiteLevel *= 4.0f;
			}
			break;
		case 12:
			priv->ofc->m_fBlackLevel = 0.0f;
			priv->ofc->m_fWhiteLevel = 255.0f;
			if (priv->m_bIsHDR && priv->m_fmt == IMGFMT_CUDA) {
				priv->ofc->m_fBlackLevel *= 256.0f;
				priv->ofc->m_fWhiteLevel *= 256.0f;
			} else if (priv->m_bIsHDR) {
				priv->ofc->m_fBlackLevel *= 4.0f;
				priv->ofc->m_fWhiteLevel *= 4.0f;
			}
			break;
		case 13:
			vf_HopperRender_adjust_frame_scalar(priv, 0);
			break;
		case 14:
			vf_HopperRender_adjust_frame_scalar(priv, 1);
			break;
		case 15:
			vf_HopperRender_adjust_frame_scalar(priv, 2);
			break;
		case 16:
			vf_HopperRender_adjust_frame_scalar(priv, 3);
			break;
		case 17:
			vf_HopperRender_adjust_frame_scalar(priv, 4);
			break;
		case 18:
			vf_HopperRender_adjust_frame_scalar(priv, 5);
			break;
	}
}

/*
* The optical flow calculation thread
*
* @param arg: The video filter private data (struct priv)
*/
void *vf_HopperRender_optical_flow_calc_thread(void *arg) {
	// Ensure the thread runs on the correct core
	if (SPREAD_CORES) {
		cpu_set_t cpuset;
		int worker_core = 0;
		CPU_ZERO(&cpuset);
		CPU_SET(worker_core, &cpuset);
		pthread_t thread = pthread_self();
		pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
	}
	printf("[HopperRender] OFC thread running on CPU %d\n", sched_getcpu());

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
			// Calculate the optical flow (frame 1 to frame 2)
			if (priv->m_foFrameOutput != BlurredFrames) {
				priv->ofc->calculateOpticalFlow(priv->ofc, priv->m_iNumIterations, priv->m_iNumSteps);
			}

			// Flip the flow array to get the flow from frame 2 to frame 1
			if (priv->m_foFrameOutput == WarpedFrame21 || 
				priv->m_foFrameOutput == BlendedFrame || 
				priv->m_foFrameOutput == SideBySide1 || 
				priv->m_foFrameOutput == SideBySide2) {
				priv->ofc->flipFlow(priv->ofc);
			}

			// Blur the flow arrays
			if (priv->m_foFrameOutput != BlurredFrames) {
				priv->ofc->blurFlowArrays(priv->ofc);
			}

			gettimeofday(&priv->m_teOFCCalcEnd, NULL);
			priv->m_dOFCCalcDuration = ((priv->m_teOFCCalcEnd.tv_sec - priv->m_teOFCCalcStart.tv_sec) * 1000.0) + ((priv->m_teOFCCalcEnd.tv_usec - priv->m_teOFCCalcStart.tv_usec) / 1000.0);

        } else if (sig == SIGTERM) {
			pthread_exit(NULL);
            break;
        }
    }
}

// Optical flow calculation initialization (declared in CUDA C++ file)
extern void initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int dimY, const int dimX, const int realDimX, unsigned char resolutionScalar, unsigned int flowBlurKernelSize, bool isHDR, int fmt);

/*
* Initializes the video filter.
*
* @param priv: The video filter private data
* @param dimY: The height of the video
* @param dimX: The stride width of the video
* @param realDimX: The real width of the video (not the stride width!)
* @param fmt: The output format of the video
*
* @return: The result of the configuration
*/
static void vf_HopperRender_init(struct priv *priv, int dimY, int dimX, int realDimX, enum mp_imgfmt fmt)
{
	priv->m_iDimX = dimX;
	priv->m_iDimY = dimY;
	priv->m_fmt = fmt;

	// Initialize the optical flow calculator
	initOpticalFlowCalc(priv->ofc, dimY, dimX, realDimX, priv->m_cResolutionScalar, priv->m_iFlowBlurKernelSize >> priv->m_cResolutionScalar, priv->m_bIsHDR, (int)fmt);
	
	// Create the optical flow calc thread
	int ret = pthread_create(&priv->m_ptOFCThreadID, NULL, vf_HopperRender_optical_flow_calc_thread, priv);
	if (ret != 0) {
		perror("pthread_create");
		exit(EXIT_FAILURE);
	}

	// Send the resolution to the AppIndicator status widget
	char buffer2[512];
	memset(buffer2, 0, sizeof(buffer2));
	for (int i = 0; i < 6; i++) {
		snprintf(buffer2, sizeof(buffer2), "RES%d %dx%d", i, realDimX >> i, dimY >> i);
		if (write(priv->m_iAppIndicatorFileDesc, buffer2, sizeof(buffer2)) == -1) {
			perror("write");
			close(priv->m_iAppIndicatorFileDesc);
			exit(EXIT_FAILURE);
		}
	}
	
	// Initialize the needed image pool
	if (IMGFMT_IS_HWACCEL(fmt)) {
		priv->m_miHWPool = calloc(MAX_NUM_BUFFERED_IMG, sizeof(struct mp_image*));
	} else {
		priv->m_miSWPool = mp_image_pool_new(NULL);
	}
}

/*
* Processes commands send to the video filter (by mpv).
*
* @param f: The video filter instance
* @param cmd: The command to process
*/
static void vf_HopperRender_command(struct mp_filter *f, struct mp_filter_command *cmd) {
	struct priv *priv = f->priv;

	// Speed change event
	if (cmd->type == MP_FILTER_COMMAND_TEXT) {
		priv->m_dPlaybackSpeed = cmd->speed;
	}
	
	// If the source is already at or above 60 FPS, we don't need interpolation
	if (priv->m_dSourceFrameTime <= priv->m_dTargetPTS) {
		priv->m_isInterpolationState = NotNeeded;
	// Reset the state either because the fps/speed change now requires interpolation, or we could now be fast enough to interpolate
	} else if (priv->m_isInterpolationState != Deactivated) {
		priv->m_isInterpolationState = Active;
	}
}

/*
* Sends the current info to the AppIndicator status widget
*
* @param priv: The video filter private data
*/
static void vf_HopperRender_update_AppIndicator_widget(struct priv *priv, double warpCalcDurations[100], double currTotalWarpDuration) {
	char buffer2[512];
	memset(buffer2, 0, sizeof(buffer2));
    int offset = snprintf(buffer2, sizeof(buffer2), 
                          "Num Steps: %d\nCalc Res: %dx%d\nTarget Time: %06.2f ms (%.1f fps)\nFrame Time: %06.2f ms (%.3f fps | %.2fx)\nOfc: %06.2f ms (%.0f fps)\nWarp Time: %06.2f ms (%.0f fps)",
                          priv->m_iNumSteps, priv->m_iDimX >> priv->m_cResolutionScalar, priv->m_iDimY >> priv->m_cResolutionScalar, priv->m_dTargetPTS * 1000.0, 1.0 / priv->m_dTargetPTS,
						  priv->m_dSourceFrameTime * 1000.0, 1.0 / priv->m_dSourceFrameTime, priv->m_dPlaybackSpeed, priv->m_dOFCCalcDuration, 1.0 / priv->m_dOFCCalcDuration * 1000.0, currTotalWarpDuration, 1.0 / currTotalWarpDuration * 1000.0);

    for (int i = 0; i < 10; i++) {
		if (i < min(priv->m_iNumIntFrames, 10)) {
        	offset += snprintf(buffer2 + offset, sizeof(buffer2) - offset, "\nWarp%d: %06.2f ms", i, warpCalcDurations[i]);
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
* Adjust optical flow calculation settings for optimal performance and quality
*
* @param priv: The video filter private data
*/
static void vf_HopperRender_auto_adjust_settings(struct priv *priv) {
	// Calculate the calculation durations
	double warpCalcDurations[100];
	double currTotalWarpDuration = 0.0;
	double currTotalCalcDuration = priv->m_dOFCCalcDuration;
	for (int i = 0; i < min(priv->m_iNumIntFrames, 100); i++) {
		warpCalcDurations[i] = ((priv->m_teWarpCalcEnd[i].tv_sec - priv->m_teWarpCalcStart[i].tv_sec) * 1000.0) + ((priv->m_teWarpCalcEnd[i].tv_usec - priv->m_teWarpCalcStart[i].tv_usec) / 1000.0);
		currTotalWarpDuration += warpCalcDurations[i];
	}
	if (currTotalWarpDuration > priv->m_dOFCCalcDuration) currTotalCalcDuration = currTotalWarpDuration;
	
	// Send the stats to the AppIndicator status widget
	vf_HopperRender_update_AppIndicator_widget(priv, warpCalcDurations, currTotalWarpDuration);

	// Get the time we have in between source frames (WE NEED TO STAY BELOW THIS!)
	double dSourceFrameTimeMS = priv->m_dSourceFrameTime * 1000.0;
	
	/*
	* Calculation took too long
	*/
	if ((currTotalCalcDuration + currTotalCalcDuration * 0.7) > dSourceFrameTimeMS) {
		//mp_msg(MSGT_privILTER, MSGL_INFO, 
		//	"Calculation took too long %.3f ms AVG SFT: %.3f NumSteps: %d",
        //     priv->m_dCurrCalcDuration, dSourceFrameTimeMS, priv->m_iNumSteps);
		if (LOG_PERFORMANCE) printf("[HopperRender] Calculation took too long %.3f ms AVG SFT: %.3f NumSteps: %d RES SCALAR: %d\n", currTotalCalcDuration, dSourceFrameTimeMS, priv->m_iNumSteps, priv->m_cResolutionScalar);

		// Decrease the number of steps to reduce calculation time
		if (AUTO_STEPS_ADJUST && priv->m_iNumSteps > MIN_NUM_STEPS) {
			priv->m_iNumSteps -= 1;
			return;
		}

		// We can't reduce the number of steps any further, so we reduce the resolution divider instead
		if (AUTO_FRAME_SCALE && priv->m_cResolutionScalar < 5) {
			// To avoid unnecessary adjustments, we only adjust the resolution divider if we have been too slow for a while
			priv->m_cNumTimesTooSlow += 1;
			if (priv->m_cNumTimesTooSlow > 1) {
				priv->m_cNumTimesTooSlow = 0;
				vf_HopperRender_adjust_frame_scalar(priv, priv->m_cResolutionScalar + 1);
			}
			return;
		}

		// Disable Interpolation if we are too slow
		if ((AUTO_FRAME_SCALE || AUTO_STEPS_ADJUST) && ((currTotalCalcDuration + currTotalCalcDuration * 0.05) > dSourceFrameTimeMS)) priv->m_isInterpolationState = TooSlow;

	/*
	* We have left over capacity
	*/
	} else if ((currTotalCalcDuration + currTotalCalcDuration * 0.9) < dSourceFrameTimeMS) {
		//mp_msg(MSGT_privILTER, MSGL_INFO, 
		//	"Calculation has capacity %.3f ms AVG SFT: %.3f NumSteps: %d",
        //     priv->m_dCurrCalcDuration, dSourceFrameTimeMS, priv->m_iNumSteps);
		if (LOG_PERFORMANCE) printf("[HopperRender] Calculation has capacity %.3f ms AVG SFT: %.3f NumSteps: %d RES SCALAR: %d\n", currTotalCalcDuration, dSourceFrameTimeMS, priv->m_iNumSteps, priv->m_cResolutionScalar);

		// Increase the frame scalar if we have enough leftover capacity
		if (AUTO_FRAME_SCALE && priv->m_cResolutionScalar > 0 && priv->m_iNumSteps >= MAX_NUM_STEPS) {
			priv->m_cNumTimesTooSlow = 0;
			vf_HopperRender_adjust_frame_scalar(priv, priv->m_cResolutionScalar - 1);
			priv->m_iNumSteps = MIN_NUM_STEPS;
		} else if (AUTO_STEPS_ADJUST) {
			priv->m_iNumSteps = fmin(priv->m_iNumSteps + 1, MAX_NUM_STEPS); // Increase the number of steps to use the leftover capacity
		}

	/*
	* Calculation takes as long as it should
	*/
	} else {
		//mp_msg(MSGT_privILTER, MSGL_INFO, 
		//	"Calculation took %.3f ms AVG SFT: %.3f NumSteps: %d",
        //     priv->m_dCurrCalcDuration, dSourceFrameTimeMS, priv->m_iNumSteps);
		if (LOG_PERFORMANCE) printf("[HopperRender] Calculation took %.3f ms AVG SFT: %.3f NumSteps: %d RES SCALAR: %d\n", currTotalCalcDuration, dSourceFrameTimeMS, priv->m_iNumSteps, priv->m_cResolutionScalar);
	}
}

/*
* Coordinates the optical flow calc thread and generates the interpolated frames
*
* @param priv: The video filter private data
* @param planes: The planes of the output frame
* @param fScalar: The scalar used to interpolate the frames
*/
void vf_HopperRender_interpolate_frame(struct priv *priv, unsigned char** planes, float fScalar)
{
	if (priv->m_iIntFrameNum == 0) {
		// Swap the blurred offset arrays
		int* temp0 = priv->ofc->m_blurredOffsetArray12[0];
		priv->ofc->m_blurredOffsetArray12[0] = priv->ofc->m_blurredOffsetArray12[1];
		priv->ofc->m_blurredOffsetArray12[1] = temp0;

		temp0 = priv->ofc->m_blurredOffsetArray21[0];
		priv->ofc->m_blurredOffsetArray21[0] = priv->ofc->m_blurredOffsetArray21[1];
		priv->ofc->m_blurredOffsetArray21[1] = temp0;
		
		// Tell the thread to calculate the optical flow
		pthread_kill(priv->m_ptOFCThreadID, SIGUSR1);
	}
	
	if (priv->m_iIntFrameNum < 100) gettimeofday(&priv->m_teWarpCalcStart[priv->m_iIntFrameNum], NULL);
	
	// Warp frames
	if (priv->m_foFrameOutput != HSprivlow && 
		priv->m_foFrameOutput != BlurredFrames) {
		priv->ofc->warpFrames(priv->ofc, fScalar, priv->m_foFrameOutput);
	}
	
	// Blend the frames together
	if (priv->m_foFrameOutput == BlendedFrame || 
		priv->m_foFrameOutput == SideBySide1) {
		priv->ofc->blendFrames(priv->ofc, fScalar);
	}
	
	// Draw the flow as an HSV image
	if (priv->m_foFrameOutput == HSprivlow) {
		priv->ofc->drawFlowAsHSV(priv->ofc, 0.5f);
	// Draw the flow as a greyscale image
	} else if (priv->m_foFrameOutput == GreyFlow) {
		priv->ofc->drawFlowAsGreyscale(priv->ofc);
	// Show side by side comparison
	} else if (priv->m_foFrameOutput == SideBySide1) {
		priv->ofc->insertFrame(priv->ofc);
	} else if (priv->m_foFrameOutput == SideBySide2) {
	    priv->ofc->sideBySideFrame(priv->ofc, fScalar, priv->m_iFrameCounter);
	}
	
	// Download the result to the output buffer
	priv->ofc->downloadFrame(priv->ofc, planes);
	if (priv->m_iIntFrameNum < 100) gettimeofday(&priv->m_teWarpCalcEnd[priv->m_iIntFrameNum], NULL);

	// Adjust the settings to process everything fast enough
	if (priv->m_foFrameOutput != BlurredFrames && 
		priv->m_iIntFrameNum == priv->m_iNumIntFrames - 1 && 
		!TEST_MODE) {
		vf_HopperRender_auto_adjust_settings(priv);
	}
}

/*
* Helper function to handle AppIndicator commands
*
* @param priv: The video filter private data
*/
static void vf_HopperRender_redirect_AppIndicator_command(struct priv *priv)
{
    char buffer[256];
    ssize_t bytesRead;
    while ((bytesRead = read(priv->m_tdAppIndicatorThreadData.pipe_fd[0], buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytesRead] = '\0';
        if (buffer[0] != '\n') {
            vf_HopperRender_process_AppIndicator_command(priv, atoi(buffer));
        }
    }
}

/*
* Checks if the hw image pool has numFrames images and if not, creates new images.
* After this function is called, the hw image pool will contain numFrames images.
*
* @param f: The video filter instance
* @param numFrames: The number of images that should be in the pool
*/
void vf_HopperRender_check_hwpool_size(struct mp_filter *f, int numFrames)
{
	struct priv *priv = f->priv;

	// If we are processing software images, we don't need to fill our custom hw pool
	if (!IMGFMT_IS_HWACCEL(priv->m_fmt))
		return;

	// Check if the image pool already has the correct number of images
	if (priv->m_iHWPoolSize >= numFrames)
		return;

	printf("[HopperRender] Adding %d images to the hw pool\n", numFrames - priv->m_iHWPoolSize);

	for (int i = priv->m_iHWPoolSize; i < numFrames; i++) {
		AVFrame *av_frame = av_frame_alloc();
		if (!av_frame)
			return NULL;
		if (av_hwframe_get_buffer(priv->m_miRefImage->hwctx, av_frame, 0) < 0) {
			av_frame_free(&av_frame);
			return NULL;
		}
		struct mp_image *dst = mp_image_from_av_frame(av_frame);
		av_frame_free(&av_frame);
		if (!dst)
			return NULL;

		if (dst->w < priv->m_miRefImage->w || dst->h < priv->m_miRefImage->h) {
			talloc_free(dst);
			return NULL;
		}

		mp_image_set_size(dst, priv->m_miRefImage->w, priv->m_miRefImage->h);

		mp_image_copy_attributes(dst, priv->m_miRefImage);
		
		priv->m_miHWPool[i] = dst;
	}
	priv->m_iHWPoolSize = numFrames;
}

/*
* Yields an image from the currently used MPI pool.
*
* @param f: The video filter instance
* @param frameNum: The frame number to yield
*/
struct mp_image *vf_HopperRender_get_image(struct mp_filter *f, int frameNum) {
	struct priv *priv = f->priv;

	// If we are processing software images, we can just use the software image pool
	if (!IMGFMT_IS_HWACCEL(priv->m_fmt)) {
		struct mp_image *mpi = mp_image_pool_get(priv->m_miSWPool, priv->m_fmt, priv->m_iDimX, priv->m_iDimY);
		mp_image_copy_attributes(mpi, priv->m_miRefImage);
		return mpi;
	}

	// If we are processing hardware images, we return a new ref to one of the images in the hardware pool
	if (frameNum < 0 || frameNum >= priv->m_iHWPoolSize) {
		printf("[HopperRender] Invalid frame number\n");
		mp_filter_internal_mark_failed(f);
		return NULL;
	}

	// TEMP BEGIN
	AVFrame *av_frame = av_frame_alloc();
	if (!av_frame)
		return NULL;
	if (av_hwframe_get_buffer(priv->m_miRefImage->hwctx, av_frame, 0) < 0) {
		av_frame_free(&av_frame);
		return NULL;
	}
	struct mp_image *dst = mp_image_from_av_frame(av_frame);
	av_frame_free(&av_frame);
	if (!dst)
		return NULL;

	if (dst->w < priv->m_miRefImage->w || dst->h < priv->m_miRefImage->h) {
		talloc_free(dst);
		return NULL;
	}

	mp_image_set_size(dst, priv->m_miRefImage->w, priv->m_miRefImage->h);

	mp_image_copy_attributes(dst, priv->m_miRefImage);

	return dst;
	// TEMP END

	return mp_image_new_ref(priv->m_miHWPool[frameNum]);
}

/*
* Delivers the intermediate frames to the output pin.
*
* @param f: The video filter instance
* @param fScalar: The scalar used to interpolate the frames
*/
static void vf_HopperRender_process_intermediate_frame(struct mp_filter *f, float fScalar)
{
	struct priv *priv = f->priv;

	struct mp_image *img = vf_HopperRender_get_image(f, priv->m_iIntFrameNum - 1);
	struct mp_frame frame = {.type = MP_FRAME_VIDEO, .data = img};
	
    // Generate the interpolated frame
    vf_HopperRender_interpolate_frame(priv, img->planes, fScalar);
	
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
    vf_HopperRender_redirect_AppIndicator_command(priv);

    // Read the new source frame
    struct mp_frame frame = mp_pin_out_read(f->ppins[0]);
    struct mp_image *img = frame.data;
	
	// Detect if the frame is an end of frame
	if (mp_frame_is_signaling(frame)) {
        mp_pin_in_write(f->ppins[1], frame);
        return;
    }

    // Initialize the filter if needed
    if (!priv->m_bInitialized) {
		priv->m_bIsHDR = (img->imgfmt == 1118) || img->params.hw_subfmt == IMGFMT_P010;
		priv->m_miRefImage = mp_image_new_ref(img);
        vf_HopperRender_init(priv, img->h, img->stride[0] / (priv->m_bIsHDR ? 2 : 1), img->w, img->imgfmt);
        priv->m_bInitialized = true;
    }

    // Update timestamps and source information
	priv->m_iFrameCounter += 1;
	priv->m_dCurrSourcePTS = img->pts;
    if (priv->m_iFrameCounter <= 4) {
        priv->m_dCurrPlaybackPTS = img->pts; // The first four frames we take the original PTS (see output: 1.0, 2.0, 3.0, 4.0, 4.1, ...)
    } else {
        priv->m_dCurrPlaybackPTS += priv->m_dTargetPTS * priv->m_dPlaybackSpeed; // The rest of the frames we increase in 60fps steps
    }
	img->pts = priv->m_dCurrPlaybackPTS;
    priv->m_dSourceFPS = img->nominal_fps;
    priv->m_dSourceFrameTime = 1.0 / (priv->m_dSourceFPS * priv->m_dPlaybackSpeed);
    priv->m_iNumIntFrames = fmax(ceil((priv->m_dSourceFrameTime + priv->m_dCurrSourcePTS - priv->m_dCurrPlaybackPTS) / priv->m_dTargetPTS), 1.0);
	vf_HopperRender_check_hwpool_size(f, priv->m_iNumIntFrames - 1);

    // Update the GPU arrays
    gettimeofday(&priv->m_teOFCCalcStart, NULL);
    priv->ofc->updateFrame(priv->ofc, img->planes, priv->m_iFrameBlurKernelSize, priv->m_foFrameOutput == BlurredFrames);
	
	// Don't interpolate the first three frames (as we need three frames in the buffer to interpolate)
    if (priv->m_isInterpolationState == Active && (priv->m_iFrameCounter > 3 || priv->m_foFrameOutput == SideBySide2)) {
		vf_HopperRender_interpolate_frame(priv, img->planes, 0.0f);
        priv->m_iIntFrameNum = 1;
        mp_filter_internal_mark_progress(f);
    } else {
        priv->ofc->processFrame(priv->ofc, img->planes, priv->m_iFrameCounter == 1);
    }
	
    // Deliver the source frame
    mp_pin_in_write(f->ppins[1], frame);
}

/*
* Main filter process function. Called on every new source frame.
*
* @param f: The video filter instance
*/
static void vf_HopperRender_process(struct mp_filter *f)
{
    struct priv *priv = f->priv;

    // Calculate the scalar for the interpolation
    float fScalar = fmax(fmin((float)(priv->m_iIntFrameNum) / (float)(priv->m_iNumIntFrames), 1.0f), 0.0f);

    // Process intermediate frames if needed
    if (priv->m_iIntFrameNum > 0 && mp_pin_in_needs_data(f->ppins[1])) {
        vf_HopperRender_process_intermediate_frame(f, fScalar);
        return;
    }

    // Process a new source frame if available
    if (priv->m_iIntFrameNum == 0 && mp_pin_can_transfer_data(f->ppins[1], f->ppins[0])) {
        vf_HopperRender_process_new_source_frame(f);
    }

	/* printf("[HopperRender] FN: %d - IN: %d - PTS: %f - PPTS: %f - Delta: %f\n",
           priv->m_iFrameCounter, priv->m_iIntFrameNum, 
           priv->m_dCurrSourcePTS + priv->m_dTargetPTS * priv->m_dPlaybackSpeed * priv->m_iIntFrameNum,
           priv->m_dCurrPlaybackPTS, 
           fabs((priv->m_dCurrSourcePTS + priv->m_dTargetPTS * priv->m_dPlaybackSpeed * priv->m_iIntFrameNum) - priv->m_dCurrPlaybackPTS)); */
}

/*
* Start the AppIndicator widget
*
* @param arg: The thread data
*/
void *vf_HopperRender_start_AppIndicator_script(void *arg) {
	ThreadData *data = (ThreadData *)arg;
    if (pipe(data->pipe_fd) == -1) {
        perror("pipe");
        pthread_exit(NULL);
    }

    data->pid = fork();
    if (data->pid == 0) {
        // Child process
		if (SPREAD_CORES) {
			cpu_set_t cpuset;
			int worker_core = SPREAD_CORES ? 2 : 0;
			CPU_ZERO(&cpuset);
			CPU_SET(worker_core, &cpuset);
			pthread_t thread = pthread_self();
			pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
		}
		printf("[HopperRender] AppIndicator thread running on CPU %d\n", sched_getcpu());
        close(data->pipe_fd[0]); // Close read end
        dup2(data->pipe_fd[1], STDOUT_FILENO); // Redirect stdout to pipe
        dup2(data->pipe_fd[1], STDERR_FILENO); // Redirect stderr to pipe
        close(data->pipe_fd[1]); // Close original write end

        // Get the home directory from the environment
		const char *home = getenv("HOME");
		if (home == NULL) {
			// Handle error if HOME is not set
			fprintf(stderr, "HOME environment variable is not set.\n");
			return 1;
		}

		// Create a buffer to store the full path
		char full_path[512]; // Adjust buffer size accordingly
		snprintf(full_path, sizeof(full_path), "%s/mpv-build/mpv/video/filter/HopperRender/HopperRenderSettingsApplet.py", home);

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
* Terminates the AppIndicator widget
*/
void vf_HopperRender_terminate_AppIndicator_script(ThreadData *data) {
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
* @param priv: The video filter instance
*/
static void vf_HopperRender_uninit(struct mp_filter *f) {
    struct priv *priv = f->priv;
	mp_image_unrefp(&priv->m_miRefImage);
	for (int i = 0; i < priv->m_iHWPoolSize; i++) {
		if (priv->m_miHWPool[i])
			mp_image_unrefp(&priv->m_miHWPool[i]);
	}
	close(priv->m_iAppIndicatorFileDesc);
	if (priv->m_bInitialized) {
		pthread_kill(priv->m_ptOFCThreadID, SIGTERM);
    	pthread_join(priv->m_ptOFCThreadID, NULL);
		priv->ofc->free(priv->ofc);
	}
	free(priv->ofc);
	vf_HopperRender_terminate_AppIndicator_script(&priv->m_tdAppIndicatorThreadData);
    //pthread_join(priv->thread, NULL);
	//free(priv);
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
* Creates the video filter and intializes the private data.
*
* @param parent: The parent filter
* @param options: The filter options
*/
static struct mp_filter *vf_HopperRender_create(struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &vf_HopperRender_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *priv = f->priv;
    priv->ofc = calloc(1, sizeof(struct OpticalFlowCalc));

	// Thread data
	priv->m_iAppIndicatorFileDesc = -1;
	priv->m_ptOFCThreadID = 0;

    // Settings
	priv->m_foFrameOutput = BlendedFrame;
	priv->m_iNumIterations = NUM_ITERATIONS;
	priv->m_iFrameBlurKernelSize = FRAME_BLUR_KERNEL_SIZE;
	priv->m_iFlowBlurKernelSize = FLOW_BLUR_KERNEL_SIZE;
	priv->m_bInitialized = false;
	struct mp_stream_info *info = mp_filter_find_stream_info(f);
    double display_fps = 60.0;
    if (info) {
        if (info->get_display_fps)
            display_fps = info->get_display_fps(info);
    }
	priv->m_dTargetPTS = 1.0 / display_fps;

	// Video info
	priv->m_iDimX = 1920;
	priv->m_iDimY = 1080;
	priv->m_bIsHDR = false;
	priv->m_fmt = IMGFMT_420P;
	priv->m_miRefImage = calloc(1, sizeof(struct mp_image));

	// Timings
	priv->m_dCurrSourcePTS = 0.0;
	priv->m_dCurrPlaybackPTS = 0.0;
    priv->m_dSourceFPS = 24000.0 / 1001.0; // Default to 23.976 FPS
    priv->m_dPlaybackSpeed = 1.0;
	priv->m_dSourceFrameTime = 1001.0 / 24000.0;
	
	// Optical Flow calculation
	priv->m_cResolutionScalar = INITIAL_RESOLUTION_SCALAR;
	priv->m_iNumSteps = INITIAL_NUM_STEPS;

	// Frame output
	priv->m_miSWPool = NULL;
	priv->m_miHWPool = NULL;
	priv->m_iHWPoolSize = 0;
	priv->m_iIntFrameNum = 0;
	priv->m_iFrameCounter = 0;
	priv->m_iNumIntFrames = 1;

	// Performance and activation status
	priv->m_cNumTimesTooSlow = 0;
	priv->m_isInterpolationState = Active;
	priv->m_dOFCCalcDuration = 0.0;

	// ######################################
	// #### LAUNCH AppIndicator WIDGET ######
	// ######################################
	pthread_t thread;
	char *fifo = "/tmp/hopperrender";
	if (access(fifo, F_OK) == -1) {
        if (mkfifo(fifo, 0666) != 0) {
            perror("Failed to create pipe");
            exit(EXIT_FAILURE);
        }
    }

	pthread_create(&thread, NULL, vf_HopperRender_start_AppIndicator_script, &priv->m_tdAppIndicatorThreadData);
	pthread_detach(thread);
	sleep(1);
	priv->m_iAppIndicatorFileDesc = open(fifo, O_WRONLY);
    if (priv->m_iAppIndicatorFileDesc == -1) {
        perror("Failed to open AppIndicator FIFO");
        exit(EXIT_FAILURE);
    }

	// ##############################
	// #### SET CPU AFFINITY #######
	// ##############################
	if (SPREAD_CORES) {
		cpu_set_t cpuset;
		int worker_core = SPREAD_CORES ? 1 : 0;
		CPU_ZERO(&cpuset);
		CPU_SET(worker_core, &cpuset);
		thread = pthread_self();
		pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
	}
	printf("[HopperRender] Warp thread running on CPU %d\n", sched_getcpu());

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