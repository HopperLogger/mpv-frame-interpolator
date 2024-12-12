#ifndef OPTICALFLOWCALC_H
#define OPTICALFLOWCALC_H

#define CL_TARGET_OPENCL_VERSION 300
#include <stdbool.h>
#include <CL/cl.h>
#include "config.h"

typedef struct OpticalFlowCalc {
	// Video properties
	int frameWidth; // Width of the frame
	int frameHeight; // Height of the frame
	float outputBlackLevel; // The black level used for the output frame
	float outputWhiteLevel; // The white level used for the output frame

	// Optical flow calculation
	int opticalFlowIterations; // Number of iterations to use in the optical flow calculation (0: As many as possible)
	int opticalFlowSteps; // How many repetitions of each iteration will be executed to find the best offset for each window
	int opticalFlowResScalar; // Determines which resolution scalar will be used for the optical flow calculation
	int opticalFlowMinResScalar; // The minimum resolution scalar
	int opticalFlowFrameWidth; // Width of the frame used by the optical flow calculation
	int opticalFlowFrameHeight; // Height of the frame used by the optical flow calculation
	int opticalFlowSearchRadius; // Search radius used for the optical flow calculation
	int directionIndexOffset; // m_iNumLayers * opticalFlowFrameHeight * opticalFlowFrameWidth
	int layerIndexOffset; // opticalFlowFrameHeight * opticalFlowFrameWidth
	int channelIndexOffset; // frameHeight * frameWidth
	volatile bool opticalFlowCalcShouldTerminate; // Whether or not the optical flow calculator should terminate
	bool frameBlurEnabled; // Whether or not the frame should be blurred
	bool flowBlurEnabled; // Whether or not the flow should be blurred

	// OpenCL variables
	cl_device_id clDeviceId;
	cl_context clContext;

	// Grids
	size_t lowGrid16x16xL[3];
	size_t lowGrid16x16x4[3];
	size_t lowGrid16x16x2[3];
	size_t lowGrid16x16x1[3];
	size_t lowGrid8x8xL[3];
	size_t grid16x16x2[3];
	size_t grid16x16x1[3];
	size_t halfGrid16x16x2[3];
	
	// Threads
	size_t threads16x16x1[3];
	size_t threads8x8x1[3];

	// Queues
	cl_command_queue queueOFC; // Queue used for the optical flow calculation
	cl_command_queue queueWarping1, queueWarping2; // Queues used for the warping

	// GPU Arrays
	cl_mem offsetArray12; // Array containing x,y offsets for each pixel of frame1
	cl_mem offsetArray21; // Array containing x,y offsets for each pixel of frame2
	cl_mem blurredOffsetArray12[2]; // Array containing x,y offsets for each pixel of frame1
	cl_mem blurredOffsetArray21[2]; // Array containing x,y offsets for each pixel of frame2
	cl_mem summedDeltaValuesArray; // Array containing the summed up delta values of each window
	cl_mem lowestLayerArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	cl_mem outputFrameArray; // Array containing the output frame
	cl_mem inputFrameArray[3]; // Array containing the last three frames
	cl_mem blurredFrameArray[3]; // Array containing the last three frames after blurring
	cl_mem warpedFrameArray12; // Array containing the warped frame (frame 1 to frame 2)
	cl_mem warpedFrameArray21; // Array containing the warped frame (frame 2 to frame 1)
	#if DUMP_IMAGES
	unsigned short* imageDumpArray; // Array containing the image data
	#endif

	// Kernels
	cl_kernel blurFrameKernel;
	cl_kernel setInitialOffsetKernel;
	cl_kernel calcDeltaSumsKernel;
	cl_kernel determineLowestLayerKernel;
	cl_kernel adjustOffsetArrayKernel;
	cl_kernel flipFlowKernel;
	cl_kernel blurFlowKernel;
	cl_kernel warpFrameKernel;
	cl_kernel blendFrameKernel;
	cl_kernel sideBySide1Kernel;
	cl_kernel sideBySide2Kernel;
	cl_kernel visualizeFlowKernel;
	cl_kernel tearingTestKernel;
} OpticalFlowCalc;

bool initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int frameHeight, const int frameWidth);
void freeOFC(struct OpticalFlowCalc *ofc);
bool adjustSearchRadius(struct OpticalFlowCalc *ofc, int newSearchRadius);
bool setKernelParameters(struct OpticalFlowCalc *ofc);
bool updateFrame(struct OpticalFlowCalc *ofc, unsigned char** inputPlanes, const bool outputBlurDirectly);
bool downloadFrame(struct OpticalFlowCalc *ofc, const cl_mem sourceBuffer, unsigned char** outputPlanes);
bool calculateOpticalFlow(struct OpticalFlowCalc *ofc);
bool flipFlow(struct OpticalFlowCalc *ofc);
bool blurFlowArrays(struct OpticalFlowCalc *ofc);
bool warpFrames(struct OpticalFlowCalc *ofc, const float blendingScalar, const int frameOutputMode, const int isNewFrame);
bool blendFrames(struct OpticalFlowCalc *ofc, const float blendingScalar);
bool sideBySide1(struct OpticalFlowCalc *ofc);
bool sideBySide2(struct OpticalFlowCalc *ofc, const float blendingScalar, const int sourceFrameNum);
bool visualizeFlow(struct OpticalFlowCalc *ofc, const int doBWOutput);
bool tearingTest(struct OpticalFlowCalc *ofc);
#if DUMP_IMAGES
bool saveImage(struct OpticalFlowCalc *ofc, const char* filePath);
#endif

#endif // OPTICALFLOWCALC_H