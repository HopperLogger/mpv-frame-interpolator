#ifndef OPTICALFLOWCALC_H
#define OPTICALFLOWCALC_H

#include <stdbool.h>
#include <CL/cl.h>

typedef struct OpticalFlowCalc {
	// Video properties
	int m_iDimX; // Width of the frame
	int m_iDimY; // Height of the frame
	float m_fBlackLevel; // The black level used for the output frame
	float m_fWhiteLevel; // The white level used for the output frame
	float m_fMaxVal; // The maximum value of the video format (255.0f for YUV420P and NV12, 1023.0f for YUV420P10, 65535.0f for P010)
	int m_iMiddleValue; // The middle value of the video format (128 for YUV420P and NV12, 512 for YUV420P10, 32768 for P010)

	// Optical flow calculation
	int m_cResolutionScalar; // Determines which resolution scalar will be used for the optical flow calculation
	int m_iLowDimX; // Width of the frame used by the optical flow calculation
	int m_iLowDimY; // Height of the frame used by the optical flow calculation
	int m_iNumLayers; // Number of layers used by the optical flow calculation
	int m_iDirectionIdxOffset; // m_iNumLayers * m_iLowDimY * m_iLowDimX
	int m_iLayerIdxOffset; // m_iLowDimY * m_iLowDimX
	int m_iChannelIdxOffset; // m_iDimY * m_iDimX
	volatile bool m_bOFCTerminate; // Whether or not the optical flow calculator should terminate

	// OpenCL variables
	cl_device_id m_clDevice_id;
	cl_context m_clContext;
    cl_kernel m_clKernel;

	// Grids
	size_t m_lowGrid16x16x5[3];
	size_t m_lowGrid16x16x2[3];
	size_t m_lowGrid16x16x1[3];
	size_t m_lowGrid8x8x5[3];
	size_t m_grid16x16x2[3];
	size_t m_grid16x16x1[3];
	size_t m_halfGrid16x16x2[3];
	size_t m_grid8x8x2[3];
	
	// Threads
	size_t m_threads16x16x1[3];
	size_t m_threads8x8x1[3];

	// Queues
	cl_command_queue m_OFCQueue1, m_OFCQueue2; // Queue used for the optical flow calculation
	cl_command_queue m_WarpQueue1, m_WarpQueue2; // Queue used for the warping

	// GPU Arrays
	cl_mem m_offsetArray12; // Array containing x,y offsets for each pixel of frame1
	cl_mem m_offsetArray21; // Array containing x,y offsets for each pixel of frame2
	cl_mem m_blurredOffsetArray12[2]; // Array containing x,y offsets for each pixel of frame1
	cl_mem m_blurredOffsetArray21[2]; // Array containing x,y offsets for each pixel of frame2
	cl_mem m_summedUpDeltaArray; // Array containing the summed up delta values of each window
	cl_mem m_normalizedDeltaArray; // Array containing the normalized delta values of each window
	cl_mem m_lowestLayerArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	cl_mem m_outputFrame; // Array containing the output frame
	cl_mem m_hitCount12; // Array containing the number of times a pixel was hit
	cl_mem m_hitCount21; // Array containing the number of times a pixel was hit
	cl_mem m_frame[3]; // Array containing the last three frames
	cl_mem m_blurredFrame[3]; // Array containing the last three frames after blurring
	cl_mem m_warpedFrame12; // Array containing the warped frame (frame 1 to frame 2)
	cl_mem m_warpedFrame21; // Array containing the warped frame (frame 2 to frame 1)
	unsigned short* m_imageArrayCPU; // Array containing the image data

	// Kernels
	cl_kernel m_processFrameKernel;
	cl_kernel m_blurFrameKernel;
	cl_kernel m_setInitialOffsetKernel;
	cl_kernel m_calcDeltaSumsKernel;
	cl_kernel m_normalizeDeltaSumsKernel;
	cl_kernel m_determineLowestLayerKernel;
	cl_kernel m_adjustOffsetArrayKernel;
	cl_kernel m_flipFlowKernel;
	cl_kernel m_blurFlowKernel1;
	cl_kernel m_blurFlowKernel2;
	cl_kernel m_cleanFlowKernel;
	cl_kernel m_warpFrameKernel;
	cl_kernel m_artifactRemovalKernel;
	cl_kernel m_blendFrameKernel;
	cl_kernel m_insertFrameKernel;
	cl_kernel m_sideBySideFrameKernel;
	cl_kernel m_convertFlowToHSVKernel;
	cl_kernel m_convertFlowToGrayscaleKernel;
	cl_kernel m_tearingTestKernel;
} OpticalFlowCalc;

bool initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int dimY, const int dimX, const int resolutionScalar, const int searchRadius);
void freeOFC(struct OpticalFlowCalc *ofc);
bool adjustSearchRadius(struct OpticalFlowCalc *ofc, int newSearchRadius);
bool setKernelParameters(struct OpticalFlowCalc *ofc);
bool updateFrame(struct OpticalFlowCalc *ofc, unsigned char** pInBuffer, const bool directOutput);
bool downloadFrame(struct OpticalFlowCalc *ofc, const cl_mem pInBuffer, unsigned char** pOutBuffer);
bool processFrame(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer, const int frameCounter);
bool blurFrameArray(struct OpticalFlowCalc *ofc, const cl_mem frame, cl_mem blurredFrame, const bool directOutput);
bool calculateOpticalFlow(struct OpticalFlowCalc *ofc, int iNumIterations);
bool flipFlow(struct OpticalFlowCalc *ofc);
bool blurFlowArrays(struct OpticalFlowCalc *ofc);
bool warpFrames(struct OpticalFlowCalc *ofc, const float fScalar, const int outputMode);
bool blendFrames(struct OpticalFlowCalc *ofc, const float fScalar);
bool insertFrame(struct OpticalFlowCalc *ofc);
bool sideBySideFrame(struct OpticalFlowCalc *ofc, const float fScalar, const int frameCounter);
bool drawFlowAsHSV(struct OpticalFlowCalc *ofc, const float blendScalar);
bool drawFlowAsGrayscale(struct OpticalFlowCalc *ofc);
bool saveImage(struct OpticalFlowCalc *ofc, const char* filePath);
bool tearingTest(struct OpticalFlowCalc *ofc);

#endif // OPTICALFLOWCALC_H