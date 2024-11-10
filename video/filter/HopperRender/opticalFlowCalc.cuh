#ifndef OPTICALFLOWCALC_CUH
#define OPTICALFLOWCALC_CUH

#include <stdbool.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpticalFlowCalc {
	void (*free)(struct OpticalFlowCalc *ofc);
	void (*updateFrame)(struct OpticalFlowCalc *ofc, unsigned char** pInBuffer, const unsigned int frameKernelSize, const unsigned int flowKernelSize, const bool directOutput);
	void (*downloadFrame)(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer);
	void (*processFrame)(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer, const bool firstFrame);
	void (*blurFrameArray)(struct OpticalFlowCalc *ofc, const void* frame, void* blurredFrame,
						const unsigned int kernelSize, const bool directOutput);
	void (*calculateOpticalFlow)(struct OpticalFlowCalc *ofc, unsigned int iNumIterations, unsigned int iNumSteps);
	void (*flipFlow)(struct OpticalFlowCalc *ofc);
	void (*blurFlowArrays)(struct OpticalFlowCalc *ofc);
	void (*warpFrames)(struct OpticalFlowCalc *ofc, float fScalar, const int outputMode);
	void (*blendFrames)(struct OpticalFlowCalc *ofc, float fScalar);
	void (*insertFrame)(struct OpticalFlowCalc *ofc);
	void (*sideBySideFrame)(struct OpticalFlowCalc *ofc, float fScalar, const unsigned int frameCounter);
	void (*drawFlowAsHSV)(struct OpticalFlowCalc *ofc, const float blendScalar);
	void (*drawFlowAsGreyscale)(struct OpticalFlowCalc *ofc);
	void (*saveImage)(struct OpticalFlowCalc *ofc, const char* filePath);

	// Video properties
	unsigned int m_iDimX; // Width of the frame
	unsigned int m_iDimY; // Height of the frame
	unsigned int m_iRealDimX; // Real width of the frame (not the stride width!)
	float m_fBlackLevel; // The black level used for the output frame
	float m_fWhiteLevel; // The white level used for the output frame
	bool m_bIsHDR; // Flag indicating if the video is HDR
	int m_iFMT; // The format of the video
	float m_fMaxVal; // The maximum value of the video format (255.0f for YUV420P and NV12, 1023.0f for YUV420P10, 65535.0f for P010)
	unsigned short m_iMiddleValue; // The middle value of the video format (128 for YUV420P and NV12, 512 for YUV420P10, 32768 for P010)

	// Optical flow calculation
	unsigned char m_cResolutionScalar; // Determines which resolution scalar will be used for the optical flow calculation
	unsigned int m_iLowDimX; // Width of the frame used by the optical flow calculation
	unsigned int m_iLowDimY; // Height of the frame used by the optical flow calculation
	unsigned int m_iNumLayers; // Number of layers used by the optical flow calculation
	unsigned int m_iDirectionIdxOffset; // m_iNumLayers * m_iLowDimY * m_iLowDimX
	unsigned int m_iLayerIdxOffset; // m_iLowDimY * m_iLowDimX
	unsigned int m_iChannelIdxOffset; // m_iDimY * m_iDimX
	unsigned int m_iFlowBlurKernelSize; // The kernel size used for the flow blur

	// CUDA streams
	cudaStream_t m_csOFCStream1, m_csOFCStream2; // CUDA streams used for the optical flow calculation
	cudaStream_t m_csWarpStream1, m_csWarpStream2; // CUDA streams used for the warping

	// Grids
	dim3 m_lowGrid32x32x1;
	dim3 m_lowGrid16x16x5;
	dim3 m_lowGrid16x16x4;
	dim3 m_lowGrid16x16x1;
	dim3 m_lowGrid8x8x5;
	dim3 m_lowGrid8x8x1;
	dim3 m_grid16x16x1;
	dim3 m_halfGrid16x16x1;
	dim3 m_grid8x8x1;
	
	// Threads
	dim3 m_threads32x32x1;
	dim3 m_threads16x16x2;
	dim3 m_threads16x16x1;
	dim3 m_threads8x8x5;
	dim3 m_threads8x8x2;
	dim3 m_threads8x8x1;

	// GPU Arrays
	int* m_offsetArray12; // Array containing x,y offsets for each pixel of frame1
	int* m_offsetArray21; // Array containing x,y offsets for each pixel of frame2
	int* m_blurredOffsetArray12[2]; // Array containing x,y offsets for each pixel of frame1
	int* m_blurredOffsetArray21[2]; // Array containing x,y offsets for each pixel of frame2
	unsigned char* m_statusArray; // Array containing the calculation status of each pixel of frame1
	unsigned int* m_summedUpDeltaArray; // Array containing the summed up delta values of each window
	unsigned char* m_lowestLayerArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	unsigned char* m_outputFrameSDR; // Array containing the output frame
	unsigned short* m_outputFrameHDR; // Array containing the output frame
	int* m_hitCount12; // Array containing the number of times a pixel was hit
	int* m_hitCount21; // Array containing the number of times a pixel was hit
	unsigned char* m_frameSDR[3]; // Array containing the last three frames
	unsigned short* m_frameHDR[3]; // Array containing the last three frames
	unsigned char* m_blurredFrameSDR[3]; // Array containing the last three frames after blurring
	unsigned short* m_blurredFrameHDR[3]; // Array containing the last three frames after blurring
	unsigned char* m_warpedFrame12SDR; // Array containing the warped frame (frame 1 to frame 2)
	unsigned short* m_warpedFrame12HDR; // Array containing the warped frame (frame 1 to frame 2)
	unsigned char* m_warpedFrame21SDR; // Array containing the warped frame (frame 2 to frame 1)
	unsigned short* m_warpedFrame21HDR; // Array containing the warped frame (frame 2 to frame 1)
	unsigned short* m_tempFrameHDR; // Temporary array for the output frame
	unsigned char* m_tempFrameSDR; // Temporary array for the output frame
	unsigned char* m_imageArrayCPU; // Array containing the image data
} OpticalFlowCalc;

/*
* Initializes the optical flow calculator
*
* @param ofc: Pointer to the optical flow calculator to be initialized
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param realDimX: The real width of the frame (not the stride width!)
* @param resolutionScalar: The resolution scalar used for the optical flow calculation
* @param flowBlurKernelSize: The kernel size used for the flow blur
* @param isHDR: Flag indicating if the video is HDR
* @param fmt: The format of the video
*/
void initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int dimY, const int dimX, const int realDimX, unsigned char resolutionScalar, unsigned int flowBlurKernelSize, bool isHDR, int fmt);

#ifdef __cplusplus
}
#endif

#endif // OPTICALFLOWCALC_CUH