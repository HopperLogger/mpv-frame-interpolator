#ifndef OPTICALFLOWCALC_CUH
#define OPTICALFLOWCALC_CUH

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpticalFlowCalc {
	void (*free)(struct OpticalFlowCalc *ofc);
	void (*reinit)(struct OpticalFlowCalc *ofc);
	void (*updateFrame)(struct OpticalFlowCalc *ofc, unsigned char** pInBuffer, const unsigned int frameKernelSize, const bool directOutput);
	void (*downloadFrame)(struct OpticalFlowCalc *ofc, const unsigned short* pInBuffer, unsigned char** pOutBuffer);
	void (*processFrame)(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer, const int frameCounter);
	void (*blurFrameArray)(struct OpticalFlowCalc *ofc, const unsigned short* frame, unsigned short* blurredFrame,
						const int kernelSize, const bool directOutput);
	void (*calculateOpticalFlow)(struct OpticalFlowCalc *ofc, unsigned int iNumIterations);
	void (*flipFlow)(struct OpticalFlowCalc *ofc);
	void (*blurFlowArrays)(struct OpticalFlowCalc *ofc);
	void (*warpFrames)(struct OpticalFlowCalc *ofc, float fScalar, const int outputMode);
	void (*blendFrames)(struct OpticalFlowCalc *ofc, float fScalar);
	void (*insertFrame)(struct OpticalFlowCalc *ofc);
	void (*sideBySideFrame)(struct OpticalFlowCalc *ofc, float fScalar, const unsigned int frameCounter);
	void (*drawFlowAsHSV)(struct OpticalFlowCalc *ofc, const float blendScalar);
	void (*drawFlowAsGreyscale)(struct OpticalFlowCalc *ofc);
	void (*saveImage)(struct OpticalFlowCalc *ofc, const char* filePath);
	void (*tearingTest)(struct OpticalFlowCalc *ofc);

	// Video properties
	unsigned int m_iDimX; // Width of the frame
	unsigned int m_iDimY; // Height of the frame
	float m_fBlackLevel; // The black level used for the output frame
	float m_fWhiteLevel; // The white level used for the output frame
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
	volatile bool m_bOFCTerminate; // Whether or not the optical flow calculator should terminate

	// GPU Arrays
	char* m_offsetArray12; // Array containing x,y offsets for each pixel of frame1
	char* m_offsetArray21; // Array containing x,y offsets for each pixel of frame2
	char* m_blurredOffsetArray12[2]; // Array containing x,y offsets for each pixel of frame1
	char* m_blurredOffsetArray21[2]; // Array containing x,y offsets for each pixel of frame2
	unsigned int* m_summedUpDeltaArray; // Array containing the summed up delta values of each window
	double* m_normalizedDeltaArray; // Array containing the normalized delta values of each window
	unsigned char* m_lowestLayerArray; // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
	unsigned short* m_outputFrame; // Array containing the output frame
	int* m_hitCount12; // Array containing the number of times a pixel was hit
	int* m_hitCount21; // Array containing the number of times a pixel was hit
	unsigned short* m_frame[3]; // Array containing the last three frames
	unsigned short* m_blurredFrame[3]; // Array containing the last three frames after blurring
	unsigned short* m_warpedFrame12; // Array containing the warped frame (frame 1 to frame 2)
	unsigned short* m_warpedFrame21; // Array containing the warped frame (frame 2 to frame 1)
	unsigned short* m_tempFrame; // Temporary array for the output frame
	unsigned short* m_imageArrayCPU; // Array containing the image data

	void* priv; // Private data
} OpticalFlowCalc;

/*
* Initializes the optical flow calculator
*
* @param ofc: Pointer to the optical flow calculator to be initialized
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param resolutionScalar: The resolution scalar used for the optical flow calculation
* @param flowBlurKernelSize: The kernel size used for the flow blur
*/
void initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int dimY, const int dimX, unsigned char resolutionScalar, unsigned int flowBlurKernelSize);

#ifdef __cplusplus
}
#endif

#endif // OPTICALFLOWCALC_CUH