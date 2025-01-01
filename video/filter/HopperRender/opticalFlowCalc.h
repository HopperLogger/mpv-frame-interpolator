#ifndef OPTICALFLOWCALC_H
#define OPTICALFLOWCALC_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdbool.h>

#include "config.h"

typedef struct OpticalFlowCalc {
    // Video properties
    bool isInitialized;      // Whether or not the optical flow calculator is initialized
    int frameWidth;          // Width of the frame
    int frameHeight;         // Height of the frame
    float outputBlackLevel;  // The black level used for the output frame
    float outputWhiteLevel;  // The white level used for the output frame

    // Optical flow calculation
    int opticalFlowResScalar;     // Determines which resolution scalar will be used for the optical flow calculation
    int opticalFlowFrameWidth;    // Width of the frame used by the optical flow calculation
    int opticalFlowFrameHeight;   // Height of the frame used by the optical flow calculation
    int opticalFlowSearchRadius;  // Search radius used for the optical flow calculation
    double ofcCalcTime;           // The time it took to calculate the optical flow
    double warpCalcTime;          // The time it took to warp the current intermediate frame

    // OpenCL variables
    cl_device_id clDeviceId;
    cl_context clContext;

    // Grids
    size_t lowGrid16x16x2[3];
    size_t lowGrid16x16x1[3];
    size_t lowGrid8x8xL[3];
    size_t grid16x16x2[3];

    // Threads
    size_t threads16x16x1[3];
    size_t threads8x8x1[3];

    // Queues
    cl_command_queue queue;  // Queue used for the optical flow calculation

    // Events
    cl_event ofcStartedEvent;   // Event marking the start of the optical flow calculation
    cl_event warpStartedEvent;  // Event marking the start of the interpolation

    // GPU Arrays
    cl_mem offsetArray;             // Array containing x,y offsets for each pixel of frame1
    cl_mem blurredOffsetArray;      // Array containing x,y offsets for each pixel of frame1
    cl_mem summedDeltaValuesArray;  // Array containing the summed up delta values of each window
    cl_mem lowestLayerArray;        // Array containing the comparison results of the two normalized delta arrays (true if the new value decreased)
    cl_mem outputFrameArray;        // Array containing the output frame
    cl_mem inputFrameArray[2];      // Array containing the last three frames

    // Kernels
    cl_kernel calcDeltaSumsKernel;
    cl_kernel determineLowestLayerKernel;
    cl_kernel adjustOffsetArrayKernel;
    cl_kernel blurFlowKernel;
    cl_kernel warpFrameKernel;
} OpticalFlowCalc;

/*
 * Initializes the optical flow calculator
 *
 * @param ofc: Pointer to the optical flow calculator to be initialized
 * @param frameHeight: The height of the video frame
 * @param frameWidth: The width of the video frame
 *
 * @return: Whether or not the optical flow calculator was initialized successfully
 */
bool initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int frameHeight, const int frameWidth);

/*
 * Frees the memory of the optical flow calculator
 *
 * @param ofc: Pointer to the optical flow calculator
 */
void freeOFC(struct OpticalFlowCalc *ofc);

/*
 * Updates the frame arrays and blurs them if necessary
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param inputPlanes: Pointer to the input planes of the new source frame
 *
 * @return: Whether or not the frame arrays were updated successfully
 */
bool updateFrame(struct OpticalFlowCalc *ofc, unsigned char **inputPlanes);

/*
 * Downloads the output frame from the GPU to the CPU
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param outputPlanes: Pointer to the output planes where the frame should be stored
 *
 * @return: Whether or not the frame was downloaded successfully
 */
bool downloadFrame(struct OpticalFlowCalc *ofc, unsigned char **outputPlanes);

/*
 * Calculates the optical flow between frame1 and frame2
 *
 * @param ofc: Pointer to the optical flow calculator
 *
 * @return: Whether or not the optical flow was calculated successfully
 */
bool calculateOpticalFlow(struct OpticalFlowCalc *ofc);

/*
 * Warps the frames according to the calculated optical flow
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param blendingScalar: The scalar to blend the frames with (i.e. the progress between frame1 and frame2)
 * @param frameOutputMode: The mode to output the frames in (0: WarpedFrame12, 1: WarpedFrame21, 2: Both)
 *
 * @return: Whether or not the frames were warped successfully
 */
bool warpFrames(struct OpticalFlowCalc *ofc, const float blendingScalar, const int frameOutputMode);

/*
 * Adjusts the search radius of the optical flow calculation
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param newSearchRadius: The new search radius
 *
 * @return: Whether or not the search radius was adjusted successfully
 */
bool adjustSearchRadius(struct OpticalFlowCalc *ofc, int newSearchRadius);

#endif  // OPTICALFLOWCALC_H