#define CL_TARGET_OPENCL_VERSION 300

#include "opticalFlowCalc.h"

#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>

#include "config.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define ERR_CHECK(cond) \
    if (cond) {         \
        return 1;       \
    }
#define ERR_MSG_CHECK(err, func)                                                              \
    if (err != CL_SUCCESS) {                                                                  \
        fprintf(stderr, "Error setting kernel parameters in function: %s - %d\n", func, err); \
        return 1;                                                                             \
    }

// Function to launch an OpenCL kernel and check for errors
static bool cl_enqueue_kernel(cl_command_queue queue, cl_kernel kernel, cl_uint workDim, const size_t* globalWorkSize,
                              const size_t* localWorkSize, const char* functionName) {
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, workDim, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing kernel in function: %s - %d\n", functionName, err);
        return 1;
    }
    return 0;
}

// Function to copy an OpenCL device buffer to another device buffer and check for errors
static bool cl_copy_buffer(cl_command_queue queue, cl_mem src, size_t srcOffset, cl_mem dst, size_t dstOffset,
                           size_t size, const char* functionName) {
    cl_int err = clEnqueueCopyBuffer(queue, src, dst, srcOffset, dstOffset, size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error copying buffer in function: %s - %d\n", functionName, err);
        return 1;
    }
    return 0;
}

// Function to upload data to an OpenCL buffer and check for errors
static bool cl_upload_buffer(cl_command_queue queue, void* src, cl_mem dst, size_t offset, size_t size,
                             const char* functionName) {
    cl_int err = clEnqueueWriteBuffer(queue, dst, CL_TRUE, offset, size, src, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error uploading buffer in function: %s - %d\n", functionName, err);
        return 1;
    }
    return 0;
}

// Function to download data from an OpenCL buffer and check for errors
static bool cl_download_buffer(cl_command_queue queue, cl_mem src, void* dst, size_t offset, size_t size,
                               const char* functionName) {
    cl_int err = clEnqueueReadBuffer(queue, src, CL_TRUE, offset, size, dst, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error downloading buffer in function: %s - %d\n", functionName, err);
        return 1;
    }
    return 0;
}

// Function to zero an OpenCL buffer and check for errors
static bool cl_zero_buffer(cl_command_queue queue, cl_mem buffer, size_t size, size_t patternSize,
                           const char* functionName) {
    cl_uint zero = 0;
    cl_int err = clEnqueueFillBuffer(queue, buffer, &zero, patternSize, 0, size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error zeroing buffer in function: %s - %d\n", functionName, err);
        return 1;
    }
    return 0;
}

// Function to create an OpenCL buffer and check for errors
static bool cl_create_buffer(cl_mem* buffer, cl_context context, cl_mem_flags flags, size_t size,
                             const char* bufferName) {
    cl_int err;
    *buffer = clCreateBuffer(context, flags, size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer %s: %d\n", bufferName, err);
        return 1;
    }
    return 0;
}

// Function to read the OpenCL kernel source code from a file
static const char* loadKernelSource(const char* fileName) {
    FILE* file = fopen(fileName, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open kernel file: %s\n", fileName);
        return NULL;
    }

    // Get the file size
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the kernel source
    char* source = (char*)malloc(fileSize + 1);  // +1 for null terminator
    if (!source) {
        fprintf(stderr, "Error: Memory allocation failed for kernel source\n");
        fclose(file);
        return NULL;
    }

    // Read the file content into the source string
    fread(source, 1, fileSize, file);
    source[fileSize] = '\0';

    fclose(file);
    return source;
}

// Function to create an OpenCL kernel
static bool cl_create_kernel(cl_kernel* kernel, cl_context context, cl_device_id deviceId, const char* kernelFunc) {
    cl_int err;

    // Assemble the path to the kernel source file
    const char* home = getenv("HOME");
    if (home == NULL) {
        fprintf(stderr, "HOME environment variable is not set.\n");
        return 1;
    }
    char kernelSourcePath[512];
    sprintf(kernelSourcePath, "%s/mpv-build/mpv/video/filter/HopperRender/Kernels/%s.cl", home, kernelFunc);

    // Load kernel source code from file
    const char* kernelSourceFile = loadKernelSource(kernelSourcePath);
    if (kernelSourceFile == NULL) {
        return 1;
    }

    // Create the compute program from the source buffer
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceFile, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create program\n");
        free((void*)kernelSourceFile);
        return 1;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get and print the build log for debugging
        char buildLog[4096];
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        fprintf(stderr, "Build log: %s\n", buildLog);
        free((void*)kernelSourceFile);
        return 1;
    }

    // Create the compute kernel in the program we wish to run
    *kernel = clCreateKernel(program, kernelFunc, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create kernel %s\n", kernelFunc);
        free((void*)kernelSourceFile);
        return 1;
    }

    clReleaseProgram(program);
    return 0;
}

/*
 * Updates the frame arrays and blurs them if necessary
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param inputPlanes: Pointer to the input planes of the new source frame
 *
 * @return: Whether or not the frame arrays were updated successfully
 */
bool updateFrame(struct OpticalFlowCalc* ofc, unsigned char** inputPlanes) {
    ERR_CHECK(cl_upload_buffer(ofc->queueOFC, inputPlanes[0], ofc->inputFrameArray[0], 0,
                               ofc->frameHeight * ofc->frameWidth * sizeof(unsigned short), "updateFrame"));
    ERR_CHECK(cl_upload_buffer(ofc->queueOFC, inputPlanes[1], ofc->inputFrameArray[0],
                               ofc->frameHeight * ofc->frameWidth * sizeof(unsigned short),
                               (ofc->frameHeight / 2) * ofc->frameWidth * sizeof(unsigned short), "updateFrame"));

    // Swap the frame buffers
    cl_mem temp0 = ofc->inputFrameArray[0];
    ofc->inputFrameArray[0] = ofc->inputFrameArray[1];
    ofc->inputFrameArray[1] = ofc->inputFrameArray[2];
    ofc->inputFrameArray[2] = temp0;

    // Swap the blurred offset arrays
    temp0 = ofc->blurredOffsetArray12[0];
    ofc->blurredOffsetArray12[0] = ofc->blurredOffsetArray12[1];
    ofc->blurredOffsetArray12[1] = temp0;

    temp0 = ofc->blurredOffsetArray21[0];
    ofc->blurredOffsetArray21[0] = ofc->blurredOffsetArray21[1];
    ofc->blurredOffsetArray21[1] = temp0;
    return 0;
}

/*
 * Downloads the output frame from the GPU to the CPU
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param sourceBuffer: The buffer to download the frame from
 * @param outputPlanes: Pointer to the output planes where the frame should be stored
 *
 * @return: Whether or not the frame was downloaded successfully
 */
bool downloadFrame(struct OpticalFlowCalc* ofc, const cl_mem sourceBuffer, unsigned char** outputPlanes) {
    ERR_CHECK(cl_download_buffer(ofc->queueOFC, sourceBuffer, outputPlanes[0], 0,
                                 ofc->frameHeight * ofc->frameWidth * sizeof(unsigned short), "downloadFrame"));
    ERR_CHECK(cl_download_buffer(ofc->queueOFC, sourceBuffer, outputPlanes[1],
                                 ofc->frameHeight * ofc->frameWidth * sizeof(unsigned short),
                                 (ofc->frameHeight / 2) * ofc->frameWidth * sizeof(unsigned short), "downloadFrame"));
    return 0;
}

/*
 * Calculates the optical flow between frame1 and frame2
 *
 * @param ofc: Pointer to the optical flow calculator
 *
 * @return: Whether or not the optical flow was calculated successfully
 */
bool calculateOpticalFlow(struct OpticalFlowCalc* ofc) {
    // We set the initial window size to the next larger power of 2
    int windowSize = 1;
    int maxDim = max(ofc->opticalFlowFrameWidth, ofc->opticalFlowFrameHeight);
    if (maxDim && !(maxDim & (maxDim - 1))) {
        windowSize = maxDim;
    } else {
        while (maxDim & (maxDim - 1)) {
            maxDim &= (maxDim - 1);
        }
        windowSize = maxDim << 1;
    }
    windowSize /= 2; // We don't want to compute movement of the entire frame, so we start with smaller windows

    // We only want to compute windows that are 2x2 or larger, so we adjust the needed iterations
    if (NUM_ITERATIONS == 0 || NUM_ITERATIONS > log2(windowSize)) {
        ofc->opticalFlowIterations = log2(windowSize);
    }

    // Reset the number of offset layers
    int currSearchRadius = ofc->opticalFlowSearchRadius;
    ERR_CHECK(adjustSearchRadius(ofc, ofc->opticalFlowSearchRadius));

    // Prepare the initial offset array
    ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->offsetArray12,
                             2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                             sizeof(short), "calculateOpticalFlow"));

    // We calculate the ideal offset array for each window size
    for (int iter = 0; iter < ofc->opticalFlowIterations; iter++) {
        for (int step = 0; step < ofc->opticalFlowSteps * 2; step++) {
            // Reset the summed up delta array
            ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->summedDeltaValuesArray,
                                     currSearchRadius * ofc->opticalFlowFrameHeight *
                                         ofc->opticalFlowFrameWidth * sizeof(unsigned int),
                                     sizeof(unsigned int), "calculateOpticalFlow"));

            // 1. Calculate the image delta and sum up the deltas of each window
            cl_int err = clSetKernelArg(ofc->calcDeltaSumsKernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[1]);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[2]);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 9, sizeof(int), &windowSize);
            int isFirstIteration = (int)(iter <= 3);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 12, sizeof(int), &isFirstIteration);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 13, sizeof(int), &step);
            ERR_MSG_CHECK(err, "calculateOpticalFlow");
            ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->calcDeltaSumsKernel, 3, ofc->lowGrid8x8xL,
                                        ofc->threads8x8x1, "calculateOpticalFlow"));

            // 2. Find the layer with the lowest delta sum
            err = clSetKernelArg(ofc->determineLowestLayerKernel, 2, sizeof(int), &windowSize);
            ERR_MSG_CHECK(err, "calculateOpticalFlow");
            ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->determineLowestLayerKernel, 2, ofc->lowGrid16x16x1,
                                        ofc->threads16x16x1, "calculateOpticalFlow"));

            // 3. Adjust the offset array based on the comparison results
            err = clSetKernelArg(ofc->adjustOffsetArrayKernel, 2, sizeof(int), &windowSize);
            err = clSetKernelArg(ofc->adjustOffsetArrayKernel, 7, sizeof(int), &step);
            ERR_MSG_CHECK(err, "calculateOpticalFlow");
            ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->adjustOffsetArrayKernel, 2, ofc->lowGrid16x16x1,
                                        ofc->threads16x16x1, "calculateOpticalFlow"));
        }

        // 4. Adjust variables for the next iteration
        windowSize = max(windowSize >> 1, (int)1);
/*         if (iter != ofc->opticalFlowIterations - 1) {
            currSearchRadius = max(ofc->opticalFlowSearchRadius - iter, 4);
            ERR_CHECK(adjustSearchRadius(ofc, currSearchRadius));
        } */
    }
    return 0;
}

/*
 * Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
 *
 * @param ofc: Pointer to the optical flow calculator
 *
 * @return: Whether or not the flow was flipped successfully
 */
bool flipFlow(struct OpticalFlowCalc* ofc) {
    // Launch kernel
    ERR_CHECK(
        cl_enqueue_kernel(ofc->queueOFC, ofc->flipFlowKernel, 3, ofc->lowGrid16x16x2, ofc->threads16x16x1, "flipFlow"));
    return 0;
}

/*
 * Blurs the offset arrays
 *
 * @param ofc: Pointer to the optical flow calculator
 *
 * @return: Whether or not the offset arrays were blurred successfully
 */
bool blurFlowArrays(struct OpticalFlowCalc* ofc) {
    // No need to blur the flow
    if (!FLOW_BLUR_ENABLED) {
        ERR_CHECK(cl_copy_buffer(ofc->queueOFC, ofc->offsetArray12, 0, ofc->blurredOffsetArray12[1], 0,
                                 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                                 "blurFlowArrays"));
        ERR_CHECK(cl_copy_buffer(ofc->queueOFC, ofc->offsetArray21, 0, ofc->blurredOffsetArray21[1], 0,
                                 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                                 "blurFlowArrays"));
        return 0;
    }

    // Launch kernels
    cl_int err = clSetKernelArg(ofc->blurFlowKernel, 2, sizeof(cl_mem), &ofc->blurredOffsetArray12[1]);
    err |= clSetKernelArg(ofc->blurFlowKernel, 3, sizeof(cl_mem), &ofc->blurredOffsetArray21[1]);
    ERR_MSG_CHECK(err, "blurFlowArrays");
    ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->blurFlowKernel, 3, ofc->lowGrid16x16x4, ofc->threads16x16x1,
                                "blurFlowArrays"));

    return 0;
}

/*
 * Warps the frames according to the calculated optical flow
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param blendingScalar: The scalar to blend the frames with (i.e. the progress between frame1 and frame2)
 * @param frameOutputMode: The mode to output the frames in (0: WarpedFrame12, 1: WarpedFrame21, 2: Both)
 * @param isNewFrame: Whether or not this is the first call for a new source frame
 *
 * @return: Whether or not the frames were warped successfully
 */
bool warpFrames(struct OpticalFlowCalc* ofc, const float blendingScalar, const int frameOutputMode,
                const int isNewFrame) {
    // Check if the blending scalar is valid
    if (blendingScalar > 1.0f) {
        printf("Error: Blending scalar is greater than 1.0\n");
        return 1;
    }

    // Calculate the blend scalar
    const float frameScalar12 = blendingScalar;
    const float frameScalar21 = 1.0f - blendingScalar;

    // Flush the warped frame buffers to avoid artifacts from previous frames
    if (isNewFrame) {
        if (frameOutputMode == 0) {
            ERR_CHECK(cl_copy_buffer(ofc->queueOFC, ofc->inputFrameArray[0], 0, ofc->outputFrameArray, 0,
                                 3 * ofc->frameHeight * ofc->frameWidth, "warpFrames1"));
        } else if (frameOutputMode == 1) {
            ERR_CHECK(cl_copy_buffer(ofc->queueOFC, ofc->inputFrameArray[1], 0, ofc->outputFrameArray, 0,
                                 3 * ofc->frameHeight * ofc->frameWidth, "warpFrames2"));
        } else {
            ERR_CHECK(cl_copy_buffer(ofc->queueOFC, ofc->inputFrameArray[0], 0, ofc->warpedFrameArray12, 0,
                                 3 * ofc->frameHeight * ofc->frameWidth, "warpFrames1"));
            ERR_CHECK(cl_copy_buffer(ofc->queueOFC, ofc->inputFrameArray[1], 0, ofc->warpedFrameArray21, 0,
                                 3 * ofc->frameHeight * ofc->frameWidth, "warpFrames2"));
        }
    }

    // Warp Frame 1 to Frame 2
    if (frameOutputMode != 1) {
        cl_int err = clSetKernelArg(ofc->warpFrameKernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[0]);
        err |= clSetKernelArg(ofc->warpFrameKernel, 1, sizeof(cl_mem), &ofc->blurredOffsetArray12[0]);
        err |= clSetKernelArg(ofc->warpFrameKernel, 2, sizeof(cl_mem),
                              (frameOutputMode < 2) ? &ofc->outputFrameArray : &ofc->warpedFrameArray12);
        err |= clSetKernelArg(ofc->warpFrameKernel, 3, sizeof(float), &frameScalar12);
        ERR_MSG_CHECK(err, "warpFrames3");
        ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->warpFrameKernel, 3, ofc->grid16x16x2, ofc->threads16x16x1,
                                    "warpFrames4"));
    }

    // Warp Frame 2 to Frame 1
    if (frameOutputMode != 0) {
        cl_int err = clSetKernelArg(ofc->warpFrameKernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[1]);
        err |= clSetKernelArg(ofc->warpFrameKernel, 1, sizeof(cl_mem), &ofc->blurredOffsetArray21[0]);
        err |= clSetKernelArg(ofc->warpFrameKernel, 2, sizeof(cl_mem),
                              (frameOutputMode < 2) ? &ofc->outputFrameArray : &ofc->warpedFrameArray21);
        err |= clSetKernelArg(ofc->warpFrameKernel, 3, sizeof(float), &frameScalar21);
        ERR_MSG_CHECK(err, "warpFrames5");
        ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->warpFrameKernel, 3, ofc->grid16x16x2, ofc->threads16x16x1,
                                    "warpFrames6"));
    }

    return 0;
}

/*
 * Blends warpedFrame1 and warpedFrame2 according to the blending scalar
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param blendingScalar: The scalar to blend the frames with (i.e. the progress between frame1 and frame2)
 *
 * @return: Whether or not the frames were blended successfully
 */
bool blendFrames(struct OpticalFlowCalc* ofc, const float blendingScalar) {
    // Calculate the blend scalar
    const float frame1Scalar = 1.0f - blendingScalar;
    const float frame2Scalar = blendingScalar;

    // Blend the frames
    cl_int err = clSetKernelArg(ofc->blendFrameKernel, 3, sizeof(float), &frame1Scalar);
    err |= clSetKernelArg(ofc->blendFrameKernel, 4, sizeof(float), &frame2Scalar);
    err |= clSetKernelArg(ofc->blendFrameKernel, 8, sizeof(float), &ofc->outputBlackLevel);
    err |= clSetKernelArg(ofc->blendFrameKernel, 9, sizeof(float), &ofc->outputWhiteLevel);
    ERR_MSG_CHECK(err, "blendFrames");
    ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->blendFrameKernel, 3, ofc->grid16x16x2, ofc->threads16x16x1,
                                "blendFrames"));
    return 0;
}

/*
 * Places the left half of inputFrameArray[0] over the outputFrame
 *
 * @param ofc: Pointer to the optical flow calculator
 *
 * @return: Whether or not the frame was inserted successfully
 */
bool sideBySide1(struct OpticalFlowCalc* ofc) {
    cl_int err = clSetKernelArg(ofc->sideBySide1Kernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[0]);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 5, sizeof(float), &ofc->outputBlackLevel);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 6, sizeof(float), &ofc->outputWhiteLevel);
    ERR_MSG_CHECK(err, "sideBySide1");
    ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->sideBySide1Kernel, 3, ofc->halfGrid16x16x2,
                                ofc->threads16x16x1, "sideBySide1"));
    return 0;
}

/*
 * Produces a side by side comparison where the current source frame is on the left and the interpolated result is on
 * the right
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param blendingScalar: The scalar to blend the frames with (i.e. the progress between frame1 and frame2)
 * @param sourceFrameNum: The current source frame number
 *
 * @return: Whether or not the frames were blended successfully
 */
bool sideBySide2(struct OpticalFlowCalc* ofc, const float blendingScalar, const int sourceFrameNum) {
    // Calculate the blend scalar
    const float frame1Scalar = 1.0f - blendingScalar;
    const float frame2Scalar = blendingScalar;
    cl_int err = 0;

    if (sourceFrameNum == 1) {
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[2]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[2]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[2]);
    } else if (sourceFrameNum == 2) {
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[1]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[1]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[1]);
    } else if (sourceFrameNum <= 3) {
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[0]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[0]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[0]);
    } else {
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[0]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 1, sizeof(cl_mem), &ofc->warpedFrameArray12);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 2, sizeof(cl_mem), &ofc->warpedFrameArray21);
    }
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 4, sizeof(float), &frame1Scalar);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 5, sizeof(float), &frame2Scalar);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 11, sizeof(float), &ofc->outputBlackLevel);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 12, sizeof(float), &ofc->outputWhiteLevel);
    ERR_MSG_CHECK(err, "sideBySide2");
    ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->sideBySide2Kernel, 3, ofc->grid16x16x2, ofc->threads16x16x1,
                                "sideBySide2"));
    return 0;
}

/*
 * Draws the offsetArray12 as an RGB image visualizing the flow
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param doBWOutput: Whether to output the flow as a black and white image
 *
 * @return: Whether or not the flow was drawn successfully
 */
bool visualizeFlow(struct OpticalFlowCalc* ofc, const int doBWOutput) {
    cl_int err = clSetKernelArg(ofc->visualizeFlowKernel, 0, sizeof(cl_mem), &ofc->blurredOffsetArray12[0]);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[1]);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 10, sizeof(int), &doBWOutput);
    ERR_MSG_CHECK(err, "visualizeFlow");
    ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->visualizeFlowKernel, 3, ofc->grid16x16x2, ofc->threads16x16x1,
                                "visualizeFlow"));
    return 0;
}

/*
 * Draws a white vertical bar on the screen that can be used to detect tearing
 *
 * @param ofc: Pointer to the optical flow calculator
 *
 * @return: Whether or not the bar was drawn successfully
 */
static int counter = 0;
bool tearingTest(struct OpticalFlowCalc* ofc) {
    const int pos_x = counter % ofc->frameWidth;
    cl_int err = clSetKernelArg(ofc->tearingTestKernel, 3, sizeof(int), &pos_x);
    ERR_MSG_CHECK(err, "tearingTest");
    ERR_CHECK(cl_enqueue_kernel(ofc->queueOFC, ofc->tearingTestKernel, 3, ofc->grid16x16x2, ofc->threads16x16x1,
                                "tearingTest"));
    counter++;
    return 0;
}

#if DUMP_IMAGES
/*
 * Saves the outputFrameArray in binary form to ~/dump
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param filePath: Path to the image file
 *
 * @return: Whether or not the image was saved successfully
 */
bool saveImage(struct OpticalFlowCalc* ofc, const char* filePath) {
    // Copy the image array to the CPU
    size_t dataSize = 1.5 * ofc->frameHeight * ofc->frameWidth;
    ERR_CHECK(cl_download_buffer(ofc->queueOFC, ofc->outputFrameArray, ofc->imageDumpArray, 0,
                                 dataSize * sizeof(unsigned short), "saveImage"));

    // Open file in binary write mode
    FILE* file = fopen(filePath, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Write the array to the file
    size_t written = fwrite(ofc->imageDumpArray, sizeof(unsigned short), dataSize, file);
    if (written != dataSize) {
        perror("Error writing to file");
        fclose(file);
        return 1;
    }

    // Close the file
    fclose(file);

    return 0;
}
#endif

/*
 * Sets the kernel parameters and other kernel related variables used in the optical flow calculation
 *
 * @param ofc: The optical flow calculation struct
 *
 * @return: Whether or not the kernel parameters were set successfully
 */
bool setKernelParameters(struct OpticalFlowCalc* ofc) {
    // Define the global and local work sizes
    ofc->lowGrid16x16xL[0] = ceil(ofc->opticalFlowFrameWidth / 16.0) * 16.0;
    ofc->lowGrid16x16xL[1] = ceil(ofc->opticalFlowFrameHeight / 16.0) * 16.0;
    ofc->lowGrid16x16xL[2] = ofc->opticalFlowSearchRadius;
    ofc->lowGrid16x16x4[0] = ceil(ofc->opticalFlowFrameWidth / 16.0) * 16.0;
    ofc->lowGrid16x16x4[1] = ceil(ofc->opticalFlowFrameHeight / 16.0) * 16.0;
    ofc->lowGrid16x16x4[2] = 4;
    ofc->lowGrid16x16x2[0] = ceil(ofc->opticalFlowFrameWidth / 16.0) * 16.0;
    ofc->lowGrid16x16x2[1] = ceil(ofc->opticalFlowFrameHeight / 16.0) * 16.0;
    ofc->lowGrid16x16x2[2] = 2;
    ofc->lowGrid16x16x1[0] = ceil(ofc->opticalFlowFrameWidth / 16.0) * 16.0;
    ofc->lowGrid16x16x1[1] = ceil(ofc->opticalFlowFrameHeight / 16.0) * 16.0;
    ofc->lowGrid16x16x1[2] = 1;
    ofc->lowGrid8x8xL[0] = ceil(ofc->opticalFlowFrameWidth / 8.0) * 8.0;
    ofc->lowGrid8x8xL[1] = ceil(ofc->opticalFlowFrameHeight / 8.0) * 8.0;
    ofc->lowGrid8x8xL[2] = ofc->opticalFlowSearchRadius;
    ofc->grid16x16x2[0] = ceil(ofc->frameWidth / 16.0) * 16.0;
    ofc->grid16x16x2[1] = ceil(ofc->frameHeight / 16.0) * 16.0;
    ofc->grid16x16x2[2] = 2;
    ofc->grid16x16x1[0] = ceil(ofc->frameWidth / 16.0) * 16.0;
    ofc->grid16x16x1[1] = ceil(ofc->frameHeight / 16.0) * 16.0;
    ofc->grid16x16x1[2] = 1;
    ofc->halfGrid16x16x2[0] = ceil((ofc->frameWidth / 2) / 16.0) * 16.0;
    ofc->halfGrid16x16x2[1] = ceil(ofc->frameHeight / 16.0) * 16.0;
    ofc->halfGrid16x16x2[2] = 2;

    ofc->threads16x16x1[0] = 16;
    ofc->threads16x16x1[1] = 16;
    ofc->threads16x16x1[2] = 1;
    ofc->threads8x8x1[0] = 8;
    ofc->threads8x8x1[1] = 8;
    ofc->threads8x8x1[2] = 1;

    // Clear the flow arrays
    ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->offsetArray12,
                             2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                             sizeof(short), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->offsetArray21,
                             2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                             sizeof(short), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->blurredOffsetArray12[0],
                             2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                             sizeof(short), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->blurredOffsetArray21[0],
                             2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                             sizeof(short), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->blurredOffsetArray12[1],
                             2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                             sizeof(short), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->queueOFC, ofc->blurredOffsetArray21[1],
                             2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                             sizeof(short), "reinit"));

    // Set kernel arguments
    cl_int err = clSetKernelArg(ofc->calcDeltaSumsKernel, 0, sizeof(cl_mem), &ofc->summedDeltaValuesArray);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 3, sizeof(cl_mem), &ofc->offsetArray12);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 4, sizeof(int), &ofc->directionIndexOffset);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 5, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 6, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 7, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 8, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 10, sizeof(int), &ofc->opticalFlowSearchRadius);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 11, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 0, sizeof(cl_mem), &ofc->summedDeltaValuesArray);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 1, sizeof(cl_mem), &ofc->lowestLayerArray);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 3, sizeof(int), &ofc->opticalFlowSearchRadius);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 4, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 5, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 0, sizeof(cl_mem), &ofc->offsetArray12);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 1, sizeof(cl_mem), &ofc->lowestLayerArray);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 3, sizeof(int), &ofc->directionIndexOffset);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 4, sizeof(int), &ofc->opticalFlowSearchRadius);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 5, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 6, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 4, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 5, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->warpFrameKernel, 6, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 7, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->warpFrameKernel, 8, sizeof(int), &ofc->directionIndexOffset);
    err |= clSetKernelArg(ofc->warpFrameKernel, 9, sizeof(int), &ofc->channelIndexOffset);
    err |= clSetKernelArg(ofc->blendFrameKernel, 0, sizeof(cl_mem), &ofc->warpedFrameArray12);
    err |= clSetKernelArg(ofc->blendFrameKernel, 1, sizeof(cl_mem), &ofc->warpedFrameArray21);
    err |= clSetKernelArg(ofc->blendFrameKernel, 2, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->blendFrameKernel, 5, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->blendFrameKernel, 6, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->blendFrameKernel, 7, sizeof(int), &ofc->channelIndexOffset);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 1, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 2, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 3, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 4, sizeof(int), &ofc->channelIndexOffset);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 3, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 6, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 7, sizeof(int), &ofc->frameWidth);
    int halfFrameHeight = ofc->frameHeight / 2;
    int halfFrameWidth = ofc->frameWidth / 2;
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 8, sizeof(int), &halfFrameHeight);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 9, sizeof(int), &halfFrameWidth);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 10, sizeof(int), &ofc->channelIndexOffset);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 1, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 3, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 4, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 5, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 6, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 7, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 8, sizeof(int), &ofc->directionIndexOffset);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 9, sizeof(int), &ofc->channelIndexOffset);
    err |= clSetKernelArg(ofc->flipFlowKernel, 0, sizeof(cl_mem), &ofc->offsetArray12);
    err |= clSetKernelArg(ofc->flipFlowKernel, 1, sizeof(cl_mem), &ofc->offsetArray21);
    err |= clSetKernelArg(ofc->flipFlowKernel, 2, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->flipFlowKernel, 3, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->flipFlowKernel, 4, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->flipFlowKernel, 5, sizeof(int), &ofc->directionIndexOffset);
    err |= clSetKernelArg(ofc->blurFlowKernel, 0, sizeof(cl_mem), &ofc->offsetArray12);
    err |= clSetKernelArg(ofc->blurFlowKernel, 1, sizeof(cl_mem), &ofc->offsetArray21);
    err |= clSetKernelArg(ofc->blurFlowKernel, 4, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->blurFlowKernel, 5, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->tearingTestKernel, 0, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->tearingTestKernel, 1, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->tearingTestKernel, 2, sizeof(int), &ofc->frameWidth);
    ERR_MSG_CHECK(err, "setKernelParameters");
    return 0;
}

/*
 * Adjusts the search radius of the optical flow calculation
 *
 * @param ofc: Pointer to the optical flow calculator
 * @param newSearchRadius: The new search radius
 *
 * @return: Whether or not the search radius was adjusted successfully
 */
bool adjustSearchRadius(struct OpticalFlowCalc* ofc, int newSearchRadius) {
    ofc->lowGrid16x16xL[2] = newSearchRadius;
    ofc->lowGrid8x8xL[2] = newSearchRadius;
    cl_int err = clSetKernelArg(ofc->calcDeltaSumsKernel, 10, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 3, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 4, sizeof(int), &newSearchRadius);
    ERR_MSG_CHECK(err, "adjustSearchRadius");
    return 0;
}

/*
 * Frees the memory of the optical flow calculator
 *
 * @param ofc: Pointer to the optical flow calculator
 */
void freeOFC(struct OpticalFlowCalc* ofc) {
    // Free the GPU arrays
    clReleaseMemObject(ofc->inputFrameArray[0]);
    clReleaseMemObject(ofc->inputFrameArray[1]);
    clReleaseMemObject(ofc->inputFrameArray[2]);
    clReleaseMemObject(ofc->warpedFrameArray12);
    clReleaseMemObject(ofc->warpedFrameArray21);
    clReleaseMemObject(ofc->outputFrameArray);
    clReleaseMemObject(ofc->offsetArray12);
    clReleaseMemObject(ofc->offsetArray21);
    clReleaseMemObject(ofc->blurredOffsetArray12[0]);
    clReleaseMemObject(ofc->blurredOffsetArray21[0]);
    clReleaseMemObject(ofc->blurredOffsetArray12[1]);
    clReleaseMemObject(ofc->blurredOffsetArray21[1]);
    clReleaseMemObject(ofc->summedDeltaValuesArray);
    clReleaseMemObject(ofc->lowestLayerArray);
#if DUMP_IMAGES
    free(ofc->imageDumpArray);
#endif

    // Release the kernels
    clReleaseKernel(ofc->calcDeltaSumsKernel);
    clReleaseKernel(ofc->determineLowestLayerKernel);
    clReleaseKernel(ofc->adjustOffsetArrayKernel);
    clReleaseKernel(ofc->flipFlowKernel);
    clReleaseKernel(ofc->blurFlowKernel);
    clReleaseKernel(ofc->warpFrameKernel);
    clReleaseKernel(ofc->blendFrameKernel);
    clReleaseKernel(ofc->sideBySide1Kernel);
    clReleaseKernel(ofc->sideBySide2Kernel);
    clReleaseKernel(ofc->visualizeFlowKernel);

    // Release the command queues
    clReleaseCommandQueue(ofc->queueOFC);

    // Release the context
    clReleaseContext(ofc->clContext);

    // Release the device
    clReleaseDevice(ofc->clDeviceId);
}

// Detects the OpenCL platforms and devices
static bool detectDevices(struct OpticalFlowCalc* ofc) {
    // Capabilities we are going to check for
    cl_ulong availableVRAM;
    const cl_ulong requiredVRAM =
        18lu * ofc->frameHeight * ofc->frameWidth +
        25lu * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth +
        MAX_SEARCH_RADIUS * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(unsigned int);
    size_t maxWorkGroupSizes[3];
    const size_t requiredWorkGroupSizes[3] = {16, 16, 1};
    cl_ulong maxSharedMemSize;
    const cl_ulong requiredSharedMemSize = 2048;

    // Query the available platforms
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        printf("Error getting platform count: %d\n", err);
        return 1;
    }
    cl_platform_id platforms[8];
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    // Iterate over the available platforms
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        // Query the available devices of this platform
        cl_uint numDevices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

        if (numDevices == 0) continue;

        cl_device_id devices[8];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        // Iterate over the available devices
        for (cl_uint j = 0; j < numDevices; ++j) {
            // Get the capabilities of the device
            char deviceName[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(availableVRAM), &availableVRAM, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkGroupSizes), maxWorkGroupSizes,
                            NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxSharedMemSize), &maxSharedMemSize, NULL);

            // Check if the device meets the requirements
            if (availableVRAM >= requiredVRAM && maxSharedMemSize >= requiredSharedMemSize &&
                maxWorkGroupSizes[0] >= requiredWorkGroupSizes[0] &&
                maxWorkGroupSizes[1] >= requiredWorkGroupSizes[1] &&
                maxWorkGroupSizes[2] >= requiredWorkGroupSizes[2]) {
                printf("[HopperRender] Using %s and %lu MB of VRAM\n", deviceName, requiredVRAM / 1024 / 1024);
                ofc->clDeviceId = devices[j];
                return 0;
            }
        }
    }

    // No suitable device found
    printf(
        "Error: No suitable OpenCL GPU found! Please make sure that your GPU supports OpenCL 1.2 or higher and the "
        "OpenCL drivers are installed.\n");
    if (availableVRAM < requiredVRAM) {
        printf("Error: Not enough VRAM available! Required: %lu MB, Available: %lu MB\n", requiredVRAM / 1024 / 1024,
               availableVRAM / 1024 / 1024);
    }
    if (maxSharedMemSize < requiredSharedMemSize) {
        printf("Error: Not enough shared memory available! Required: %lu bytes, Available: %lu bytes\n",
               requiredSharedMemSize, maxSharedMemSize);
    }
    if (maxWorkGroupSizes[0] < requiredWorkGroupSizes[0] || maxWorkGroupSizes[1] < requiredWorkGroupSizes[1] ||
        maxWorkGroupSizes[2] < requiredWorkGroupSizes[2]) {
        printf("Error: Not enough work group sizes available! Required: %lu, %lu, %lu, Available: %lu, %lu, %lu\n",
               requiredWorkGroupSizes[0], requiredWorkGroupSizes[1], requiredWorkGroupSizes[2], maxWorkGroupSizes[0],
               maxWorkGroupSizes[1], maxWorkGroupSizes[2]);
    }
    return 1;
}

/*
 * Initializes the optical flow calculator
 *
 * @param ofc: Pointer to the optical flow calculator to be initialized
 * @param frameHeight: The height of the video frame
 * @param frameWidth: The width of the video frame
 *
 * @return: Whether or not the optical flow calculator was initialized successfully
 */
bool initOpticalFlowCalc(struct OpticalFlowCalc* ofc, const int frameHeight, const int frameWidth) {
    // Video properties
    ofc->frameWidth = frameWidth;
    ofc->frameHeight = frameHeight;
    ofc->outputBlackLevel = 0.0f;
    ofc->outputWhiteLevel = 65535.0f;

    // Optical flow calculation
    ofc->opticalFlowIterations = NUM_ITERATIONS;
    ofc->opticalFlowSearchRadius = DUMP_IMAGES ? MAX_SEARCH_RADIUS : MIN_SEARCH_RADIUS;
    ofc->opticalFlowMAXSearchRadius = MAX_SEARCH_RADIUS;
    ofc->opticalFlowSteps = DUMP_IMAGES ? MAX_NUM_STEPS : 1;
    ofc->opticalFlowResScalar = MAX_RES_SCALAR;
    ofc->opticalFlowMinResScalar = MAX_RES_SCALAR;
    ofc->opticalFlowFrameWidth = ofc->frameWidth >> ofc->opticalFlowResScalar;
    ofc->opticalFlowFrameHeight = ofc->frameHeight >> ofc->opticalFlowResScalar;
    ofc->directionIndexOffset = ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth;
    ofc->channelIndexOffset = ofc->frameHeight * ofc->frameWidth;
    if (ofc->frameHeight > 1080) {
        ofc->opticalFlowMinResScalar += 1;
        ofc->opticalFlowResScalar += 1;
    }

    // Check if the Kernels are accessible
    const char* home = getenv("HOME");
    if (home == NULL) {
        fprintf(stderr, "HOME environment variable is not set.\n");
        return 1;
    }
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/mpv-build/mpv/video/filter/HopperRender/Kernels", home);
    struct stat st;
    if (!(stat(full_path, &st) == 0 && S_ISDIR(st.st_mode))) {
        printf("[HopperRender] OpenCL Kernels not found in %s\n", full_path);
        return 1;
    }

    // Detect platforms and devices
    ERR_CHECK(detectDevices(ofc));

    // Create a context
    cl_int err;
    ofc->clContext = clCreateContext(0, 1, &ofc->clDeviceId, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create context\n");
        return 1;
    }

    // Create the command queues
    ofc->queueOFC = clCreateCommandQueueWithProperties(ofc->clContext, ofc->clDeviceId, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create command queue\n");
        return 1;
    }

    // Allocate the GPU Arrays
    ERR_CHECK(cl_create_buffer(&ofc->inputFrameArray[0], ofc->clContext, CL_MEM_READ_ONLY, 3 * frameHeight * frameWidth,
                               "frame[0]"));
    ERR_CHECK(cl_create_buffer(&ofc->inputFrameArray[1], ofc->clContext, CL_MEM_READ_ONLY, 3 * frameHeight * frameWidth,
                               "frame[1]"));
    ERR_CHECK(cl_create_buffer(&ofc->inputFrameArray[2], ofc->clContext, CL_MEM_READ_ONLY, 3 * frameHeight * frameWidth,
                               "frame[2]"));
    ERR_CHECK(cl_create_buffer(&ofc->warpedFrameArray12, ofc->clContext, CL_MEM_READ_WRITE,
                               3 * frameHeight * frameWidth, "warpedFrame12"));
    ERR_CHECK(cl_create_buffer(&ofc->warpedFrameArray21, ofc->clContext, CL_MEM_READ_WRITE,
                               3 * frameHeight * frameWidth, "warpedFrame21"));
    ERR_CHECK(cl_create_buffer(&ofc->outputFrameArray, ofc->clContext, CL_MEM_WRITE_ONLY, 3 * frameHeight * frameWidth,
                               "outputFrame"));
    ERR_CHECK(cl_create_buffer(&ofc->offsetArray12, ofc->clContext, CL_MEM_READ_WRITE,
                               2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                               "offsetArray12"));
    ERR_CHECK(cl_create_buffer(&ofc->offsetArray21, ofc->clContext, CL_MEM_READ_WRITE,
                               2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                               "offsetArray21"));
    ERR_CHECK(cl_create_buffer(&ofc->blurredOffsetArray12[0], ofc->clContext, CL_MEM_READ_WRITE,
                               2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                               "blurredOffsetArray12[0]"));
    ERR_CHECK(cl_create_buffer(&ofc->blurredOffsetArray21[0], ofc->clContext, CL_MEM_READ_WRITE,
                               2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                               "blurredOffsetArray21[0]"));
    ERR_CHECK(cl_create_buffer(&ofc->blurredOffsetArray12[1], ofc->clContext, CL_MEM_READ_WRITE,
                               2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                               "blurredOffsetArray12[1]"));
    ERR_CHECK(cl_create_buffer(&ofc->blurredOffsetArray21[1], ofc->clContext, CL_MEM_READ_WRITE,
                               2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short),
                               "blurredOffsetArray21[1]"));
    ERR_CHECK(cl_create_buffer(&ofc->summedDeltaValuesArray, ofc->clContext, CL_MEM_READ_WRITE,
                               MAX_SEARCH_RADIUS * ofc->opticalFlowFrameHeight *
                                   ofc->opticalFlowFrameWidth * sizeof(unsigned int),
                               "summedUpDeltaArray"));
    ERR_CHECK(cl_create_buffer(&ofc->lowestLayerArray, ofc->clContext, CL_MEM_READ_WRITE,
                               ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth, "lowestLayerArray"));
#if DUMP_IMAGES
    ofc->imageDumpArray = (unsigned short*)malloc(3 * frameHeight * frameWidth);
    if (!ofc->imageDumpArray) {
        fprintf(stderr, "Error allocating CPU memory for imageDumpArray\n");
        return 1;
    }
#endif

    // Compile the kernels
    ERR_CHECK(cl_create_kernel(&ofc->calcDeltaSumsKernel, ofc->clContext, ofc->clDeviceId, "calcDeltaSumsKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->determineLowestLayerKernel, ofc->clContext, ofc->clDeviceId,
                               "determineLowestLayerKernel"));
    ERR_CHECK(
        cl_create_kernel(&ofc->adjustOffsetArrayKernel, ofc->clContext, ofc->clDeviceId, "adjustOffsetArrayKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->flipFlowKernel, ofc->clContext, ofc->clDeviceId, "flipFlowKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->blurFlowKernel, ofc->clContext, ofc->clDeviceId, "blurFlowKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->warpFrameKernel, ofc->clContext, ofc->clDeviceId, "warpFrameKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->blendFrameKernel, ofc->clContext, ofc->clDeviceId, "blendFrameKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->sideBySide1Kernel, ofc->clContext, ofc->clDeviceId, "sideBySide1Kernel"));
    ERR_CHECK(cl_create_kernel(&ofc->sideBySide2Kernel, ofc->clContext, ofc->clDeviceId, "sideBySide2Kernel"));
    ERR_CHECK(cl_create_kernel(&ofc->visualizeFlowKernel, ofc->clContext, ofc->clDeviceId, "visualizeFlowKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->tearingTestKernel, ofc->clContext, ofc->clDeviceId, "tearingTestKernel"));

    // Set kernel arguments
    setKernelParameters(ofc);

    return 0;
}