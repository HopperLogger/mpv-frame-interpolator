#define CL_TARGET_OPENCL_VERSION 300

#include "opticalFlowCalc.h"

#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>

#include "config.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define CHECK_ERROR(err)                                                                \
    if (err != CL_SUCCESS) {                                                            \
        fprintf(stderr, "OpenCL error occurred in function: %s (%d)\n", __func__, err); \
        return 1;                                                                       \
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

bool updateFrame(struct OpticalFlowCalc* ofc, unsigned char** inputPlanes) {
    CHECK_ERROR(!ofc->isInitialized);
    CHECK_ERROR(clEnqueueWriteBuffer(ofc->queue, ofc->inputFrameArray[0], CL_TRUE, 0, ofc->frameHeight * ofc->frameWidth, inputPlanes[0], 0, NULL, &ofc->ofcStartedEvent));
    CHECK_ERROR(clEnqueueWriteBuffer(ofc->queue, ofc->inputFrameArray[0], CL_TRUE, ofc->frameHeight * ofc->frameWidth,
                                     (ofc->frameHeight / 2) * ofc->frameWidth, inputPlanes[1], 0, NULL, NULL));

    // Swap the frame buffers
    cl_mem temp0 = ofc->inputFrameArray[0];
    ofc->inputFrameArray[0] = ofc->inputFrameArray[1];
    ofc->inputFrameArray[1] = temp0;

    return 0;
}

bool downloadFrame(struct OpticalFlowCalc* ofc, unsigned char** outputPlanes) {
    CHECK_ERROR(!ofc->isInitialized);
    cl_event warpEndEvent;
    CHECK_ERROR(clEnqueueReadBuffer(ofc->queue, ofc->outputFrameArray, CL_TRUE, 0, ofc->frameHeight * ofc->frameWidth, outputPlanes[0], 0, NULL, NULL));
    CHECK_ERROR(clEnqueueReadBuffer(ofc->queue, ofc->outputFrameArray, CL_TRUE, ofc->frameHeight * ofc->frameWidth,
                                    (ofc->frameHeight / 2) * ofc->frameWidth, outputPlanes[1], 0, NULL, &warpEndEvent));

    // Evaluate how long the interpolation took
    CHECK_ERROR(clWaitForEvents(1, &ofc->warpStartedEvent));
    CHECK_ERROR(clWaitForEvents(1, &warpEndEvent));
    cl_ulong start_time, end_time;
    CHECK_ERROR(clGetEventProfilingInfo(ofc->warpStartedEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, NULL));
    CHECK_ERROR(clGetEventProfilingInfo(warpEndEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL));
    ofc->warpCalcTime = (double)(end_time - start_time) / 1e9;

    return 0;
}

bool calculateOpticalFlow(struct OpticalFlowCalc* ofc) {
    CHECK_ERROR(!ofc->isInitialized);

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
    windowSize /= 2;  // We don't want to compute movement of the entire frame, so we start with smaller windows

    // We only want to compute windows that are 2x2 or larger, so we adjust the needed iterations
    int opticalFlowIterations = NUM_ITERATIONS;
    if (NUM_ITERATIONS == 0 || NUM_ITERATIONS > log2(windowSize)) {
        opticalFlowIterations = log2(windowSize);
    }

    // Reset the number of offset layers
    int currSearchRadius = ofc->opticalFlowSearchRadius;
    CHECK_ERROR(adjustSearchRadius(ofc, ofc->opticalFlowSearchRadius));

    // Prepare the initial offset array
    cl_short zeroS = 0;
    CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->offsetArray12, &zeroS, sizeof(short), 0, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), 0, NULL, NULL));

    // We calculate the ideal offset array for each window size
    for (int iter = 0; iter < opticalFlowIterations; iter++) {
        for (int step = 0; step < 2; step++) {
            // Reset the summed up delta array
            cl_uint zeroUI = 0;
            CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->summedDeltaValuesArray, &zeroUI, sizeof(unsigned int), 0,
                                            currSearchRadius * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(unsigned int), 0, NULL, NULL));

            // 1. Calculate the image delta and sum up the deltas of each window
            cl_int err = clSetKernelArg(ofc->calcDeltaSumsKernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[0]);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[1]);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 9, sizeof(int), &windowSize);
            int isFirstIteration = (int)(iter <= 3);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 12, sizeof(int), &isFirstIteration);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 13, sizeof(int), &step);
            CHECK_ERROR(err);
            CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->calcDeltaSumsKernel, 3, NULL, ofc->lowGrid8x8xL, ofc->threads8x8x1, 0, NULL, NULL));

            // 2. Find the layer with the lowest delta sum
            CHECK_ERROR(clSetKernelArg(ofc->determineLowestLayerKernel, 2, sizeof(int), &windowSize));
            CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->determineLowestLayerKernel, 2, NULL, ofc->lowGrid16x16x1, ofc->threads16x16x1, 0, NULL, NULL));

            // 3. Adjust the offset array based on the comparison results
            err = clSetKernelArg(ofc->adjustOffsetArrayKernel, 2, sizeof(int), &windowSize);
            err = clSetKernelArg(ofc->adjustOffsetArrayKernel, 7, sizeof(int), &step);
            CHECK_ERROR(err);
            CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->adjustOffsetArrayKernel, 2, NULL, ofc->lowGrid16x16x1, ofc->threads16x16x1, 0, NULL, NULL));
        }

        // 4. Adjust variables for the next iteration
        windowSize = max(windowSize >> 1, (int)1);
    }

    // Flip the flow
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->flipFlowKernel, 3, NULL, ofc->lowGrid16x16x2, ofc->threads16x16x1, 0, NULL, NULL));

    // Blur the flow arrays
    CHECK_ERROR(clSetKernelArg(ofc->blurFlowKernel, 2, sizeof(cl_mem), &ofc->blurredOffsetArray12));
    CHECK_ERROR(clSetKernelArg(ofc->blurFlowKernel, 3, sizeof(cl_mem), &ofc->blurredOffsetArray21));
    cl_event ofcEndEvent;
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->blurFlowKernel, 3, NULL, ofc->lowGrid16x16x4, ofc->threads16x16x1, 0, NULL, &ofcEndEvent));

    // Evaluate how long the calculation took
    CHECK_ERROR(clWaitForEvents(1, &ofc->ofcStartedEvent));
    CHECK_ERROR(clWaitForEvents(1, &ofcEndEvent));
    cl_ulong start_time, end_time;
    CHECK_ERROR(clGetEventProfilingInfo(ofc->ofcStartedEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, NULL));
    CHECK_ERROR(clGetEventProfilingInfo(ofcEndEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL));
    ofc->ofcCalcTime = (double)(end_time - start_time) / 1e9;

    return 0;
}

bool warpFrames(struct OpticalFlowCalc* ofc, const float blendingScalar, const int frameOutputMode, const int isNewFrame) {
    CHECK_ERROR(!ofc->isInitialized);

    // Check if the blending scalar is valid
    if (blendingScalar > 1.0f) {
        printf("Error: Blending scalar is greater than 1.0\n");
        return 1;
    }

    // Calculate the blend scalar
    const float frameScalar12 = blendingScalar;
    const float frameScalar21 = 1.0f - blendingScalar;

    // Mark the beginning of the warp calculation
    CHECK_ERROR(clEnqueueMarkerWithWaitList(ofc->queue, 0, NULL, &ofc->warpStartedEvent));

    // Flush the warped frame buffers to avoid artifacts from previous frames
    if (isNewFrame) {
        if (frameOutputMode == 0) {
            CHECK_ERROR(clEnqueueCopyBuffer(ofc->queue, ofc->inputFrameArray[0], ofc->outputFrameArray, 0, 0, 1.5 * ofc->frameHeight * ofc->frameWidth, 0, NULL, NULL));
        } else if (frameOutputMode == 1) {
            CHECK_ERROR(clEnqueueCopyBuffer(ofc->queue, ofc->inputFrameArray[1], ofc->outputFrameArray, 0, 0, 1.5 * ofc->frameHeight * ofc->frameWidth, 0, NULL, NULL));
        } else {
            CHECK_ERROR(clEnqueueCopyBuffer(ofc->queue, ofc->inputFrameArray[0], ofc->warpedFrameArray12, 0, 0, 1.5 * ofc->frameHeight * ofc->frameWidth, 0, NULL, NULL));
            CHECK_ERROR(clEnqueueCopyBuffer(ofc->queue, ofc->inputFrameArray[1], ofc->warpedFrameArray21, 0, 0, 1.5 * ofc->frameHeight * ofc->frameWidth, 0, NULL, NULL));
        }
    }

    // Warp Frame 1 to Frame 2
    if (frameOutputMode != 1) {
    cl_int err = clSetKernelArg(ofc->warpFrameKernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[0]);
        err |= clSetKernelArg(ofc->warpFrameKernel, 1, sizeof(cl_mem), &ofc->blurredOffsetArray12);
        err |= clSetKernelArg(ofc->warpFrameKernel, 2, sizeof(cl_mem), (frameOutputMode < 2) ? &ofc->outputFrameArray : &ofc->warpedFrameArray12);
        err |= clSetKernelArg(ofc->warpFrameKernel, 3, sizeof(float), &frameScalar12);
        CHECK_ERROR(err);
        CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->warpFrameKernel, 3, NULL, ofc->grid16x16x2, ofc->threads16x16x1, 0, NULL, NULL));
    }

    // Warp Frame 2 to Frame 1
    if (frameOutputMode != 0) {
        cl_int err = clSetKernelArg(ofc->warpFrameKernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[1]);
        err |= clSetKernelArg(ofc->warpFrameKernel, 1, sizeof(cl_mem), &ofc->blurredOffsetArray21);
        err |= clSetKernelArg(ofc->warpFrameKernel, 2, sizeof(cl_mem), (frameOutputMode < 2) ? &ofc->outputFrameArray : &ofc->warpedFrameArray21);
        err |= clSetKernelArg(ofc->warpFrameKernel, 3, sizeof(float), &frameScalar21);
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->warpFrameKernel, 3, NULL, ofc->grid16x16x2, ofc->threads16x16x1, 0, NULL, NULL));
    }

    return 0;
}

bool blendFrames(struct OpticalFlowCalc* ofc, const float blendingScalar) {
    // Calculate the blend scalar
    const float frame1Scalar = 1.0f - blendingScalar;
    const float frame2Scalar = blendingScalar;

    // Blend the frames
    cl_int err = clSetKernelArg(ofc->blendFrameKernel, 3, sizeof(float), &frame1Scalar);
    err |= clSetKernelArg(ofc->blendFrameKernel, 4, sizeof(float), &frame2Scalar);
    err |= clSetKernelArg(ofc->blendFrameKernel, 8, sizeof(float), &ofc->outputBlackLevel);
    err |= clSetKernelArg(ofc->blendFrameKernel, 9, sizeof(float), &ofc->outputWhiteLevel);
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->blendFrameKernel, 3, NULL, ofc->grid16x16x2, ofc->threads16x16x1, 0, NULL, NULL));
    return 0;
}

bool sideBySide1(struct OpticalFlowCalc* ofc) {
    CHECK_ERROR(!ofc->isInitialized);
    cl_int err = clSetKernelArg(ofc->sideBySide1Kernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[0]);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 5, sizeof(float), &ofc->outputBlackLevel);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 6, sizeof(float), &ofc->outputWhiteLevel);
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->sideBySide1Kernel, 3, NULL, ofc->halfGrid16x16x2, ofc->threads16x16x1, 0, NULL, &ofc->warpStartedEvent));
    return 0;
}

bool sideBySide2(struct OpticalFlowCalc* ofc, const float blendingScalar, const int sourceFrameNum) {
    CHECK_ERROR(!ofc->isInitialized);

    // Calculate the blend scalar
    const float frame1Scalar = 1.0f - blendingScalar;
    const float frame2Scalar = blendingScalar;
    cl_int err = 0;

    if (sourceFrameNum == 1) {
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[1]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[1]);
        err |= clSetKernelArg(ofc->sideBySide2Kernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[1]);
    } else if (sourceFrameNum == 2) {
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
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->sideBySide2Kernel, 3, NULL, ofc->grid16x16x2, ofc->threads16x16x1, 0, NULL, &ofc->warpStartedEvent));
    return 0;
}

bool visualizeFlow(struct OpticalFlowCalc* ofc, const int doBWOutput) {
    CHECK_ERROR(!ofc->isInitialized);
    cl_int err = clSetKernelArg(ofc->visualizeFlowKernel, 0, sizeof(cl_mem), &ofc->blurredOffsetArray12);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 9, sizeof(int), &doBWOutput);
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->visualizeFlowKernel, 3, NULL, ofc->grid16x16x2, ofc->threads16x16x1, 0, NULL, &ofc->warpStartedEvent));
    return 0;
}

static int counter = 0;
bool tearingTest(struct OpticalFlowCalc* ofc) {
    CHECK_ERROR(!ofc->isInitialized);
    const int pos_x = counter % ofc->frameWidth;
    CHECK_ERROR(clSetKernelArg(ofc->tearingTestKernel, 3, sizeof(int), &pos_x));
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->tearingTestKernel, 3, NULL, ofc->grid16x16x2, ofc->threads16x16x1, 0, NULL, &ofc->warpStartedEvent));
    counter++;
    return 0;
}

#if DUMP_IMAGES
bool saveImage(struct OpticalFlowCalc* ofc, const char* filePath) {
    CHECK_ERROR(!ofc->isInitialized);

    // Copy the image array to the CPU
    size_t dataSize = 1.5 * ofc->frameHeight * ofc->frameWidth;
    CHECK_ERROR(clEnqueueReadBuffer(ofc->queue, ofc->outputFrameArray, CL_TRUE, 0, dataSize, ofc->imageDumpArray, 0, NULL, NULL));

    // Open file in binary write mode
    FILE* file = fopen(filePath, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Write the array to the file
    size_t written = fwrite(ofc->imageDumpArray, sizeof(unsigned char), dataSize, file);
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

bool adjustSearchRadius(struct OpticalFlowCalc* ofc, int newSearchRadius) {
    CHECK_ERROR(!ofc->isInitialized);
    ofc->lowGrid8x8xL[2] = newSearchRadius;
    cl_int err = clSetKernelArg(ofc->calcDeltaSumsKernel, 10, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 3, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 4, sizeof(int), &newSearchRadius);
    CHECK_ERROR(err);
    return 0;
}

void freeOFC(struct OpticalFlowCalc* ofc) {
    // Wait for all commands to finish
    clFinish(ofc->queue);

    // Free the GPU arrays
    clReleaseMemObject(ofc->inputFrameArray[0]);
    clReleaseMemObject(ofc->inputFrameArray[1]);
    clReleaseMemObject(ofc->warpedFrameArray12);
    clReleaseMemObject(ofc->warpedFrameArray21);
    clReleaseMemObject(ofc->outputFrameArray);
    clReleaseMemObject(ofc->offsetArray12);
    clReleaseMemObject(ofc->offsetArray21);
    clReleaseMemObject(ofc->blurredOffsetArray12);
    clReleaseMemObject(ofc->blurredOffsetArray21);
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
    clReleaseCommandQueue(ofc->queue);

    // Release the context
    clReleaseContext(ofc->clContext);

    // Release the device
    clReleaseDevice(ofc->clDeviceId);
}

// Detects the OpenCL platforms and devices
static bool detectDevices(struct OpticalFlowCalc* ofc) {
    // Capabilities we are going to check for
    cl_ulong availableVRAM;
    const cl_ulong requiredVRAM = 7.5 * ofc->frameHeight * ofc->frameWidth + 17lu * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth +
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
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkGroupSizes), maxWorkGroupSizes, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxSharedMemSize), &maxSharedMemSize, NULL);

            // Check if the device meets the requirements
            if (availableVRAM >= requiredVRAM && maxSharedMemSize >= requiredSharedMemSize && maxWorkGroupSizes[0] >= requiredWorkGroupSizes[0] && maxWorkGroupSizes[1] >= requiredWorkGroupSizes[1] &&
                maxWorkGroupSizes[2] >= requiredWorkGroupSizes[2]) {
                printf("[HopperRender] Using %s and %lu MB of VRAM\n", deviceName, requiredVRAM / 1024 / 1024);
                ofc->clDeviceId = devices[j];
                return 0;
            }
        }
    }

    // No suitable device found
    printf("Error: No suitable OpenCL GPU found! Please make sure that your GPU supports OpenCL 1.2 or higher and the OpenCL drivers are installed.\n");
    if (availableVRAM < requiredVRAM) {
        printf("Error: Not enough VRAM available! Required: %lu MB, Available: %lu MB\n", requiredVRAM / 1024 / 1024, availableVRAM / 1024 / 1024);
    }
    if (maxSharedMemSize < requiredSharedMemSize) {
        printf("Error: Not enough shared memory available! Required: %lu bytes, Available: %lu bytes\n", requiredSharedMemSize, maxSharedMemSize);
    }
    if (maxWorkGroupSizes[0] < requiredWorkGroupSizes[0] || maxWorkGroupSizes[1] < requiredWorkGroupSizes[1] || maxWorkGroupSizes[2] < requiredWorkGroupSizes[2]) {
        printf("Error: Not enough work group sizes available! Required: %lu, %lu, %lu, Available: %lu, %lu, %lu\n", requiredWorkGroupSizes[0], requiredWorkGroupSizes[1], requiredWorkGroupSizes[2],
               maxWorkGroupSizes[0], maxWorkGroupSizes[1], maxWorkGroupSizes[2]);
    }
    return 1;
}

bool initOpticalFlowCalc(struct OpticalFlowCalc* ofc, const int frameHeight, const int frameWidth) {
    // Video properties
    ofc->frameWidth = frameWidth;
    ofc->frameHeight = frameHeight;
    ofc->outputBlackLevel = 0.0f;
    ofc->outputWhiteLevel = 255.0f;

    // Optical flow calculation
    ofc->opticalFlowSearchRadius = DUMP_IMAGES ? MAX_SEARCH_RADIUS : MIN_SEARCH_RADIUS;
    ofc->opticalFlowResScalar = 0;
    while (frameHeight >> ofc->opticalFlowResScalar > MAX_CALC_RES) {
        ofc->opticalFlowResScalar++;
    }
    ofc->opticalFlowFrameWidth = ofc->frameWidth >> ofc->opticalFlowResScalar;
    ofc->opticalFlowFrameHeight = ofc->frameHeight >> ofc->opticalFlowResScalar;
    int directionIndexOffset = ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth;
    int channelIndexOffset = ofc->frameHeight * ofc->frameWidth;
    ofc->ofcCalcTime = 0.0;
    ofc->warpCalcTime = 0.0;

    // Define the global and local work sizes
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
    ofc->halfGrid16x16x2[0] = ceil((ofc->frameWidth / 2) / 16.0) * 16.0;
    ofc->halfGrid16x16x2[1] = ceil(ofc->frameHeight / 16.0) * 16.0;
    ofc->halfGrid16x16x2[2] = 2;

    ofc->threads16x16x1[0] = 16;
    ofc->threads16x16x1[1] = 16;
    ofc->threads16x16x1[2] = 1;
    ofc->threads8x8x1[0] = 8;
    ofc->threads8x8x1[1] = 8;
    ofc->threads8x8x1[2] = 1;

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
    CHECK_ERROR(detectDevices(ofc));

    // Create a context
    cl_int err;
    ofc->clContext = clCreateContext(0, 1, &ofc->clDeviceId, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create the command queues
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    ofc->queue = clCreateCommandQueueWithProperties(ofc->clContext, ofc->clDeviceId, properties, &err);
    CHECK_ERROR(err);

    // Allocate the buffers
    ofc->inputFrameArray[0] = clCreateBuffer(ofc->clContext, CL_MEM_READ_ONLY, 1.5 * frameHeight * frameWidth, NULL, &err);
    ofc->inputFrameArray[1] = clCreateBuffer(ofc->clContext, CL_MEM_READ_ONLY, 1.5 * frameHeight * frameWidth, NULL, &err);
    ofc->warpedFrameArray12 = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 1.5 * frameHeight * frameWidth, NULL, &err);
    ofc->warpedFrameArray21 = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 1.5 * frameHeight * frameWidth, NULL, &err);
    ofc->outputFrameArray = clCreateBuffer(ofc->clContext, CL_MEM_WRITE_ONLY, 1.5 * frameHeight * frameWidth, NULL, &err);
    ofc->offsetArray12 = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), NULL, &err);
    ofc->offsetArray21 = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), NULL, &err);
    ofc->blurredOffsetArray12 = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), NULL, &err);
    ofc->blurredOffsetArray21 = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), NULL, &err);
    ofc->summedDeltaValuesArray = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, MAX_SEARCH_RADIUS * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(unsigned int), NULL, &err);
    ofc->lowestLayerArray = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth, NULL, &err);
    CHECK_ERROR(err);
#if DUMP_IMAGES
    ofc->imageDumpArray = (unsigned short*)malloc(3 * frameHeight * frameWidth);
    if (!ofc->imageDumpArray) {
        fprintf(stderr, "Error allocating CPU memory for imageDumpArray\n");
        return 1;
    }
#endif

    // Clear the flow arrays
    cl_short zero = 0;
    CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->offsetArray12, &zero, sizeof(short), 0, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), 0, NULL, NULL));
    CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->offsetArray21, &zero, sizeof(short), 0, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), 0, NULL, NULL));
    CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->blurredOffsetArray12, &zero, sizeof(short), 0, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), 0, NULL, NULL));
    CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->blurredOffsetArray21, &zero, sizeof(short), 0, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), 0, NULL, NULL));

    // Compile the kernels
    CHECK_ERROR(cl_create_kernel(&ofc->calcDeltaSumsKernel, ofc->clContext, ofc->clDeviceId, "calcDeltaSumsKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->determineLowestLayerKernel, ofc->clContext, ofc->clDeviceId, "determineLowestLayerKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->adjustOffsetArrayKernel, ofc->clContext, ofc->clDeviceId, "adjustOffsetArrayKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->flipFlowKernel, ofc->clContext, ofc->clDeviceId, "flipFlowKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->blurFlowKernel, ofc->clContext, ofc->clDeviceId, "blurFlowKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->warpFrameKernel, ofc->clContext, ofc->clDeviceId, "warpFrameKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->blendFrameKernel, ofc->clContext, ofc->clDeviceId, "blendFrameKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->sideBySide1Kernel, ofc->clContext, ofc->clDeviceId, "sideBySide1Kernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->sideBySide2Kernel, ofc->clContext, ofc->clDeviceId, "sideBySide2Kernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->visualizeFlowKernel, ofc->clContext, ofc->clDeviceId, "visualizeFlowKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->tearingTestKernel, ofc->clContext, ofc->clDeviceId, "tearingTestKernel"));

    // Set kernel arguments
    err = clSetKernelArg(ofc->calcDeltaSumsKernel, 0, sizeof(cl_mem), &ofc->summedDeltaValuesArray);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 3, sizeof(cl_mem), &ofc->offsetArray12);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 4, sizeof(int), &directionIndexOffset);
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
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 3, sizeof(int), &directionIndexOffset);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 4, sizeof(int), &ofc->opticalFlowSearchRadius);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 5, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 6, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 4, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 5, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->warpFrameKernel, 6, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 7, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->warpFrameKernel, 8, sizeof(int), &directionIndexOffset);
    err |= clSetKernelArg(ofc->warpFrameKernel, 9, sizeof(int), &channelIndexOffset);
    err |= clSetKernelArg(ofc->blendFrameKernel, 0, sizeof(cl_mem), &ofc->warpedFrameArray12);
    err |= clSetKernelArg(ofc->blendFrameKernel, 1, sizeof(cl_mem), &ofc->warpedFrameArray21);
    err |= clSetKernelArg(ofc->blendFrameKernel, 2, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->blendFrameKernel, 5, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->blendFrameKernel, 6, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->blendFrameKernel, 7, sizeof(int), &channelIndexOffset);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 1, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 2, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 3, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->sideBySide1Kernel, 4, sizeof(int), &channelIndexOffset);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 3, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 6, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 7, sizeof(int), &ofc->frameWidth);
    int halfFrameHeight = ofc->frameHeight / 2;
    int halfFrameWidth = ofc->frameWidth / 2;
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 8, sizeof(int), &halfFrameHeight);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 9, sizeof(int), &halfFrameWidth);
    err |= clSetKernelArg(ofc->sideBySide2Kernel, 10, sizeof(int), &channelIndexOffset);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 1, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 2, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 3, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 4, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 5, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 6, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 7, sizeof(int), &directionIndexOffset);
    err |= clSetKernelArg(ofc->visualizeFlowKernel, 8, sizeof(int), &channelIndexOffset);
    err |= clSetKernelArg(ofc->flipFlowKernel, 0, sizeof(cl_mem), &ofc->offsetArray12);
    err |= clSetKernelArg(ofc->flipFlowKernel, 1, sizeof(cl_mem), &ofc->offsetArray21);
    err |= clSetKernelArg(ofc->flipFlowKernel, 2, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->flipFlowKernel, 3, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->flipFlowKernel, 4, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->flipFlowKernel, 5, sizeof(int), &directionIndexOffset);
    err |= clSetKernelArg(ofc->blurFlowKernel, 0, sizeof(cl_mem), &ofc->offsetArray12);
    err |= clSetKernelArg(ofc->blurFlowKernel, 1, sizeof(cl_mem), &ofc->offsetArray21);
    err |= clSetKernelArg(ofc->blurFlowKernel, 4, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->blurFlowKernel, 5, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->tearingTestKernel, 0, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->tearingTestKernel, 1, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->tearingTestKernel, 2, sizeof(int), &ofc->frameWidth);
    CHECK_ERROR(err);

    ofc->isInitialized = true;
    return 0;
}