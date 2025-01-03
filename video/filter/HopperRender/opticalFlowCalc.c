#define CL_TARGET_OPENCL_VERSION 300

#include "opticalFlowCalc.h"
#include "config.h"
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
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

    // Prepare the initial offset array
    cl_uint zero = 0;
    CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->offsetArray, &zero, sizeof(short), 0, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), 0, NULL, NULL));

    // We calculate the ideal offset array for each window size
    for (int iter = 0; iter < opticalFlowIterations; iter++) {
        for (int step = 0; step < 2; step++) {
            // Reset the summed up delta array
            CHECK_ERROR(clEnqueueFillBuffer(ofc->queue, ofc->summedDeltaValuesArray, &zero, sizeof(unsigned int), 0,
                                            ofc->opticalFlowSearchRadius * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(unsigned int), 0, NULL, NULL));

            // 1. Calculate the image delta and sum up the deltas of each window
            cl_event test;
            cl_int err = clSetKernelArg(ofc->calcDeltaSumsKernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[0]);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 2, sizeof(cl_mem), &ofc->inputFrameArray[1]);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 9, sizeof(int), &windowSize);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 12, sizeof(int), &iter);
            err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 13, sizeof(int), &step);
            CHECK_ERROR(err);
            CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->calcDeltaSumsKernel, 3, NULL, ofc->lowGrid16x16xL, ofc->threads16x16x1, 0, NULL, &test));
/*             CHECK_ERROR(clWaitForEvents(1, &test));
            cl_ulong start_time, end_time;
            CHECK_ERROR(clGetEventProfilingInfo(test, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL));
            CHECK_ERROR(clGetEventProfilingInfo(test, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL));
            if ((double)(end_time - start_time) / 1e6 > 1.5)
                printf("Took too long. %d\n", iter); */

            // 2. Find the layer with the lowest delta sum
            CHECK_ERROR(clSetKernelArg(ofc->determineLowestLayerKernel, 2, sizeof(int), &windowSize));
            CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->determineLowestLayerKernel, 2, NULL, ofc->lowGrid16x16x1, ofc->threads16x16x1, 0, NULL, NULL));

            // 3. Adjust the offset array based on the comparison results
            err = clSetKernelArg(ofc->adjustOffsetArrayKernel, 2, sizeof(int), &windowSize);
            err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 7, sizeof(int), &step);
            CHECK_ERROR(err);
            CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->adjustOffsetArrayKernel, 2, NULL, ofc->lowGrid16x16x1, ofc->threads16x16x1, 0, NULL, NULL));
        }

        // 4. Adjust variables for the next iteration
        windowSize = max(windowSize >> 1, (int)1);
    }

    // Blur the flow array
    cl_event ofcEndEvent;
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->blurFlowKernel, 3, NULL, ofc->lowGrid16x16x2, ofc->threads16x16x1, 0, NULL, &ofcEndEvent));

    // Evaluate how long the calculation took
    CHECK_ERROR(clWaitForEvents(1, &ofc->ofcStartedEvent));
    CHECK_ERROR(clWaitForEvents(1, &ofcEndEvent));
    cl_ulong start_time, end_time;
    CHECK_ERROR(clGetEventProfilingInfo(ofc->ofcStartedEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, NULL));
    CHECK_ERROR(clGetEventProfilingInfo(ofcEndEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL));
    ofc->ofcCalcTime = (double)(end_time - start_time) / 1e9;
    return 0;
}

bool warpFrames(struct OpticalFlowCalc* ofc, const float blendingScalar, const int frameOutputMode) {
    CHECK_ERROR(!ofc->isInitialized);

    // Check if the blending scalar is valid
    if (blendingScalar > 1.0f) {
        printf("Error: Blending scalar is greater than 1.0\n");
        return 1;
    }

    // Calculate the blend scalar
    const float frameScalar12 = blendingScalar;
    const float frameScalar21 = 1.0f - blendingScalar;

    // Warp Frames
    int cz = 0; // Y-Plane
    cl_int err = clSetKernelArg(ofc->warpFrameKernel, 0, sizeof(cl_mem), &ofc->inputFrameArray[0]);
    err |= clSetKernelArg(ofc->warpFrameKernel, 1, sizeof(cl_mem), &ofc->inputFrameArray[1]);
    err |= clSetKernelArg(ofc->warpFrameKernel, 4, sizeof(float), &frameScalar12);
    err |= clSetKernelArg(ofc->warpFrameKernel, 5, sizeof(float), &frameScalar21);
    err |= clSetKernelArg(ofc->warpFrameKernel, 12, sizeof(int), &frameOutputMode);
    err |= clSetKernelArg(ofc->warpFrameKernel, 13, sizeof(float), &ofc->outputBlackLevel);
    err |= clSetKernelArg(ofc->warpFrameKernel, 14, sizeof(float), &ofc->outputWhiteLevel);
    err |= clSetKernelArg(ofc->warpFrameKernel, 15, sizeof(int), &cz);
    CHECK_ERROR(err);
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->warpFrameKernel, 2, NULL, ofc->grid16x16x1, ofc->threads16x16x1, 0, NULL, &ofc->warpStartedEvent));
    cz = 1; // UV-Plane
    CHECK_ERROR(clSetKernelArg(ofc->warpFrameKernel, 15, sizeof(int), &cz));
    CHECK_ERROR(clEnqueueNDRangeKernel(ofc->queue, ofc->warpFrameKernel, 2, NULL, ofc->halfGrid16x16x1, ofc->threads16x16x1, 0, NULL, NULL));
    return 0;
}

bool adjustSearchRadius(struct OpticalFlowCalc* ofc, int newSearchRadius) {
    CHECK_ERROR(!ofc->isInitialized);
    ofc->lowGrid16x16xL[2] = newSearchRadius;
    cl_int err = clSetKernelArg(ofc->calcDeltaSumsKernel, 10, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->determineLowestLayerKernel, 3, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 4, sizeof(int), &newSearchRadius);
    CHECK_ERROR(err);
    return 0;
}

void freeOFC(struct OpticalFlowCalc* ofc) {
    clFinish(ofc->queue);
    clReleaseMemObject(ofc->inputFrameArray[0]);
    clReleaseMemObject(ofc->inputFrameArray[1]);
    clReleaseMemObject(ofc->outputFrameArray);
    clReleaseMemObject(ofc->offsetArray);
    clReleaseMemObject(ofc->blurredOffsetArray);
    clReleaseMemObject(ofc->summedDeltaValuesArray);
    clReleaseMemObject(ofc->lowestLayerArray);
    clReleaseKernel(ofc->calcDeltaSumsKernel);
    clReleaseKernel(ofc->determineLowestLayerKernel);
    clReleaseKernel(ofc->adjustOffsetArrayKernel);
    clReleaseKernel(ofc->blurFlowKernel);
    clReleaseKernel(ofc->warpFrameKernel);
    clReleaseCommandQueue(ofc->queue);
    clReleaseContext(ofc->clContext);
    clReleaseDevice(ofc->clDeviceId);
}

// Detects the OpenCL platforms and devices
static bool detectDevices(struct OpticalFlowCalc* ofc) {
    // Capabilities we are going to check for
    cl_ulong availableVRAM;
    const cl_ulong requiredVRAM = 4.5 * ofc->frameHeight * ofc->frameWidth +
                                  4lu * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short) +
                                  MAX_SEARCH_RADIUS * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(unsigned int) +
                                  ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth;
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
    // Set up variables
    ofc->frameWidth = frameWidth;
    ofc->frameHeight = frameHeight;
    ofc->outputBlackLevel = 0.0f;
    ofc->outputWhiteLevel = 255.0f;
    ofc->opticalFlowSearchRadius = MIN_SEARCH_RADIUS;
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
    ofc->lowGrid16x16xL[0] = ceil(ofc->opticalFlowFrameWidth / 16.0) * 16.0;
    ofc->lowGrid16x16xL[1] = ceil(ofc->opticalFlowFrameHeight / 16.0) * 16.0;
    ofc->lowGrid16x16xL[2] = ofc->opticalFlowSearchRadius;
    ofc->lowGrid16x16x2[0] = ceil(ofc->opticalFlowFrameWidth / 16.0) * 16.0;
    ofc->lowGrid16x16x2[1] = ceil(ofc->opticalFlowFrameHeight / 16.0) * 16.0;
    ofc->lowGrid16x16x2[2] = 2;
    ofc->lowGrid16x16x1[0] = ceil(ofc->opticalFlowFrameWidth / 16.0) * 16.0;
    ofc->lowGrid16x16x1[1] = ceil(ofc->opticalFlowFrameHeight / 16.0) * 16.0;
    ofc->lowGrid16x16x1[2] = 1;
    ofc->halfGrid16x16x1[0] = ceil(ofc->frameWidth / 16.0) * 16.0;
    ofc->halfGrid16x16x1[1] = ceil((ofc->frameHeight >> 1) / 16.0) * 16.0;
    ofc->halfGrid16x16x1[2] = 1;
    ofc->grid16x16x1[0] = ceil(ofc->frameWidth / 16.0) * 16.0;
    ofc->grid16x16x1[1] = ceil(ofc->frameHeight / 16.0) * 16.0;
    ofc->grid16x16x1[2] = 1;

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
    ofc->outputFrameArray = clCreateBuffer(ofc->clContext, CL_MEM_WRITE_ONLY, 1.5 * frameHeight * frameWidth, NULL, &err);
    ofc->offsetArray = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), NULL, &err);
    ofc->blurredOffsetArray = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, 2 * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(short), NULL, &err);
    ofc->summedDeltaValuesArray = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, MAX_SEARCH_RADIUS * ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth * sizeof(unsigned int), NULL, &err);
    ofc->lowestLayerArray = clCreateBuffer(ofc->clContext, CL_MEM_READ_WRITE, ofc->opticalFlowFrameHeight * ofc->opticalFlowFrameWidth, NULL, &err);
    CHECK_ERROR(err);

    // Compile the kernels
    CHECK_ERROR(cl_create_kernel(&ofc->calcDeltaSumsKernel, ofc->clContext, ofc->clDeviceId, "calcDeltaSumsKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->determineLowestLayerKernel, ofc->clContext, ofc->clDeviceId, "determineLowestLayerKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->adjustOffsetArrayKernel, ofc->clContext, ofc->clDeviceId, "adjustOffsetArrayKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->blurFlowKernel, ofc->clContext, ofc->clDeviceId, "blurFlowKernel"));
    CHECK_ERROR(cl_create_kernel(&ofc->warpFrameKernel, ofc->clContext, ofc->clDeviceId, "warpFrameKernel"));

    // Set kernel arguments
    err = clSetKernelArg(ofc->calcDeltaSumsKernel, 0, sizeof(cl_mem), &ofc->summedDeltaValuesArray);
    err |= clSetKernelArg(ofc->calcDeltaSumsKernel, 3, sizeof(cl_mem), &ofc->offsetArray);
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
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 0, sizeof(cl_mem), &ofc->offsetArray);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 1, sizeof(cl_mem), &ofc->lowestLayerArray);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 3, sizeof(int), &directionIndexOffset);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 4, sizeof(int), &ofc->opticalFlowSearchRadius);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 5, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->adjustOffsetArrayKernel, 6, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 2, sizeof(cl_mem), &ofc->blurredOffsetArray);
    err |= clSetKernelArg(ofc->warpFrameKernel, 3, sizeof(cl_mem), &ofc->outputFrameArray);
    err |= clSetKernelArg(ofc->warpFrameKernel, 6, sizeof(int), &ofc->opticalFlowFrameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 7, sizeof(int), &ofc->frameHeight);
    err |= clSetKernelArg(ofc->warpFrameKernel, 8, sizeof(int), &ofc->frameWidth);
    err |= clSetKernelArg(ofc->warpFrameKernel, 9, sizeof(int), &ofc->opticalFlowResScalar);
    err |= clSetKernelArg(ofc->warpFrameKernel, 10, sizeof(int), &directionIndexOffset);
    err |= clSetKernelArg(ofc->warpFrameKernel, 11, sizeof(int), &channelIndexOffset);
    err |= clSetKernelArg(ofc->blurFlowKernel, 0, sizeof(cl_mem), &ofc->offsetArray);
    err |= clSetKernelArg(ofc->blurFlowKernel, 1, sizeof(cl_mem), &ofc->blurredOffsetArray);
    err |= clSetKernelArg(ofc->blurFlowKernel, 2, sizeof(int), &ofc->opticalFlowFrameHeight);
    err |= clSetKernelArg(ofc->blurFlowKernel, 3, sizeof(int), &ofc->opticalFlowFrameWidth);
    CHECK_ERROR(err);

    ofc->isInitialized = true;
    return 0;
}