#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include <sys/stat.h>
#include "opticalFlowCalc.h"
#include "config.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define ERR_CHECK(cond) if (cond) { return 1; }

// Function to wait for a command queue to finish and check for errors
static bool cl_finish_queue(cl_command_queue queue, const char* function_name) {
    cl_int err = clFinish(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error finishing queue in function: %s - %d\n", function_name, err);
        return 1;
    }
    return 0;
}

// Function to launch an OpenCL kernel and check for errors
static bool cl_enqueue_kernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_size, const size_t* local_work_size, const char* function_name) {
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing kernel in function: %s - %d\n", function_name, err);
        return 1;
    }
    return 0;
}

// Function to copy an OpenCL device buffer to another device buffer and check for errors
static bool cl_copy_buffer(cl_command_queue queue, cl_mem src, size_t src_offset, cl_mem dst, size_t dst_offset, size_t size, const char* function_name) {
    cl_int err = clEnqueueCopyBuffer(queue, src, dst, src_offset, dst_offset, size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error copying buffer in function: %s - %d\n", function_name, err);
        return 1;
    }
    ERR_CHECK(cl_finish_queue(queue, function_name));
    return 0;
}

// Function to upload data to an OpenCL buffer and check for errors
static bool cl_upload_buffer(cl_command_queue queue, void* src, cl_mem dst, size_t offset, size_t size, const char* function_name) {
    cl_int err = clEnqueueWriteBuffer(queue, dst, CL_TRUE, offset, size, src, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error uploading buffer in function: %s - %d\n", function_name, err);
        return 1;
    }
    ERR_CHECK(cl_finish_queue(queue, function_name));
    return 0;
}

// Function to download data from an OpenCL buffer and check for errors
static bool cl_download_buffer(cl_command_queue queue, cl_mem src, void* dst, size_t offset, size_t size, const char* function_name) {
    cl_int err = clEnqueueReadBuffer(queue, src, CL_TRUE, offset, size, dst, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error downloading buffer in function: %s - %d\n", function_name, err);
        return 1;
    }
    ERR_CHECK(cl_finish_queue(queue, function_name));
    return 0;
}

// Function to zero an OpenCL buffer and check for errors
static bool cl_zero_buffer(cl_command_queue queue, cl_mem buffer, size_t size, size_t pattern_size, const char* function_name) {
    cl_uint zero = 0;
    cl_int err = clEnqueueFillBuffer(queue, buffer, &zero, pattern_size, 0, size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error zeroing buffer in function: %s - %d\n", function_name, err);
        return 1;
    }
    ERR_CHECK(cl_finish_queue(queue, function_name));
    return 0;
}

// Function to create an OpenCL buffer and check for errors
static bool cl_create_buffer(cl_mem* buffer, cl_context context, cl_mem_flags flags, size_t size, const char* buffer_name) {
    cl_int err;
    *buffer = clCreateBuffer(context, flags, size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer %s: %d\n", buffer_name, err);
        return 1;
    }
    return 0;
}

// Function to read the OpenCL kernel source code from a file
static const char* loadKernelSource(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open kernel file: %s\n", filename);
        return NULL;
    }

    // Get the file size
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the kernel source
    char* source = (char*)malloc(fileSize + 1); // +1 for null terminator
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
static bool cl_create_kernel(cl_kernel* kernel, cl_context context, cl_device_id device_id, const char* kernelFunc) {
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
    const char *kernelSourceFile = loadKernelSource(kernelSourcePath);
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
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
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
* Blurs a frame
*
* @param ofc: Pointer to the optical flow calculator
* @param frame: Pointer to the frame to blur
* @param blurredFrame: Pointer to the blurred frame
* @param directOutput: Whether to output the blurred frame directly
*/
bool blurFrameArray(struct OpticalFlowCalc *ofc, const cl_mem frame, cl_mem blurredFrame, const bool directOutput) {
	// Early exit if kernel size is too small to blur
	if (FRAME_BLUR_KERNEL_SIZE < 4) {
        ERR_CHECK(cl_copy_buffer(ofc->m_OFCQueue, frame, 0, blurredFrame, 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), "blurFrameArray"));
		return 0;
	}

	// Calculate grid and thread dimensions
	const size_t globalGrid[2] = {ceil(ofc->m_iDimX / 16.0) * 16.0, ceil(ofc->m_iDimY / 16.0) * 16.0};
	const size_t localGrid[2] = {16, 16};

    // Set the arguments to our compute kernel
    cl_int err = clSetKernelArg(ofc->m_blurFrameKernel, 0, sizeof(cl_mem), &frame);
    err |= clSetKernelArg(ofc->m_blurFrameKernel, 1, sizeof(cl_mem), &blurredFrame);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
 
    // Execute the kernel over the entire range of the data set  
    ERR_CHECK(cl_enqueue_kernel(ofc->m_OFCQueue, ofc->m_blurFrameKernel, 2, globalGrid, localGrid, "blurFrameArray"));
 
    // Wait for the command queue to be serviced before reading back results
    ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "blurFrameArray"));

	// Handle direct output if necessary
	if (directOutput) {
        ERR_CHECK(cl_copy_buffer(ofc->m_OFCQueue, blurredFrame, 0, ofc->m_outputFrame, 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), "blurFrameArray"));
        ERR_CHECK(cl_copy_buffer(ofc->m_OFCQueue, frame, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), ofc->m_outputFrame, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), 0.5 * (ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short)), "blurFrameArray"));
        ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "blurFrameArray"));
	}
    return 0;
}

/*
* Updates the frame arrays and blurs them if necessary
*
* @param ofc: Pointer to the optical flow calculator
* @param pInBuffer: Pointer to the input frame
* @param directOutput: Whether to output the blurred frame directly
*/
bool updateFrame(struct OpticalFlowCalc *ofc, unsigned char** pInBuffer, const bool directOutput) {
    ERR_CHECK(cl_upload_buffer(ofc->m_OFCQueue, pInBuffer[0], ofc->m_frame[0], 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), "updateFrame"));
    ERR_CHECK(cl_upload_buffer(ofc->m_OFCQueue, pInBuffer[1], ofc->m_frame[0], ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), (ofc->m_iDimY / 2) * ofc->m_iDimX * sizeof(unsigned short), "updateFrame"));
    ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "updateFrame"));
	
	// Blur the frame
	ERR_CHECK(blurFrameArray(ofc, ofc->m_frame[0], ofc->m_blurredFrame[0], directOutput));
	
	// Swap the frame buffers
	cl_mem temp1 = ofc->m_frame[0];
	ofc->m_frame[0] = ofc->m_frame[1];
	ofc->m_frame[1] = ofc->m_frame[2];
	ofc->m_frame[2] = temp1;

	temp1 = ofc->m_blurredFrame[0];
	ofc->m_blurredFrame[0] = ofc->m_blurredFrame[1];
	ofc->m_blurredFrame[1] = ofc->m_blurredFrame[2];
	ofc->m_blurredFrame[2] = temp1;
    return 0;
}

/*
* Downloads the output frame from the GPU to the CPU
*
* @param ofc: Pointer to the optical flow calculator
* @param pInBuffer: Pointer to the input buffer
* @param pOutBuffer: Pointer to the output buffer
*/
bool downloadFrame(struct OpticalFlowCalc *ofc, const cl_mem pInBuffer, unsigned char** pOutBuffer) {
    ERR_CHECK(cl_download_buffer(ofc->m_WarpQueue1, pInBuffer, pOutBuffer[0], 0, ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), "downloadFrame"));
    ERR_CHECK(cl_download_buffer(ofc->m_WarpQueue1, pInBuffer, pOutBuffer[1], ofc->m_iDimY * ofc->m_iDimX * sizeof(unsigned short), (ofc->m_iDimY >> 1) * ofc->m_iDimX * sizeof(unsigned short), "downloadFrame"));
    return 0;
}

/*
* Copies the frame in the correct format to the output buffer
*
* @param ofc: Pointer to the optical flow calculator
* @param pOutBuffer: Pointer to the output frame
* @param frameCounter: The current source frame counter
*
* @note: Calling this function repeatedly starting from the first frame (i.e. frameCounter == 1) will result in the following frame output:
* 	     - Frame 1
*		 - Frame 1
*		 - Frame 2
*		 - Frame 3
* 	     - Frame 4
*		 - ...
*/
bool processFrame(struct OpticalFlowCalc *ofc, unsigned char** pOutBuffer, const int frameCounter) {
	//struct priv *priv = (struct priv*)ofc->priv;

	// First frame is directly output, therefore we don't change the output buffer
	if (frameCounter == 1) {
		return 0;
	// After the first frame, we always output the previous frame
	} else {
		ERR_CHECK(downloadFrame(ofc, ofc->m_frame[1], pOutBuffer));
	}
    return 0;

/* 	if (ofc->m_fBlackLevel == 0.0f && ofc->m_fWhiteLevel == 1023.0f) {
		HIP_CHECK(hipMemcpy(ofc->m_outputFrame, firstFrame ? ofc->m_frame[2] : ofc->m_frame[1], ofc->m_iDimY * ofc->m_iDimX * 3, hipMemcpyDeviceToDevice));
		downloadFrame(ofc, pOutBuffer);
	} else {
		processFrameKernel<<<priv->m_grid16x16x1, priv->m_threads16x16x1, 0, priv->m_csWarpStream1>>>(firstFrame ? ofc->m_frame[2] : ofc->m_frame[1],
												ofc->m_outputFrame,
												ofc->m_iDimY, ofc->m_iDimX, ofc->m_fBlackLevel, ofc->m_fWhiteLevel, ofc->m_fMaxVal);
		HIP_CHECK(hipStreamSynchronize(priv->m_csWarpStream1));
		HIP_CHECK(hipMemcpy(ofc->m_outputFrame + ofc->m_iChannelIdxOffset, (firstFrame ? ofc->m_frame[2] : ofc->m_frame[1]) + ofc->m_iChannelIdxOffset, ofc->m_iDimY * (ofc->m_iDimX >> 1) * sizeof(unsigned short), hipMemcpyDeviceToDevice));
		downloadFrame(ofc, pOutBuffer);
	} */
}

/*
* Adjusts the search radius of the optical flow calculation
*
* @param ofc: Pointer to the optical flow calculator
* @param newSearchRadius: The new search radius
*/
bool adjustSearchRadius(struct OpticalFlowCalc *ofc, int newSearchRadius) {
    int newNumLayers = newSearchRadius * newSearchRadius;
	ofc->m_lowGrid16x16xL[2] = newNumLayers;
	ofc->m_lowGrid8x8xL[2] = newNumLayers;
    cl_int err = clSetKernelArg(ofc->m_setInitialOffsetKernel, 1, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->m_determineLowestLayerKernel, 3, sizeof(int), &newNumLayers);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 5, sizeof(int), &newSearchRadius);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 6, sizeof(int), &newNumLayers);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    return 0;
}

/*
* Calculates the optical flow between frame1 and frame2
*
* @param ofc: Pointer to the optical flow calculator
* @param iNumIterations: Number of iterations to calculate the optical flow
*/
bool calculateOpticalFlow(struct OpticalFlowCalc *ofc, int iNumIterations) {
    // Swap the blurred offset arrays
    cl_mem temp0 = ofc->m_blurredOffsetArray12[0];
    ofc->m_blurredOffsetArray12[0] = ofc->m_blurredOffsetArray12[1];
    ofc->m_blurredOffsetArray12[1] = temp0;

    temp0 = ofc->m_blurredOffsetArray21[0];
    ofc->m_blurredOffsetArray21[0] = ofc->m_blurredOffsetArray21[1];
    ofc->m_blurredOffsetArray21[1] = temp0;

	// We set the initial window size to the next larger power of 2
	int windowDim = 1;
	int maxDim = max(ofc->m_iLowDimX, ofc->m_iLowDimY);
    if (maxDim && !(maxDim & (maxDim - 1))) {
		windowDim = maxDim;
	} else {
		while (maxDim & (maxDim - 1)) {
			maxDim &= (maxDim - 1);
		}
		windowDim = maxDim << 1;
	}

	if (iNumIterations == 0 || (double)(iNumIterations) > ceil(log2(windowDim))) {
		iNumIterations = (int)(ceil(log2(windowDim))) + 1;
	}

    // Reset the number of offset layers
    int currSearchRadius = ofc->m_iSearchRadius;
    ERR_CHECK(adjustSearchRadius(ofc, ofc->m_iSearchRadius));

	// Prepare the initial offset array
    ERR_CHECK(cl_enqueue_kernel(ofc->m_OFCQueue, ofc->m_setInitialOffsetKernel, 3, ofc->m_lowGrid16x16xL, ofc->m_threads16x16x1, "calculateOpticalFlow1"));
    ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "calculateOpticalFlow1"));

	// We calculate the ideal offset array for each window size (entire frame, ..., individual pixels)
	for (int iter = 0; iter < iNumIterations; iter++) {
		if (ofc->m_bOFCTerminate) return 0;
        
		// Reset the summed up delta array
        ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_summedUpDeltaArray, currSearchRadius * currSearchRadius * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(unsigned int), sizeof(unsigned int), "calculateOpticalFlow2"));
        ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "calculateOpticalFlow4"));

		// 1. Calculate the image delta and sum up the deltas of each window
        cl_int err = clSetKernelArg(ofc->m_calcDeltaSumsKernel, 1, sizeof(cl_mem), &ofc->m_blurredFrame[1]);
        err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 2, sizeof(cl_mem), &ofc->m_blurredFrame[2]);
        err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 9, sizeof(int), &windowDim);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_OFCQueue, ofc->m_calcDeltaSumsKernel, 3, ofc->m_lowGrid8x8xL, ofc->m_threads8x8x1, "calculateOpticalFlow3"));
        ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "calculateOpticalFlow4"));

		// 2. Find the layer with the lowest delta sum
        err = clSetKernelArg(ofc->m_determineLowestLayerKernel, 2, sizeof(int), &windowDim);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_OFCQueue, ofc->m_determineLowestLayerKernel, 2, ofc->m_lowGrid16x16x1, ofc->m_threads16x16x1, "calculateOpticalFlow7"));
        ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "calculateOpticalFlow8"));

		// 3. Adjust the offset array based on the comparison results
        err = clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 2, sizeof(int), &windowDim);
        const int lastRun = (int)(iter == iNumIterations - 1);
        err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 9, sizeof(int), &lastRun);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_OFCQueue, ofc->m_adjustOffsetArrayKernel, 2, ofc->m_lowGrid16x16x1, ofc->m_threads16x16x1, "calculateOpticalFlow9"));
		ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "calculateOpticalFlow10"));
	
		// 4. Adjust variables for the next iteration
		windowDim = max(windowDim >> 1, (int)1);
        if (!lastRun) {
            currSearchRadius = max(ofc->m_iSearchRadius - iter, 5);
            ERR_CHECK(adjustSearchRadius(ofc, currSearchRadius));
        }
	}
    return 0;
}

/*
* Warps the frames according to the calculated optical flow
*
* @param ofc: Pointer to the optical flow calculator
* @param fScalar: The scalar to blend the frames with
* @param outputMode: The mode to output the frames in (0: WarpedFrame 1->2, 1: WarpedFrame 2->1, 2: Both for blending)
*/
bool warpFrames(struct OpticalFlowCalc *ofc, const float fScalar, const int outputMode) {
	// Calculate the blend scalar
	const float frameScalar12 = fScalar;
	const float frameScalar21 = 1.0f - fScalar;

	// Reset the hit count array
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_hitCount12, ofc->m_iDimY * ofc->m_iDimX * sizeof(int), sizeof(int), "warpFrames"));
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_hitCount21, ofc->m_iDimY * ofc->m_iDimX * sizeof(int), sizeof(int), "warpFrames"));

	// #####################
	// ###### WARPING ######
	// #####################
	// Frame 1 to Frame 2
    if (outputMode != 1) {
        cl_int err = clSetKernelArg(ofc->m_warpFrameKernel, 0, sizeof(cl_mem), &ofc->m_frame[0]);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 1, sizeof(cl_mem), &ofc->m_blurredOffsetArray12[0]);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 2, sizeof(cl_mem), &ofc->m_hitCount12);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 3, sizeof(cl_mem), (outputMode < 2) ? &ofc->m_outputFrame : &ofc->m_warpedFrame12);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 4, sizeof(float), &frameScalar12);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_warpFrameKernel, 3, ofc->m_grid16x16x2, ofc->m_threads16x16x1, "warpFrames"));
    }

	// Frame 2 to Frame 1
	if (outputMode != 0) {
        cl_int err = clSetKernelArg(ofc->m_warpFrameKernel, 0, sizeof(cl_mem), &ofc->m_frame[1]);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 1, sizeof(cl_mem), &ofc->m_blurredOffsetArray21[0]);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 2, sizeof(cl_mem), &ofc->m_hitCount21);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 3, sizeof(cl_mem), (outputMode < 2) ? &ofc->m_outputFrame : &ofc->m_warpedFrame21);
        err |= clSetKernelArg(ofc->m_warpFrameKernel, 4, sizeof(float), &frameScalar21);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue2, ofc->m_warpFrameKernel, 3, ofc->m_grid16x16x2, ofc->m_threads16x16x1, "warpFrames"));
	}
	if (outputMode != 1) ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "warpFrames"));
	if (outputMode != 0) ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue2, "warpFrames"));
	
	// ##############################
	// ###### ARTIFACT REMOVAL ######
	// ##############################
	// Frame 1 to Frame 2
	if (outputMode != 1) {
        cl_int err = clSetKernelArg(ofc->m_artifactRemovalKernel, 0, sizeof(cl_mem), &ofc->m_frame[0]);
        err |= clSetKernelArg(ofc->m_artifactRemovalKernel, 1, sizeof(cl_mem), &ofc->m_hitCount12);
        err |= clSetKernelArg(ofc->m_artifactRemovalKernel, 2, sizeof(cl_mem), (outputMode < 2) ? &ofc->m_outputFrame : &ofc->m_warpedFrame12);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_artifactRemovalKernel, 3, ofc->m_grid16x16x2, ofc->m_threads8x8x1, "warpFrames"));
	}
	// Frame 2 to Frame 1
	if (outputMode != 0) {
        cl_int err = clSetKernelArg(ofc->m_artifactRemovalKernel, 0, sizeof(cl_mem), &ofc->m_frame[1]);
        err |= clSetKernelArg(ofc->m_artifactRemovalKernel, 1, sizeof(cl_mem), &ofc->m_hitCount21);
        err |= clSetKernelArg(ofc->m_artifactRemovalKernel, 2, sizeof(cl_mem), (outputMode < 2) ? &ofc->m_outputFrame : &ofc->m_warpedFrame21);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue2, ofc->m_artifactRemovalKernel, 3, ofc->m_grid16x16x2, ofc->m_threads8x8x1, "warpFrames"));
	}
	if (outputMode != 1) ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "warpFrames"));
	if (outputMode != 0) ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue2, "warpFrames"));
    return 0;
}

/*
* Blends warpedFrame1 to warpedFrame2
*
* @param ofc: Pointer to the optical flow calculator
* @param fScalar: The scalar to blend the frames with
*/
bool blendFrames(struct OpticalFlowCalc *ofc, const float fScalar) {
	// Calculate the blend scalar
	const float frame1Scalar = 1.0f - fScalar;
	const float frame2Scalar = fScalar;

	// Blend the frames
    cl_int err = clSetKernelArg(ofc->m_blendFrameKernel, 3, sizeof(float), &frame1Scalar);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 4, sizeof(float), &frame2Scalar);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 8, sizeof(float), &ofc->m_fBlackLevel);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 9, sizeof(float), &ofc->m_fWhiteLevel);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 10, sizeof(float), &ofc->m_fMaxVal);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_blendFrameKernel, 3, ofc->m_grid16x16x2, ofc->m_threads16x16x1, "blendFrames"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "blendFrames"));
    return 0;
}

/*
* Places left half of frame1 over the outputFrame
*
* @param ofc: Pointer to the optical flow calculator
*/
bool insertFrame(struct OpticalFlowCalc *ofc) {
    cl_int err = clSetKernelArg(ofc->m_insertFrameKernel, 0, sizeof(cl_mem), &ofc->m_frame[0]);
    err |= clSetKernelArg(ofc->m_insertFrameKernel, 5, sizeof(float), &ofc->m_fBlackLevel);
    err |= clSetKernelArg(ofc->m_insertFrameKernel, 6, sizeof(float), &ofc->m_fWhiteLevel);
    err |= clSetKernelArg(ofc->m_insertFrameKernel, 7, sizeof(float), &ofc->m_fMaxVal);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_insertFrameKernel, 3, ofc->m_halfGrid16x16x2, ofc->m_threads16x16x1, "insertFrames"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "insertFrames"));
    return 0;
}

/*
* Places frame 1 scaled down on the left side and the blendedFrame on the right side of the outputFrame
*
* @param ofc: Pointer to the optical flow calculator
* @param dScalar: The scalar to blend the frames with
* @param frameCounter: The current frame counter
*/
bool sideBySideFrame(struct OpticalFlowCalc *ofc, const float fScalar, const int frameCounter) {
	// Calculate the blend scalar
	const float frame1Scalar = 1.0f - fScalar;
	const float frame2Scalar = fScalar;
    cl_int err = 0;

	if (frameCounter == 1) {
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 0, sizeof(cl_mem), &ofc->m_frame[2]);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 1, sizeof(cl_mem), &ofc->m_frame[2]);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 2, sizeof(cl_mem), &ofc->m_frame[2]);
	} else if (frameCounter == 2) {
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 0, sizeof(cl_mem), &ofc->m_frame[1]);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 1, sizeof(cl_mem), &ofc->m_frame[1]);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 2, sizeof(cl_mem), &ofc->m_frame[1]);
	} else if (frameCounter <= 3) {
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 0, sizeof(cl_mem), &ofc->m_frame[0]);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 1, sizeof(cl_mem), &ofc->m_frame[0]);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 2, sizeof(cl_mem), &ofc->m_frame[0]);
	} else {
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 0, sizeof(cl_mem), &ofc->m_frame[0]);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 1, sizeof(cl_mem), &ofc->m_warpedFrame12);
        err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 2, sizeof(cl_mem), &ofc->m_warpedFrame21);
	}
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 4, sizeof(float), &frame1Scalar);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 5, sizeof(float), &frame2Scalar);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 11, sizeof(float), &ofc->m_fBlackLevel);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 12, sizeof(float), &ofc->m_fWhiteLevel);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 13, sizeof(float), &ofc->m_fMaxVal);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 14, sizeof(int), &ofc->m_iMiddleValue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_sideBySideFrameKernel, 3, ofc->m_grid16x16x2, ofc->m_threads16x16x1, "sideBySideFrame"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "sideBySideFrame"));
    return 0;
}

/*
* Draws the flow as an RGB image
*
* @param ofc: Pointer to the optical flow calculator
* @param blendScalar: The scalar that determines how much of the source frame is blended with the flow
*/
bool drawFlowAsHSV(struct OpticalFlowCalc *ofc, const float blendScalar) {
    cl_int err = clSetKernelArg(ofc->m_convertFlowToHSVKernel, 0, sizeof(cl_mem), &ofc->m_blurredOffsetArray12[0]);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 2, sizeof(cl_mem), &ofc->m_frame[1]);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 3, sizeof(float), &blendScalar);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_convertFlowToHSVKernel, 3, ofc->m_grid16x16x2, ofc->m_threads16x16x1, "drawFlowAsHSV"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "drawFlowAsHSV"));
    return 0;
}

/*
* Draws the flow as an grayscale image
*
* @param ofc: Pointer to the optical flow calculator
*/
bool drawFlowAsGrayscale(struct OpticalFlowCalc *ofc) {
    cl_int err = clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 0, sizeof(cl_mem), &ofc->m_blurredOffsetArray12[0]);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 2, sizeof(cl_mem), &ofc->m_frame[1]);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 10, sizeof(float), &ofc->m_fMaxVal);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 11, sizeof(int), &ofc->m_iMiddleValue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_convertFlowToGrayscaleKernel, 3, ofc->m_grid16x16x2, ofc->m_threads16x16x1, "drawFlowAsGrayscale"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "drawFlowAsGrayscale"));
    return 0;
}

/*
* Translates a flow array from frame 1 to frame 2 into a flow array from frame 2 to frame 1
*
* @param ofc: Pointer to the optical flow calculator
*/
bool flipFlow(struct OpticalFlowCalc *ofc) {
	if (ofc->m_bOFCTerminate) return 0;

	// Reset the offset array
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_offsetArray21, ofc->m_iLowDimY * ofc->m_iLowDimX * 2, sizeof(char), "flipFlow"));
    ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "flipFlow"));

	// Launch kernel
    ERR_CHECK(cl_enqueue_kernel(ofc->m_OFCQueue, ofc->m_flipFlowKernel, 3, ofc->m_lowGrid16x16x2, ofc->m_threads16x16x1, "flipFlow"));
    ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "flipFlow"));
    return 0;
}

/*
* Blurs the offset arrays
*
* @param ofc: Pointer to the optical flow calculator
*/
bool blurFlowArrays(struct OpticalFlowCalc *ofc) {
	if (ofc->m_bOFCTerminate) return 0;

	// Calculate the number of blocks needed
	const size_t globalGrid[3] = {ceil(ofc->m_iLowDimX / 16.0) * 16.0, ceil(ofc->m_iLowDimY / 16.0) * 16.0, 4};
	const size_t localGrid[3] = {16, 16, 1};

	// No need to blur the flow if the kernel size is less than 4
	if (FLOW_BLUR_KERNEL_SIZE < 4) {
		// Offset12 X-Dir
        ERR_CHECK(cl_copy_buffer(ofc->m_OFCQueue, ofc->m_offsetArray12, 0, ofc->m_blurredOffsetArray12[1], 0, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "blurFlowArrays"));
		// Offset21 X&Y-Dir
        ERR_CHECK(cl_copy_buffer(ofc->m_OFCQueue, ofc->m_offsetArray21, 0, ofc->m_blurredOffsetArray21[1], 0, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "blurFlowArrays"));
	} else {
		// Launch kernels
        cl_int err = clSetKernelArg(ofc->m_blurFlowKernel, 2, sizeof(cl_mem), &ofc->m_blurredOffsetArray12[1]);
        err |= clSetKernelArg(ofc->m_blurFlowKernel, 3, sizeof(cl_mem), &ofc->m_blurredOffsetArray21[1]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Unable to set kernel arguments\n");
            return 1;
        }
        ERR_CHECK(cl_enqueue_kernel(ofc->m_OFCQueue, ofc->m_blurFlowKernel, 3, globalGrid, localGrid, "blurFlowArrays"));
        
        ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "blurFlowArrays"));
	}
    return 0;
}

/*
* Saves an image to a file
*
* @param ofc: Pointer to the optical flow calculator
* @param filePath: Path to the image file
*/
bool saveImage(struct OpticalFlowCalc *ofc, const char* filePath) {
	// Copy the image array to the CPU
	size_t dataSize = 1.5 * ofc->m_iDimY * ofc->m_iDimX;
    ERR_CHECK(cl_download_buffer(ofc->m_OFCQueue, ofc->m_outputFrame, ofc->m_imageArrayCPU, 0, dataSize * sizeof(unsigned short), "saveImage"));

	// Open file in binary write mode
    FILE *file = fopen(filePath, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Write the array to the file
    size_t written = fwrite(ofc->m_imageArrayCPU, sizeof(unsigned short), dataSize, file);
    if (written != dataSize) {
        perror("Error writing to file");
        fclose(file);
        return 1;
    }

    // Close the file
    fclose(file);

    return 0;
}

/*
* Runs a tearing test on the GPU
*
* @param ofc: Pointer to the optical flow calculator
*/
static int counter = 0;
bool tearingTest(struct OpticalFlowCalc *ofc) {
	ERR_CHECK(cl_zero_buffer(ofc->m_WarpQueue1, ofc->m_outputFrame, ofc->m_iDimY * ofc->m_iDimX * 1.5 * sizeof(unsigned short), sizeof(unsigned short), "tearingTest"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "tearingTest"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue2, "tearingTest"));
    const int pos_x = counter % ofc->m_iDimX;
    cl_int err = clSetKernelArg(ofc->m_tearingTestKernel, 4, sizeof(int), &pos_x);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    ERR_CHECK(cl_enqueue_kernel(ofc->m_WarpQueue1, ofc->m_tearingTestKernel, 2, ofc->m_grid16x16x1, ofc->m_threads16x16x1, "tearingTest"));
    ERR_CHECK(cl_finish_queue(ofc->m_WarpQueue1, "tearingTest"));
	counter++;
    return 0;
}

/*
* Sets the kernel parameters and other kernel related variables used in the optical flow calculation
*
* @param ofc: Pointer to the optical flow calculator
*/
bool setKernelParameters(struct OpticalFlowCalc *ofc) {
    const int numLayers = ofc->m_iSearchRadius * ofc->m_iSearchRadius;

    // Define the global and local work sizes
	ofc->m_lowGrid16x16xL[0] = ceil(ofc->m_iLowDimX / 16.0) * 16.0;
	ofc->m_lowGrid16x16xL[1] = ceil(ofc->m_iLowDimY / 16.0) * 16.0;
	ofc->m_lowGrid16x16xL[2] = numLayers;
    ofc->m_lowGrid16x16x2[0] = ceil(ofc->m_iLowDimX / 16.0) * 16.0;
	ofc->m_lowGrid16x16x2[1] = ceil(ofc->m_iLowDimY / 16.0) * 16.0;
    ofc->m_lowGrid16x16x2[2] = 2;
	ofc->m_lowGrid16x16x1[0] = ceil(ofc->m_iLowDimX / 16.0) * 16.0;
	ofc->m_lowGrid16x16x1[1] = ceil(ofc->m_iLowDimY / 16.0) * 16.0;
    ofc->m_lowGrid16x16x1[2] = 1;
	ofc->m_lowGrid8x8xL[0] = ceil(ofc->m_iLowDimX / 8.0) * 8.0;
	ofc->m_lowGrid8x8xL[1] = ceil(ofc->m_iLowDimY / 8.0) * 8.0;
	ofc->m_lowGrid8x8xL[2] = numLayers;
	ofc->m_grid16x16x2[0] = ceil(ofc->m_iDimX / 16.0) * 16.0;
	ofc->m_grid16x16x2[1] = ceil(ofc->m_iDimY / 16.0) * 16.0;
    ofc->m_grid16x16x2[2] = 2;
    ofc->m_grid16x16x1[0] = ceil(ofc->m_iDimX / 16.0) * 16.0;
	ofc->m_grid16x16x1[1] = ceil(ofc->m_iDimY / 16.0) * 16.0;
    ofc->m_grid16x16x1[2] = 1;
    ofc->m_halfGrid16x16x2[0] = ceil((ofc->m_iDimX / 2) / 16.0) * 16.0;
	ofc->m_halfGrid16x16x2[1] = ceil(ofc->m_iDimY / 16.0) * 16.0;
    ofc->m_halfGrid16x16x2[2] = 2;

	ofc->m_threads16x16x1[0] = 16;
	ofc->m_threads16x16x1[1] = 16;
    ofc->m_threads16x16x1[2] = 1;
	ofc->m_threads8x8x1[0] = 8;
	ofc->m_threads8x8x1[1] = 8;
    ofc->m_threads8x8x1[2] = 1;

	// Clear the flow arrays
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_offsetArray12, numLayers * 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, sizeof(char), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_offsetArray21, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, sizeof(char), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_blurredOffsetArray12[0], 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, sizeof(char), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_blurredOffsetArray21[0], 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, sizeof(char), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_blurredOffsetArray12[1], 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, sizeof(char), "reinit"));
    ERR_CHECK(cl_zero_buffer(ofc->m_OFCQueue, ofc->m_blurredOffsetArray21[1], 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, sizeof(char), "reinit"));
    ERR_CHECK(cl_finish_queue(ofc->m_OFCQueue, "reinit"));

    // Set kernel arguments
    cl_int err = clSetKernelArg(ofc->m_blurFrameKernel, 2, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_blurFrameKernel, 3, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_setInitialOffsetKernel, 0, sizeof(cl_mem), &ofc->m_offsetArray12);
    err |= clSetKernelArg(ofc->m_setInitialOffsetKernel, 1, sizeof(int), &ofc->m_iSearchRadius);
    err |= clSetKernelArg(ofc->m_setInitialOffsetKernel, 2, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_setInitialOffsetKernel, 3, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_setInitialOffsetKernel, 4, sizeof(int), &ofc->m_iDirectionIdxOffset);
    err |= clSetKernelArg(ofc->m_setInitialOffsetKernel, 5, sizeof(int), &ofc->m_iLayerIdxOffset);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 0, sizeof(cl_mem), &ofc->m_summedUpDeltaArray);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 3, sizeof(cl_mem), &ofc->m_offsetArray12);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 4, sizeof(int), &ofc->m_iDirectionIdxOffset);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 5, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 6, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 7, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 8, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_calcDeltaSumsKernel, 10, sizeof(int), &ofc->m_cResolutionScalar);
    err |= clSetKernelArg(ofc->m_determineLowestLayerKernel, 0, sizeof(cl_mem), &ofc->m_summedUpDeltaArray);
    err |= clSetKernelArg(ofc->m_determineLowestLayerKernel, 1, sizeof(cl_mem), &ofc->m_lowestLayerArray);
    err |= clSetKernelArg(ofc->m_determineLowestLayerKernel, 3, sizeof(int), &numLayers);
    err |= clSetKernelArg(ofc->m_determineLowestLayerKernel, 4, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_determineLowestLayerKernel, 5, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 0, sizeof(cl_mem), &ofc->m_offsetArray12);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 1, sizeof(cl_mem), &ofc->m_lowestLayerArray);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 3, sizeof(int), &ofc->m_iDirectionIdxOffset);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 4, sizeof(int), &ofc->m_iLayerIdxOffset);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 5, sizeof(int), &ofc->m_iSearchRadius);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 6, sizeof(int), &numLayers);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 7, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_adjustOffsetArrayKernel, 8, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_warpFrameKernel, 5, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_warpFrameKernel, 6, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_warpFrameKernel, 7, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_warpFrameKernel, 8, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_warpFrameKernel, 9, sizeof(int), &ofc->m_cResolutionScalar);
    err |= clSetKernelArg(ofc->m_warpFrameKernel, 10, sizeof(int), &ofc->m_iDirectionIdxOffset);
    err |= clSetKernelArg(ofc->m_warpFrameKernel, 11, sizeof(int), &ofc->m_iChannelIdxOffset);
    err |= clSetKernelArg(ofc->m_artifactRemovalKernel, 3, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_artifactRemovalKernel, 4, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_artifactRemovalKernel, 5, sizeof(int), &ofc->m_iChannelIdxOffset);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 0, sizeof(cl_mem), &ofc->m_warpedFrame12);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 1, sizeof(cl_mem), &ofc->m_warpedFrame21);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 2, sizeof(cl_mem), &ofc->m_outputFrame);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 5, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 6, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_blendFrameKernel, 7, sizeof(int), &ofc->m_iChannelIdxOffset);
    err |= clSetKernelArg(ofc->m_insertFrameKernel, 1, sizeof(cl_mem), &ofc->m_outputFrame);
    err |= clSetKernelArg(ofc->m_insertFrameKernel, 2, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_insertFrameKernel, 3, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_insertFrameKernel, 4, sizeof(int), &ofc->m_iChannelIdxOffset);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 3, sizeof(cl_mem), &ofc->m_outputFrame);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 6, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 7, sizeof(int), &ofc->m_iDimX);
    int halfDimY = ofc->m_iDimY / 2;
    int halfDimX = ofc->m_iDimX / 2;
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 8, sizeof(int), &halfDimY);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 9, sizeof(int), &halfDimX);
    err |= clSetKernelArg(ofc->m_sideBySideFrameKernel, 10, sizeof(int), &ofc->m_iChannelIdxOffset);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 1, sizeof(cl_mem), &ofc->m_outputFrame);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 4, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 5, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 6, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 7, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 8, sizeof(int), &ofc->m_cResolutionScalar);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 9, sizeof(int), &ofc->m_iDirectionIdxOffset);
    err |= clSetKernelArg(ofc->m_convertFlowToHSVKernel, 10, sizeof(int), &ofc->m_iChannelIdxOffset);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 1, sizeof(cl_mem), &ofc->m_outputFrame);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 3, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 4, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 5, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 6, sizeof(int), &ofc->m_iDimX);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 7, sizeof(int), &ofc->m_cResolutionScalar);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 8, sizeof(int), &ofc->m_iDirectionIdxOffset);
    err |= clSetKernelArg(ofc->m_convertFlowToGrayscaleKernel, 9, sizeof(int), &ofc->m_iChannelIdxOffset);
    err |= clSetKernelArg(ofc->m_flipFlowKernel, 0, sizeof(cl_mem), &ofc->m_offsetArray12);
    err |= clSetKernelArg(ofc->m_flipFlowKernel, 1, sizeof(cl_mem), &ofc->m_offsetArray21);
    err |= clSetKernelArg(ofc->m_flipFlowKernel, 2, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_flipFlowKernel, 3, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_flipFlowKernel, 4, sizeof(int), &ofc->m_cResolutionScalar);
    err |= clSetKernelArg(ofc->m_flipFlowKernel, 5, sizeof(int), &ofc->m_iDirectionIdxOffset);
    err |= clSetKernelArg(ofc->m_blurFlowKernel, 0, sizeof(cl_mem), &ofc->m_offsetArray12);
    err |= clSetKernelArg(ofc->m_blurFlowKernel, 1, sizeof(cl_mem), &ofc->m_offsetArray21);
    err |= clSetKernelArg(ofc->m_blurFlowKernel, 4, sizeof(int), &ofc->m_iLowDimY);
    err |= clSetKernelArg(ofc->m_blurFlowKernel, 5, sizeof(int), &ofc->m_iLowDimX);
    err |= clSetKernelArg(ofc->m_tearingTestKernel, 0, sizeof(cl_mem), &ofc->m_outputFrame);
    err |= clSetKernelArg(ofc->m_tearingTestKernel, 1, sizeof(int), &ofc->m_iDimY);
    err |= clSetKernelArg(ofc->m_tearingTestKernel, 2, sizeof(int), &ofc->m_iDimX);
    const int width = 10;
    err |= clSetKernelArg(ofc->m_tearingTestKernel, 3, sizeof(int), &width);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to set kernel arguments\n");
        return 1;
    }
    return 0;
}

/*
* Frees the memory of the optical flow calculator
*
* @param ofc: Pointer to the optical flow calculator
*/
void freeOFC(struct OpticalFlowCalc *ofc) {
    // Free the GPU arrays
    clReleaseMemObject(ofc->m_frame[0]);
    clReleaseMemObject(ofc->m_frame[1]);
    clReleaseMemObject(ofc->m_frame[2]);
    clReleaseMemObject(ofc->m_blurredFrame[0]);
    clReleaseMemObject(ofc->m_blurredFrame[1]);
    clReleaseMemObject(ofc->m_blurredFrame[2]);
    clReleaseMemObject(ofc->m_warpedFrame12);
    clReleaseMemObject(ofc->m_warpedFrame21);
    clReleaseMemObject(ofc->m_outputFrame);
    clReleaseMemObject(ofc->m_offsetArray12);
    clReleaseMemObject(ofc->m_offsetArray21);
    clReleaseMemObject(ofc->m_blurredOffsetArray12[0]);
    clReleaseMemObject(ofc->m_blurredOffsetArray21[0]);
    clReleaseMemObject(ofc->m_blurredOffsetArray12[1]);
    clReleaseMemObject(ofc->m_blurredOffsetArray21[1]);
    clReleaseMemObject(ofc->m_summedUpDeltaArray);
    clReleaseMemObject(ofc->m_lowestLayerArray);
    clReleaseMemObject(ofc->m_hitCount12);
    clReleaseMemObject(ofc->m_hitCount21);
    free(ofc->m_imageArrayCPU);

    // Release the kernels
    clReleaseKernel(ofc->m_processFrameKernel);
    clReleaseKernel(ofc->m_blurFrameKernel);
    clReleaseKernel(ofc->m_setInitialOffsetKernel);
    clReleaseKernel(ofc->m_calcDeltaSumsKernel);
    clReleaseKernel(ofc->m_determineLowestLayerKernel);
    clReleaseKernel(ofc->m_adjustOffsetArrayKernel);
    clReleaseKernel(ofc->m_flipFlowKernel);
    clReleaseKernel(ofc->m_blurFlowKernel);
    clReleaseKernel(ofc->m_cleanFlowKernel);
    clReleaseKernel(ofc->m_warpFrameKernel);
    clReleaseKernel(ofc->m_artifactRemovalKernel);
    clReleaseKernel(ofc->m_blendFrameKernel);
    clReleaseKernel(ofc->m_insertFrameKernel);
    clReleaseKernel(ofc->m_sideBySideFrameKernel);
    clReleaseKernel(ofc->m_convertFlowToHSVKernel);
    clReleaseKernel(ofc->m_convertFlowToGrayscaleKernel);

    // Release the command queues
    clReleaseCommandQueue(ofc->m_OFCQueue);
    clReleaseCommandQueue(ofc->m_WarpQueue1);
    clReleaseCommandQueue(ofc->m_WarpQueue2);

    // Release the context
    clReleaseContext(ofc->m_clContext);

    // Release the device
    clReleaseDevice(ofc->m_clDevice_id);
}

// Detects the OpenCL platforms and devices
static bool detectDevices(struct OpticalFlowCalc *ofc) {
    // Capabilities we are going to check for
    cl_ulong availableVRAM;
    const int numLayers = ofc->m_iSearchRadius * ofc->m_iSearchRadius;
    const cl_ulong requiredVRAM = (32 * ofc->m_iDimY * ofc->m_iDimX) +
                                  ((numLayers + 5) * 2 * ofc->m_iLowDimY * ofc->m_iLowDimX) +
                                  (numLayers * ofc->m_iLowDimY * ofc->m_iLowDimX * 4) +
                                  (ofc->m_iLowDimY * ofc->m_iLowDimX * 2);
    printf("[HopperRender] Required VRAM: %lu MB\n", requiredVRAM / 1024 / 1024);
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
            if (availableVRAM >= requiredVRAM && 
                maxSharedMemSize >= requiredSharedMemSize && 
                maxWorkGroupSizes[0] >= requiredWorkGroupSizes[0] &&
                maxWorkGroupSizes[1] >= requiredWorkGroupSizes[1] &&
                maxWorkGroupSizes[2] >= requiredWorkGroupSizes[2]) {
                printf("[HopperRender] Using %s\n", deviceName);
                ofc->m_clDevice_id = devices[j];
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
        printf("Error: Not enough work group sizes available! Required: %lu, %lu, %lu, Available: %lu, %lu, %lu\n", requiredWorkGroupSizes[0], requiredWorkGroupSizes[1], requiredWorkGroupSizes[2], maxWorkGroupSizes[0], maxWorkGroupSizes[1], maxWorkGroupSizes[2]);
    }
    return 1;
}

/*
* Initializes the optical flow calculator
*
* @param ofc: Pointer to the optical flow calculator to be initialized
* @param dimY: The height of the frame
* @param dimX: The width of the frame
* @param resolutionScalar: The resolution scalar used for the optical flow calculation
* @param searchRadius: The search radius used for the optical flow calculation
*/
bool initOpticalFlowCalc(struct OpticalFlowCalc *ofc, const int dimY, const int dimX, const int resolutionScalar, const int searchRadius)
{
	// Video properties
	ofc->m_iDimX = dimX;
	ofc->m_iDimY = dimY;
	ofc->m_fBlackLevel = 0.0f;
	ofc->m_fMaxVal = 65535.0f;
	ofc->m_iMiddleValue = 32768;
	ofc->m_fWhiteLevel = ofc->m_fMaxVal;

	// Optical flow calculation
	ofc->m_cResolutionScalar = resolutionScalar;
	ofc->m_iLowDimX = dimX >> ofc->m_cResolutionScalar;
	ofc->m_iLowDimY = dimY >> ofc->m_cResolutionScalar;
	ofc->m_iSearchRadius = searchRadius;
	ofc->m_iDirectionIdxOffset = ofc->m_iLowDimY * ofc->m_iLowDimX;
	ofc->m_iLayerIdxOffset = 2 * ofc->m_iLowDimY * ofc->m_iLowDimX;
	ofc->m_iChannelIdxOffset = ofc->m_iDimY * ofc->m_iDimX;
	ofc->m_bOFCTerminate = false;

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
    ofc->m_clContext = clCreateContext(0, 1, &ofc->m_clDevice_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create context\n");
        return 1;
    }

    // Create the command queues
    ofc->m_OFCQueue = clCreateCommandQueueWithProperties(ofc->m_clContext, ofc->m_clDevice_id, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create command queue\n");
        return 1;
    }
    ofc->m_WarpQueue1 = clCreateCommandQueueWithProperties(ofc->m_clContext, ofc->m_clDevice_id, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create command queue\n");
        return 1;
    }
    ofc->m_WarpQueue2 = clCreateCommandQueueWithProperties(ofc->m_clContext, ofc->m_clDevice_id, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Unable to create command queue\n");
        return 1;
    }

	// Allocate the GPU Arrays
    ERR_CHECK(cl_create_buffer(&ofc->m_frame[0], ofc->m_clContext, CL_MEM_READ_ONLY, 3 * dimY * dimX, "frame[0]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_frame[1], ofc->m_clContext, CL_MEM_READ_ONLY, 3 * dimY * dimX, "frame[1]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_frame[2], ofc->m_clContext, CL_MEM_READ_ONLY, 3 * dimY * dimX, "frame[2]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_blurredFrame[0], ofc->m_clContext, CL_MEM_READ_WRITE, dimY * dimX * sizeof(unsigned short), "blurredFrame[0]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_blurredFrame[1], ofc->m_clContext, CL_MEM_READ_WRITE, dimY * dimX * sizeof(unsigned short), "blurredFrame[1]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_blurredFrame[2], ofc->m_clContext, CL_MEM_READ_WRITE, dimY * dimX * sizeof(unsigned short), "blurredFrame[2]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_warpedFrame12, ofc->m_clContext, CL_MEM_READ_WRITE, 3 * dimY * dimX, "warpedFrame12"));
	ERR_CHECK(cl_create_buffer(&ofc->m_warpedFrame21, ofc->m_clContext, CL_MEM_READ_WRITE, 3 * dimY * dimX, "warpedFrame21"));
	ERR_CHECK(cl_create_buffer(&ofc->m_outputFrame, ofc->m_clContext, CL_MEM_WRITE_ONLY, 3 * dimY * dimX, "outputFrame"));
	ERR_CHECK(cl_create_buffer(&ofc->m_offsetArray12, ofc->m_clContext, CL_MEM_READ_WRITE, MAX_SEARCH_RADIUS * MAX_SEARCH_RADIUS * 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "offsetArray12"));
	ERR_CHECK(cl_create_buffer(&ofc->m_offsetArray21, ofc->m_clContext, CL_MEM_READ_WRITE, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "offsetArray21"));
	ERR_CHECK(cl_create_buffer(&ofc->m_blurredOffsetArray12[0], ofc->m_clContext, CL_MEM_READ_WRITE, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "blurredOffsetArray12[0]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_blurredOffsetArray21[0], ofc->m_clContext, CL_MEM_READ_WRITE, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "blurredOffsetArray21[0]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_blurredOffsetArray12[1], ofc->m_clContext, CL_MEM_READ_WRITE, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "blurredOffsetArray12[1]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_blurredOffsetArray21[1], ofc->m_clContext, CL_MEM_READ_WRITE, 2 * ofc->m_iLowDimY * ofc->m_iLowDimX, "blurredOffsetArray21[1]"));
	ERR_CHECK(cl_create_buffer(&ofc->m_summedUpDeltaArray, ofc->m_clContext, CL_MEM_READ_WRITE, MAX_SEARCH_RADIUS * MAX_SEARCH_RADIUS * ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(unsigned int), "summedUpDeltaArray"));
	ERR_CHECK(cl_create_buffer(&ofc->m_lowestLayerArray, ofc->m_clContext, CL_MEM_READ_WRITE, ofc->m_iLowDimY * ofc->m_iLowDimX * sizeof(unsigned short), "lowestLayerArray"));
	ERR_CHECK(cl_create_buffer(&ofc->m_hitCount12, ofc->m_clContext, CL_MEM_READ_WRITE, dimY * dimX * sizeof(int), "hitCount12"));
	ERR_CHECK(cl_create_buffer(&ofc->m_hitCount21, ofc->m_clContext, CL_MEM_READ_WRITE, dimY * dimX * sizeof(int), "hitCount21"));
    ofc->m_imageArrayCPU = (unsigned short*)malloc(3 * dimY * dimX);
    if (!ofc->m_imageArrayCPU) {
        fprintf(stderr, "Error allocating CPU memory for m_imageArrayCPU\n");
        return 1;
    }

    // Compile the kernels
    ERR_CHECK(cl_create_kernel(&ofc->m_processFrameKernel, ofc->m_clContext, ofc->m_clDevice_id, "processFrameKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_blurFrameKernel, ofc->m_clContext, ofc->m_clDevice_id, "blurFrameKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_setInitialOffsetKernel, ofc->m_clContext, ofc->m_clDevice_id, "setInitialOffsetKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_calcDeltaSumsKernel, ofc->m_clContext, ofc->m_clDevice_id, "calcDeltaSumsKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_determineLowestLayerKernel, ofc->m_clContext, ofc->m_clDevice_id, "determineLowestLayerKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_adjustOffsetArrayKernel, ofc->m_clContext, ofc->m_clDevice_id, "adjustOffsetArrayKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_flipFlowKernel, ofc->m_clContext, ofc->m_clDevice_id, "flipFlowKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_blurFlowKernel, ofc->m_clContext, ofc->m_clDevice_id, "blurFlowKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_cleanFlowKernel, ofc->m_clContext, ofc->m_clDevice_id, "cleanFlowKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_warpFrameKernel, ofc->m_clContext, ofc->m_clDevice_id, "warpFrameKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_artifactRemovalKernel, ofc->m_clContext, ofc->m_clDevice_id, "artifactRemovalKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_blendFrameKernel, ofc->m_clContext, ofc->m_clDevice_id, "blendFrameKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_insertFrameKernel, ofc->m_clContext, ofc->m_clDevice_id, "insertFrameKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_sideBySideFrameKernel, ofc->m_clContext, ofc->m_clDevice_id, "sideBySideFrameKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_convertFlowToHSVKernel, ofc->m_clContext, ofc->m_clDevice_id, "convertFlowToHSVKernel"));
	ERR_CHECK(cl_create_kernel(&ofc->m_convertFlowToGrayscaleKernel, ofc->m_clContext, ofc->m_clDevice_id, "convertFlowToGrayscaleKernel"));
    ERR_CHECK(cl_create_kernel(&ofc->m_tearingTestKernel, ofc->m_clContext, ofc->m_clDevice_id, "tearingTestKernel"));

    // Set kernel arguments
    setKernelParameters(ofc);

    return 0;
}