#ifndef _CUDA_HOOK_FUNC_H_
#define _CUDA_HOOK_FUNC_H_
#include <cuda.h>

/**
 * This file is only used for input of parser and codegen.
 *
 */

#define DRIVER_ENTRY_POINT_ACCESS
#define MEMORY_MANAGEMENT
#define STREAM_ORDERED_MEMORY_ALLOCATOR
#define EXECUTION_CONTROL

#ifdef DRIVER_ENTRY_POINT_ACCESS

/* Special function */
CUresult CUDAAPI
cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);

#endif /* DRIVER_ENTRY_POINT_ACCESS */

#ifdef MEMORY_MANAGEMENT

CUresult CUDAAPI
cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);

CUresult CUDAAPI
cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);

CUresult CUDAAPI
cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
    size_t Height, unsigned int ElementSizeBytes);

CUresult CUDAAPI
cuMemFree(CUdeviceptr dptr);

CUresult CUDAAPI
cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);

CUresult CUDAAPI
cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);

CUresult CUDAAPI
cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);

CUresult CUDAAPI
cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

CUresult CUDAAPI
cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);

CUresult CUDAAPI
cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

CUresult CUDAAPI
cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);

CUresult CUDAAPI
cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);

CUresult CUDAAPI
cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
    CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);

CUresult CUDAAPI
cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
    CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);

CUresult CUDAAPI
cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);

CUresult CUDAAPI
cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);

CUresult CUDAAPI
cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
    unsigned short us, size_t Width, size_t Height);

CUresult CUDAAPI
cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
    unsigned short us, size_t Width, size_t Height, CUstream hStream);

CUresult CUDAAPI
cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
    unsigned int ui, size_t Width, size_t Height);

CUresult CUDAAPI
cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
    unsigned int ui, size_t Width, size_t Height, CUstream hStream);

CUresult CUDAAPI
cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
    unsigned char uc, size_t Width, size_t Height);

CUresult CUDAAPI
cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
    unsigned char uc, size_t Width, size_t Height, CUstream hStream);

CUresult CUDAAPI
cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);

CUresult CUDAAPI
cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);

CUresult CUDAAPI
cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);

CUresult CUDAAPI
cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);

#endif /* MEMORY_MANAGEMENT */

#ifdef STREAM_ORDERED_MEMORY_ALLOCATOR

CUresult CUDAAPI
cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);

CUresult CUDAAPI
cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);

#endif /* STREAM_ORDERED_MEMORY_ALLOCATOR */

#ifdef EXECUTION_CONTROL

CUresult CUDAAPI
cuLaunchCooperativeKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);

/*
__CUDA_DEPRECATED CUresult CUDAAPI
cuLaunchCooperativeKernelMultiDevice(
    CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags);
*/

CUresult CUDAAPI
cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData);

CUresult CUDAAPI
cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

#endif /* EXECUTION_CONTROL */

#endif /* _CUDA_HOOK_FUNC_H_ */
