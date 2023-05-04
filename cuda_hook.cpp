#include <iostream>
#include <cstring>
#include <pthread.h>

#include <cuda.h>

#include "cuda_hook.hpp"
#include "hook.hpp"
#include "logging.hpp"

using namespace std::string_literals;
using std::string;

static struct cudaHookInfo cuda_hook_info;
static pthread_once_t cuda_hook_init_done = PTHREAD_ONCE_INIT;

/* prehook, proxy, posthook functions start */
CUresult cuGetProcAddress_prehook(
    const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
{
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_proxy(
    const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
{
    typedef decltype(&cuGetProcAddress) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_GET_PROC_ADDRESS])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuGetProcAddress));
        cuda_hook_info.func_actual[CU_GET_PROC_ADDRESS] = actual_func;
    }
    return ((func_type)actual_func)(symbol, pfn, cudaVersion, flags);
}

/* cuGetProcAddress() is the entry of cuda api functions for cuda version >= 11.3 */
CUresult cuGetProcAddress_posthook(
    const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
{
    hook_log.debug("cuGetProcAddress: symbol "s + string(symbol) + ", cudaVersion "s + std::to_string(cudaVersion));

    /* Hook functions for cuda version >= 11.3 */
    if(strcmp(symbol, "cuGetProcAddress") == 0) {
        cuda_hook_info.func_actual[CU_GET_PROC_ADDRESS] = *pfn;
        *pfn = reinterpret_cast<void *>(cuGetProcAddress);
    }
    else if(strcmp(symbol, "cuMemAlloc") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemAlloc);
    }
    else if(strcmp(symbol, "cuMemAllocManaged") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_MANAGED] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemAllocManaged);
    }
    else if(strcmp(symbol, "cuMemAllocPitch") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_PITCH] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemAllocPitch);
    }
    else if(strcmp(symbol, "cuMemFree") == 0) {
        cuda_hook_info.func_actual[CU_MEM_FREE] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemFree);
    }
    else if(strcmp(symbol, "cuMemcpy") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpy);
    }
    else if(strcmp(symbol, "cuMemcpyAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyAsync);
    }
    else if(strcmp(symbol, "cuMemcpyDtoD") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_D] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyDtoD);
    }
    else if(strcmp(symbol, "cuMemcpyDtoDAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_D_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyDtoDAsync);
    }
    else if(strcmp(symbol, "cuMemcpyDtoH") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_H] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyDtoH);
    }
    else if(strcmp(symbol, "cuMemcpyDtoHAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_H_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyDtoHAsync);
    }
    else if(strcmp(symbol, "cuMemcpyHtoD") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_D] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyHtoD);
    }
    else if(strcmp(symbol, "cuMemcpyHtoDAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_D_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyHtoDAsync);
    }
    else if(strcmp(symbol, "cuMemcpyPeer") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_PEER] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyPeer);
    }
    else if(strcmp(symbol, "cuMemcpyPeerAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_PEER_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemcpyPeerAsync);
    }
    else if(strcmp(symbol, "cuMemsetD16") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D16] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD16);
    }
    else if(strcmp(symbol, "cuMemsetD16Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D16_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD16Async);
    }
    else if(strcmp(symbol, "cuMemsetD2D16") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D16] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD2D16);
    }
    else if(strcmp(symbol, "cuMemsetD2D16Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D16_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD2D16Async);
    }
    else if(strcmp(symbol, "cuMemsetD2D32") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D32] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD2D32);
    }
    else if(strcmp(symbol, "cuMemsetD2D32Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D32_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD2D32Async);
    }
    else if(strcmp(symbol, "cuMemsetD2D8") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D8] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD2D8);
    }
    else if(strcmp(symbol, "cuMemsetD2D8Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D8_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD2D8Async);
    }
    else if(strcmp(symbol, "cuMemsetD32") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D32] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD32);
    }
    else if(strcmp(symbol, "cuMemsetD32Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D32_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD32Async);
    }
    else if(strcmp(symbol, "cuMemsetD8") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D8] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD8);
    }
    else if(strcmp(symbol, "cuMemsetD8Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D8_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemsetD8Async);
    }
    else if(strcmp(symbol, "cuMemAllocAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemAllocAsync);
    }
    else if(strcmp(symbol, "cuMemFreeAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEM_FREE_ASYNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuMemFreeAsync);
    }
    else if(strcmp(symbol, "cuLaunchCooperativeKernel") == 0) {
        cuda_hook_info.func_actual[CU_LAUNCH_COOPERATIVE_KERNEL] = *pfn;
        *pfn = reinterpret_cast<void *>(cuLaunchCooperativeKernel);
    }
    else if(strcmp(symbol, "cuLaunchHostFunc") == 0) {
        cuda_hook_info.func_actual[CU_LAUNCH_HOST_FUNC] = *pfn;
        *pfn = reinterpret_cast<void *>(cuLaunchHostFunc);
    }
    else if(strcmp(symbol, "cuLaunchKernel") == 0) {
        cuda_hook_info.func_actual[CU_LAUNCH_KERNEL] = *pfn;
        *pfn = reinterpret_cast<void *>(cuLaunchKernel);
    }
    trace_dump.dump("cuGetProcAddress");
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_prehook(
    CUdeviceptr *dptr, size_t bytesize)
{
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_proxy(
    CUdeviceptr *dptr, size_t bytesize)
{
    typedef decltype(&cuMemAlloc) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEM_ALLOC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemAlloc));
        cuda_hook_info.func_actual[CU_MEM_ALLOC] = actual_func;
    }
    return ((func_type)actual_func)(dptr, bytesize);
}

CUresult cuMemAlloc_posthook(
    CUdeviceptr *dptr, size_t bytesize)
{
    trace_dump.dump("cuMemAlloc");
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged_prehook(
    CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged_proxy(
    CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    typedef decltype(&cuMemAllocManaged) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEM_ALLOC_MANAGED])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemAllocManaged));
        cuda_hook_info.func_actual[CU_MEM_ALLOC_MANAGED] = actual_func;
    }
    return ((func_type)actual_func)(dptr, bytesize, flags);
}

CUresult cuMemAllocManaged_posthook(
    CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    trace_dump.dump("cuMemAllocManaged");
    return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch_prehook(
    CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height,
    unsigned int ElementSizeBytes)
{
    return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch_proxy(
    CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height,
    unsigned int ElementSizeBytes)
{
    typedef decltype(&cuMemAllocPitch) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEM_ALLOC_PITCH])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemAllocPitch));
        cuda_hook_info.func_actual[CU_MEM_ALLOC_PITCH] = actual_func;
    }
    return ((func_type)actual_func)(dptr, pPitch, WidthInBytes, Height,
        ElementSizeBytes);
}

CUresult cuMemAllocPitch_posthook(
    CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height,
    unsigned int ElementSizeBytes)
{
    trace_dump.dump("cuMemAllocPitch");
    return CUDA_SUCCESS;
}

CUresult cuMemFree_prehook(
    CUdeviceptr dptr)
{
    return CUDA_SUCCESS;
}

CUresult cuMemFree_proxy(
    CUdeviceptr dptr)
{
    typedef decltype(&cuMemFree) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEM_FREE])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemFree));
        cuda_hook_info.func_actual[CU_MEM_FREE] = actual_func;
    }
    return ((func_type)actual_func)(dptr);
}

CUresult cuMemFree_posthook(
    CUdeviceptr dptr)
{
    trace_dump.dump("cuMemFree");
    return CUDA_SUCCESS;
}

CUresult cuMemcpy_prehook(
    CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpy_proxy(
    CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    typedef decltype(&cuMemcpy) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpy));
        cuda_hook_info.func_actual[CU_MEMCPY] = actual_func;
    }
    return ((func_type)actual_func)(dst, src, ByteCount);
}

CUresult cuMemcpy_posthook(
    CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    trace_dump.dump("cuMemcpy");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync_prehook(
    CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync_proxy(
    CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    typedef decltype(&cuMemcpyAsync) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyAsync));
        cuda_hook_info.func_actual[CU_MEMCPY_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dst, src, ByteCount, hStream);
}

CUresult cuMemcpyAsync_posthook(
    CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    trace_dump.dump("cuMemcpyAsync");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_prehook(
    CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_proxy(
    CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    typedef decltype(&cuMemcpyDtoD) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_DTO_D])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyDtoD));
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_D] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpyDtoD_posthook(
    CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    trace_dump.dump("cuMemcpyDtoD");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoDAsync_prehook(
    CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoDAsync_proxy(
    CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    typedef decltype(&cuMemcpyDtoDAsync) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_DTO_D_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyDtoDAsync));
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_D_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoDAsync_posthook(
    CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    trace_dump.dump("cuMemcpyDtoDAsync");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_prehook(
    void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_proxy(
    void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    typedef decltype(&cuMemcpyDtoH) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_DTO_H])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyDtoH));
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_H] = actual_func;
    }
    return ((func_type)actual_func)(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyDtoH_posthook(
    void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    trace_dump.dump("cuMemcpyDtoH");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync_prehook(
    void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync_proxy(
    void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    typedef decltype(&cuMemcpyDtoHAsync) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_DTO_H_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyDtoHAsync));
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_H_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstHost, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoHAsync_posthook(
    void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    trace_dump.dump("cuMemcpyDtoHAsync");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_prehook(
    CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_proxy(
    CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    typedef decltype(&cuMemcpyHtoD) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_HTO_D])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyHtoD));
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_D] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyHtoD_posthook(
    CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    trace_dump.dump("cuMemcpyHtoD");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoDAsync_prehook(
    CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoDAsync_proxy(
    CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    typedef decltype(&cuMemcpyHtoDAsync) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_HTO_D_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyHtoDAsync));
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_D_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, srcHost, ByteCount, hStream);
}

CUresult cuMemcpyHtoDAsync_posthook(
    CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    trace_dump.dump("cuMemcpyHtoDAsync");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeer_prehook(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeer_proxy(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount)
{
    typedef decltype(&cuMemcpyPeer) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_PEER])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyPeer));
        cuda_hook_info.func_actual[CU_MEMCPY_PEER] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstContext, srcDevice, srcContext,
        ByteCount);
}

CUresult cuMemcpyPeer_posthook(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount)
{
    trace_dump.dump("cuMemcpyPeer");
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeerAsync_prehook(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeerAsync_proxy(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount, CUstream hStream)
{
    typedef decltype(&cuMemcpyPeerAsync) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMCPY_PEER_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemcpyPeerAsync));
        cuda_hook_info.func_actual[CU_MEMCPY_PEER_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstContext, srcDevice, srcContext,
        ByteCount, hStream);
}

CUresult cuMemcpyPeerAsync_posthook(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount, CUstream hStream)
{
    trace_dump.dump("cuMemcpyPeerAsync");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16_prehook(
    CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16_proxy(
    CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    typedef decltype(&cuMemsetD16) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D16])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD16));
        cuda_hook_info.func_actual[CU_MEMSET_D16] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, us, N);
}

CUresult cuMemsetD16_posthook(
    CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    trace_dump.dump("cuMemsetD16");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16Async_prehook(
    CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16Async_proxy(
    CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    typedef decltype(&cuMemsetD16Async) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D16_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD16Async));
        cuda_hook_info.func_actual[CU_MEMSET_D16_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, us, N, hStream);
}

CUresult cuMemsetD16Async_posthook(
    CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    trace_dump.dump("cuMemsetD16Async");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16_prehook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
    size_t Height)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16_proxy(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
    size_t Height)
{
    typedef decltype(&cuMemsetD2D16) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D2D16])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD2D16));
        cuda_hook_info.func_actual[CU_MEMSET_D2D16] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstPitch, us, Width,
        Height);
}

CUresult cuMemsetD2D16_posthook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
    size_t Height)
{
    trace_dump.dump("cuMemsetD2D16");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16Async_prehook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
    size_t Height, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16Async_proxy(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
    size_t Height, CUstream hStream)
{
    typedef decltype(&cuMemsetD2D16Async) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D2D16_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD2D16Async));
        cuda_hook_info.func_actual[CU_MEMSET_D2D16_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstPitch, us, Width,
        Height, hStream);
}

CUresult cuMemsetD2D16Async_posthook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
    size_t Height, CUstream hStream)
{
    trace_dump.dump("cuMemsetD2D16Async");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32_prehook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width,
    size_t Height)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32_proxy(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width,
    size_t Height)
{
    typedef decltype(&cuMemsetD2D32) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D2D32])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD2D32));
        cuda_hook_info.func_actual[CU_MEMSET_D2D32] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstPitch, ui, Width,
        Height);
}

CUresult cuMemsetD2D32_posthook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width,
    size_t Height)
{
    trace_dump.dump("cuMemsetD2D32");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32Async_prehook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width,
    size_t Height, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32Async_proxy(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width,
    size_t Height, CUstream hStream)
{
    typedef decltype(&cuMemsetD2D32Async) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D2D32_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD2D32Async));
        cuda_hook_info.func_actual[CU_MEMSET_D2D32_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstPitch, ui, Width,
        Height, hStream);
}

CUresult cuMemsetD2D32Async_posthook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width,
    size_t Height, CUstream hStream)
{
    trace_dump.dump("cuMemsetD2D32Async");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8_prehook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width,
    size_t Height)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8_proxy(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width,
    size_t Height)
{
    typedef decltype(&cuMemsetD2D8) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D2D8])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD2D8));
        cuda_hook_info.func_actual[CU_MEMSET_D2D8] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstPitch, uc, Width,
        Height);
}

CUresult cuMemsetD2D8_posthook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width,
    size_t Height)
{
    trace_dump.dump("cuMemsetD2D8");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8Async_prehook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width,
    size_t Height, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8Async_proxy(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width,
    size_t Height, CUstream hStream)
{
    typedef decltype(&cuMemsetD2D8Async) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D2D8_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD2D8Async));
        cuda_hook_info.func_actual[CU_MEMSET_D2D8_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, dstPitch, uc, Width,
        Height, hStream);
}

CUresult cuMemsetD2D8Async_posthook(
    CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width,
    size_t Height, CUstream hStream)
{
    trace_dump.dump("cuMemsetD2D8Async");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32_prehook(
    CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32_proxy(
    CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    typedef decltype(&cuMemsetD32) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D32])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD32));
        cuda_hook_info.func_actual[CU_MEMSET_D32] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, ui, N);
}

CUresult cuMemsetD32_posthook(
    CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    trace_dump.dump("cuMemsetD32");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32Async_prehook(
    CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32Async_proxy(
    CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    typedef decltype(&cuMemsetD32Async) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D32_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD32Async));
        cuda_hook_info.func_actual[CU_MEMSET_D32_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, ui, N, hStream);
}

CUresult cuMemsetD32Async_posthook(
    CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    trace_dump.dump("cuMemsetD32Async");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8_prehook(
    CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8_proxy(
    CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    typedef decltype(&cuMemsetD8) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D8])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD8));
        cuda_hook_info.func_actual[CU_MEMSET_D8] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, uc, N);
}

CUresult cuMemsetD8_posthook(
    CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    trace_dump.dump("cuMemsetD8");
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async_prehook(
    CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async_proxy(
    CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    typedef decltype(&cuMemsetD8Async) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEMSET_D8_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemsetD8Async));
        cuda_hook_info.func_actual[CU_MEMSET_D8_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dstDevice, uc, N, hStream);
}

CUresult cuMemsetD8Async_posthook(
    CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    trace_dump.dump("cuMemsetD8Async");
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync_prehook(
    CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync_proxy(
    CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
{
    typedef decltype(&cuMemAllocAsync) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEM_ALLOC_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemAllocAsync));
        cuda_hook_info.func_actual[CU_MEM_ALLOC_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dptr, bytesize, hStream);
}

CUresult cuMemAllocAsync_posthook(
    CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
{
    trace_dump.dump("cuMemAllocAsync");
    return CUDA_SUCCESS;
}

CUresult cuMemFreeAsync_prehook(
    CUdeviceptr dptr, CUstream hStream)
{
    return CUDA_SUCCESS;
}

CUresult cuMemFreeAsync_proxy(
    CUdeviceptr dptr, CUstream hStream)
{
    typedef decltype(&cuMemFreeAsync) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_MEM_FREE_ASYNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuMemFreeAsync));
        cuda_hook_info.func_actual[CU_MEM_FREE_ASYNC] = actual_func;
    }
    return ((func_type)actual_func)(dptr, hStream);
}

CUresult cuMemFreeAsync_posthook(
    CUdeviceptr dptr, CUstream hStream)
{
    trace_dump.dump("cuMemFreeAsync");
    return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel_prehook(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams)
{
    return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel_proxy(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams)
{
    typedef decltype(&cuLaunchCooperativeKernel) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_LAUNCH_COOPERATIVE_KERNEL])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuLaunchCooperativeKernel));
        cuda_hook_info.func_actual[CU_LAUNCH_COOPERATIVE_KERNEL] = actual_func;
    }
    return ((func_type)actual_func)(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes,
        hStream, kernelParams);
}

CUresult cuLaunchCooperativeKernel_posthook(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams)
{
    trace_dump.dump("cuLaunchCooperativeKernel");
    return CUDA_SUCCESS;
}

CUresult cuLaunchHostFunc_prehook(
    CUstream hStream, CUhostFn fn, void *userData)
{
    return CUDA_SUCCESS;
}

CUresult cuLaunchHostFunc_proxy(
    CUstream hStream, CUhostFn fn, void *userData)
{
    typedef decltype(&cuLaunchHostFunc) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_LAUNCH_HOST_FUNC])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuLaunchHostFunc));
        cuda_hook_info.func_actual[CU_LAUNCH_HOST_FUNC] = actual_func;
    }
    return ((func_type)actual_func)(hStream, fn, userData);
}

CUresult cuLaunchHostFunc_posthook(
    CUstream hStream, CUhostFn fn, void *userData)
{
    trace_dump.dump("cuLaunchHostFunc");
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel_prehook(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams, void **extra)
{
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel_proxy(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams, void **extra)
{
    typedef decltype(&cuLaunchKernel) func_type;
    void *actual_func;
    if(!(actual_func = cuda_hook_info.func_actual[CU_LAUNCH_KERNEL])) {
        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(cuLaunchKernel));
        cuda_hook_info.func_actual[CU_LAUNCH_KERNEL] = actual_func;
    }
    return ((func_type)actual_func)(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes,
        hStream, kernelParams, extra);
}

CUresult cuLaunchKernel_posthook(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams, void **extra)
{
    trace_dump.dump("cuLaunchKernel");
    return CUDA_SUCCESS;
}
/* prehook, proxy, posthook functions end */

static void cuda_hook_init()
{
    cuda_hook_info.func_prehook[CU_GET_PROC_ADDRESS] =
        reinterpret_cast<void *>(cuGetProcAddress_prehook);
    cuda_hook_info.func_proxy[CU_GET_PROC_ADDRESS] =
        reinterpret_cast<void *>(cuGetProcAddress_proxy);
    cuda_hook_info.func_posthook[CU_GET_PROC_ADDRESS] =
        reinterpret_cast<void *>(cuGetProcAddress_posthook);
    cuda_hook_info.func_prehook[CU_MEM_ALLOC] =
        reinterpret_cast<void *>(cuMemAlloc_prehook);
    cuda_hook_info.func_proxy[CU_MEM_ALLOC] =
        reinterpret_cast<void *>(cuMemAlloc_proxy);
    cuda_hook_info.func_posthook[CU_MEM_ALLOC] =
        reinterpret_cast<void *>(cuMemAlloc_posthook);
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_MANAGED] =
        reinterpret_cast<void *>(cuMemAllocManaged_prehook);
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_MANAGED] =
        reinterpret_cast<void *>(cuMemAllocManaged_proxy);
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_MANAGED] =
        reinterpret_cast<void *>(cuMemAllocManaged_posthook);
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_PITCH] =
        reinterpret_cast<void *>(cuMemAllocPitch_prehook);
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_PITCH] =
        reinterpret_cast<void *>(cuMemAllocPitch_proxy);
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_PITCH] =
        reinterpret_cast<void *>(cuMemAllocPitch_posthook);
    cuda_hook_info.func_prehook[CU_MEM_FREE] =
        reinterpret_cast<void *>(cuMemFree_prehook);
    cuda_hook_info.func_proxy[CU_MEM_FREE] =
        reinterpret_cast<void *>(cuMemFree_proxy);
    cuda_hook_info.func_posthook[CU_MEM_FREE] =
        reinterpret_cast<void *>(cuMemFree_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY] =
        reinterpret_cast<void *>(cuMemcpy_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY] =
        reinterpret_cast<void *>(cuMemcpy_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY] =
        reinterpret_cast<void *>(cuMemcpy_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyAsync_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyAsync_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyAsync_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_D] =
        reinterpret_cast<void *>(cuMemcpyDtoD_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_D] =
        reinterpret_cast<void *>(cuMemcpyDtoD_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_D] =
        reinterpret_cast<void *>(cuMemcpyDtoD_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_D_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyDtoDAsync_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_D_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyDtoDAsync_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_D_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyDtoDAsync_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_H] =
        reinterpret_cast<void *>(cuMemcpyDtoH_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_H] =
        reinterpret_cast<void *>(cuMemcpyDtoH_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_H] =
        reinterpret_cast<void *>(cuMemcpyDtoH_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_H_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyDtoHAsync_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_H_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyDtoHAsync_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_H_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyDtoHAsync_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_HTO_D] =
        reinterpret_cast<void *>(cuMemcpyHtoD_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_HTO_D] =
        reinterpret_cast<void *>(cuMemcpyHtoD_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_HTO_D] =
        reinterpret_cast<void *>(cuMemcpyHtoD_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_HTO_D_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyHtoDAsync_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_HTO_D_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyHtoDAsync_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_HTO_D_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyHtoDAsync_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_PEER] =
        reinterpret_cast<void *>(cuMemcpyPeer_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_PEER] =
        reinterpret_cast<void *>(cuMemcpyPeer_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_PEER] =
        reinterpret_cast<void *>(cuMemcpyPeer_posthook);
    cuda_hook_info.func_prehook[CU_MEMCPY_PEER_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyPeerAsync_prehook);
    cuda_hook_info.func_proxy[CU_MEMCPY_PEER_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyPeerAsync_proxy);
    cuda_hook_info.func_posthook[CU_MEMCPY_PEER_ASYNC] =
        reinterpret_cast<void *>(cuMemcpyPeerAsync_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D16] =
        reinterpret_cast<void *>(cuMemsetD16_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D16] =
        reinterpret_cast<void *>(cuMemsetD16_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D16] =
        reinterpret_cast<void *>(cuMemsetD16_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D16_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD16Async_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D16_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD16Async_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D16_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD16Async_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D2D16] =
        reinterpret_cast<void *>(cuMemsetD2D16_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D2D16] =
        reinterpret_cast<void *>(cuMemsetD2D16_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D2D16] =
        reinterpret_cast<void *>(cuMemsetD2D16_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D2D16_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D16Async_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D2D16_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D16Async_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D2D16_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D16Async_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D2D32] =
        reinterpret_cast<void *>(cuMemsetD2D32_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D2D32] =
        reinterpret_cast<void *>(cuMemsetD2D32_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D2D32] =
        reinterpret_cast<void *>(cuMemsetD2D32_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D2D32_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D32Async_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D2D32_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D32Async_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D2D32_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D32Async_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D2D8] =
        reinterpret_cast<void *>(cuMemsetD2D8_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D2D8] =
        reinterpret_cast<void *>(cuMemsetD2D8_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D2D8] =
        reinterpret_cast<void *>(cuMemsetD2D8_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D2D8_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D8Async_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D2D8_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D8Async_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D2D8_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD2D8Async_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D32] =
        reinterpret_cast<void *>(cuMemsetD32_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D32] =
        reinterpret_cast<void *>(cuMemsetD32_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D32] =
        reinterpret_cast<void *>(cuMemsetD32_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D32_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD32Async_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D32_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD32Async_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D32_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD32Async_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D8] =
        reinterpret_cast<void *>(cuMemsetD8_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D8] =
        reinterpret_cast<void *>(cuMemsetD8_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D8] =
        reinterpret_cast<void *>(cuMemsetD8_posthook);
    cuda_hook_info.func_prehook[CU_MEMSET_D8_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD8Async_prehook);
    cuda_hook_info.func_proxy[CU_MEMSET_D8_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD8Async_proxy);
    cuda_hook_info.func_posthook[CU_MEMSET_D8_ASYNC] =
        reinterpret_cast<void *>(cuMemsetD8Async_posthook);
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_ASYNC] =
        reinterpret_cast<void *>(cuMemAllocAsync_prehook);
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_ASYNC] =
        reinterpret_cast<void *>(cuMemAllocAsync_proxy);
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_ASYNC] =
        reinterpret_cast<void *>(cuMemAllocAsync_posthook);
    cuda_hook_info.func_prehook[CU_MEM_FREE_ASYNC] =
        reinterpret_cast<void *>(cuMemFreeAsync_prehook);
    cuda_hook_info.func_proxy[CU_MEM_FREE_ASYNC] =
        reinterpret_cast<void *>(cuMemFreeAsync_proxy);
    cuda_hook_info.func_posthook[CU_MEM_FREE_ASYNC] =
        reinterpret_cast<void *>(cuMemFreeAsync_posthook);
    cuda_hook_info.func_prehook[CU_LAUNCH_COOPERATIVE_KERNEL] =
        reinterpret_cast<void *>(cuLaunchCooperativeKernel_prehook);
    cuda_hook_info.func_proxy[CU_LAUNCH_COOPERATIVE_KERNEL] =
        reinterpret_cast<void *>(cuLaunchCooperativeKernel_proxy);
    cuda_hook_info.func_posthook[CU_LAUNCH_COOPERATIVE_KERNEL] =
        reinterpret_cast<void *>(cuLaunchCooperativeKernel_posthook);
    cuda_hook_info.func_prehook[CU_LAUNCH_HOST_FUNC] =
        reinterpret_cast<void *>(cuLaunchHostFunc_prehook);
    cuda_hook_info.func_proxy[CU_LAUNCH_HOST_FUNC] =
        reinterpret_cast<void *>(cuLaunchHostFunc_proxy);
    cuda_hook_info.func_posthook[CU_LAUNCH_HOST_FUNC] =
        reinterpret_cast<void *>(cuLaunchHostFunc_posthook);
    cuda_hook_info.func_prehook[CU_LAUNCH_KERNEL] =
        reinterpret_cast<void *>(cuLaunchKernel_prehook);
    cuda_hook_info.func_proxy[CU_LAUNCH_KERNEL] =
        reinterpret_cast<void *>(cuLaunchKernel_proxy);
    cuda_hook_info.func_posthook[CU_LAUNCH_KERNEL] =
        reinterpret_cast<void *>(cuLaunchKernel_posthook);
}

/* hook function start */
CUDA_HOOK_GEN(
    CU_GET_PROC_ADDRESS,
    ,
    cuGetProcAddress,
    (const char *symbol, void **pfn,
    int cudaVersion, cuuint64_t flags),
    symbol, pfn, cudaVersion, flags)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC,
    ,
    cuMemAlloc,
    (CUdeviceptr *dptr, size_t bytesize),
    dptr, bytesize)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_MANAGED,
    ,
    cuMemAllocManaged,
    (CUdeviceptr *dptr, size_t bytesize,
    unsigned int flags),
    dptr, bytesize, flags)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_PITCH,
    ,
    cuMemAllocPitch,
    (CUdeviceptr *dptr, size_t *pPitch,
    size_t WidthInBytes, size_t Height,
    unsigned int ElementSizeBytes),
    dptr, pPitch, WidthInBytes, Height,
    ElementSizeBytes)

CUDA_HOOK_GEN(
    CU_MEM_FREE,
    ,
    cuMemFree,
    (CUdeviceptr dptr),
    dptr)

CUDA_HOOK_GEN(
    CU_MEMCPY,
    ,
    cuMemcpy,
    (CUdeviceptr dst, CUdeviceptr src,
    size_t ByteCount),
    dst, src, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_ASYNC,
    ,
    cuMemcpyAsync,
    (CUdeviceptr dst, CUdeviceptr src,
    size_t ByteCount, CUstream hStream),
    dst, src, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_D,
    ,
    cuMemcpyDtoD,
    (CUdeviceptr dstDevice, CUdeviceptr srcDevice,
    size_t ByteCount),
    dstDevice, srcDevice, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_D_ASYNC,
    ,
    cuMemcpyDtoDAsync,
    (CUdeviceptr dstDevice, CUdeviceptr srcDevice,
    size_t ByteCount, CUstream hStream),
    dstDevice, srcDevice, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_H,
    ,
    cuMemcpyDtoH,
    (void *dstHost, CUdeviceptr srcDevice,
    size_t ByteCount),
    dstHost, srcDevice, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_H_ASYNC,
    ,
    cuMemcpyDtoHAsync,
    (void *dstHost, CUdeviceptr srcDevice,
    size_t ByteCount, CUstream hStream),
    dstHost, srcDevice, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_HTO_D,
    ,
    cuMemcpyHtoD,
    (CUdeviceptr dstDevice, const void *srcHost,
    size_t ByteCount),
    dstDevice, srcHost, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_HTO_D_ASYNC,
    ,
    cuMemcpyHtoDAsync,
    (CUdeviceptr dstDevice, const void *srcHost,
    size_t ByteCount, CUstream hStream),
    dstDevice, srcHost, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_PEER,
    ,
    cuMemcpyPeer,
    (CUdeviceptr dstDevice, CUcontext dstContext,
    CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount),
    dstDevice, dstContext, srcDevice, srcContext,
    ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_PEER_ASYNC,
    ,
    cuMemcpyPeerAsync,
    (CUdeviceptr dstDevice, CUcontext dstContext,
    CUdeviceptr srcDevice, CUcontext srcContext,
    size_t ByteCount, CUstream hStream),
    dstDevice, dstContext, srcDevice, srcContext,
    ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D16,
    ,
    cuMemsetD16,
    (CUdeviceptr dstDevice, unsigned short us,
    size_t N),
    dstDevice, us, N)

CUDA_HOOK_GEN(
    CU_MEMSET_D16_ASYNC,
    ,
    cuMemsetD16Async,
    (CUdeviceptr dstDevice, unsigned short us,
    size_t N, CUstream hStream),
    dstDevice, us, N, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D16,
    ,
    cuMemsetD2D16,
    (CUdeviceptr dstDevice, size_t dstPitch,
    unsigned short us, size_t Width,
    size_t Height),
    dstDevice, dstPitch, us, Width,
    Height)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D16_ASYNC,
    ,
    cuMemsetD2D16Async,
    (CUdeviceptr dstDevice, size_t dstPitch,
    unsigned short us, size_t Width,
    size_t Height, CUstream hStream),
    dstDevice, dstPitch, us, Width,
    Height, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D32,
    ,
    cuMemsetD2D32,
    (CUdeviceptr dstDevice, size_t dstPitch,
    unsigned int ui, size_t Width,
    size_t Height),
    dstDevice, dstPitch, ui, Width,
    Height)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D32_ASYNC,
    ,
    cuMemsetD2D32Async,
    (CUdeviceptr dstDevice, size_t dstPitch,
    unsigned int ui, size_t Width,
    size_t Height, CUstream hStream),
    dstDevice, dstPitch, ui, Width,
    Height, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D8,
    ,
    cuMemsetD2D8,
    (CUdeviceptr dstDevice, size_t dstPitch,
    unsigned char uc, size_t Width,
    size_t Height),
    dstDevice, dstPitch, uc, Width,
    Height)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D8_ASYNC,
    ,
    cuMemsetD2D8Async,
    (CUdeviceptr dstDevice, size_t dstPitch,
    unsigned char uc, size_t Width,
    size_t Height, CUstream hStream),
    dstDevice, dstPitch, uc, Width,
    Height, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D32,
    ,
    cuMemsetD32,
    (CUdeviceptr dstDevice, unsigned int ui,
    size_t N),
    dstDevice, ui, N)

CUDA_HOOK_GEN(
    CU_MEMSET_D32_ASYNC,
    ,
    cuMemsetD32Async,
    (CUdeviceptr dstDevice, unsigned int ui,
    size_t N, CUstream hStream),
    dstDevice, ui, N, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D8,
    ,
    cuMemsetD8,
    (CUdeviceptr dstDevice, unsigned char uc,
    size_t N),
    dstDevice, uc, N)

CUDA_HOOK_GEN(
    CU_MEMSET_D8_ASYNC,
    ,
    cuMemsetD8Async,
    (CUdeviceptr dstDevice, unsigned char uc,
    size_t N, CUstream hStream),
    dstDevice, uc, N, hStream)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_ASYNC,
    ,
    cuMemAllocAsync,
    (CUdeviceptr *dptr, size_t bytesize,
    CUstream hStream),
    dptr, bytesize, hStream)

CUDA_HOOK_GEN(
    CU_MEM_FREE_ASYNC,
    ,
    cuMemFreeAsync,
    (CUdeviceptr dptr, CUstream hStream),
    dptr, hStream)

CUDA_HOOK_GEN(
    CU_LAUNCH_COOPERATIVE_KERNEL,
    ,
    cuLaunchCooperativeKernel,
    (CUfunction f, unsigned int gridDimX,
    unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams),
    f, gridDimX, gridDimY, gridDimZ,
    blockDimX, blockDimY, blockDimZ, sharedMemBytes,
    hStream, kernelParams)

CUDA_HOOK_GEN(
    CU_LAUNCH_HOST_FUNC,
    ,
    cuLaunchHostFunc,
    (CUstream hStream, CUhostFn fn,
    void *userData),
    hStream, fn, userData)

CUDA_HOOK_GEN(
    CU_LAUNCH_KERNEL,
    ,
    cuLaunchKernel,
    (CUfunction f, unsigned int gridDimX,
    unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams,
    void **extra),
    f, gridDimX, gridDimY, gridDimZ,
    blockDimX, blockDimY, blockDimZ, sharedMemBytes,
    hStream, kernelParams, extra)
/* hook function end */
