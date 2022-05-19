#define __USE_GNU
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#include <cuda.h>
#include <cudnn.h>

#include "hook.h"
#include "debug.h"

extern "C" {
void *__libc_dlopen_mode(const char *__name, int __mode);
void *__libc_dlsym(void *__map, const char *__name);
int __libc_dlclose(void *__map);
}

void *libdlHandle;
void *libcudaHandle;
void *libcudnnHandle;

FILE *fp_trace;

__attribute__((constructor))
void hookInit(void)
{
    DEBUG("[%s] Enter func\n", __func__);

    libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);
    libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
    libcudnnHandle = __libc_dlopen_mode("libcudnn.so", RTLD_LAZY);

#ifdef _TRACE_DUMP_ENABLE
    fp_trace = fopen("trace.txt", "w");
#endif /* _TRACE_DUMP_ENABLE */

    DEBUG("[%s] Leave func\n", __func__);
}

__attribute__((destructor))
void hookFini(void)
{
    DEBUG("[%s] Enter func\n", __func__);

    __libc_dlclose(libdlHandle);
    __libc_dlclose(libcudaHandle);
    __libc_dlclose(libcudnnHandle);

#ifdef _TRACE_DUMP_ENABLE
    fclose(fp_trace);
#endif /* _TRACE_DUMP_ENABLE */

    DEBUG("[%s] Leave func\n", __func__);
}

void *actualDlsym(void *handle, const char *symbol)
{
    DEBUG("[%s] Load %s\n", __func__, symbol);

    typedef decltype(&dlsym) funcType;
    void *dlsymFunc = __libc_dlsym(libdlHandle, "dlsym");

    void *actualFunc = ((funcType)dlsymFunc)(handle, symbol);
    if(!actualFunc)
        ERROR("[%s] Cannot load %s\n", __func__, symbol);
    return actualFunc;
}

void *dlsym(void *handle, const char *symbol)
{
    DEBUG("[      %s] Hook %s\n", __func__, symbol);

    if(strncmp(symbol, "cu", 2) != 0)
        return actualDlsym(handle, symbol);

#ifdef _CUDA_HOOK_ENABLE
    /* Hook functions for cuda version < 11.3 */
    if(strcmp(symbol, SYMBOL_STRING(cuGetProcAddress)) == 0) {
        return (void *)cuGetProcAddress;
    } else if(strcmp(symbol, SYMBOL_STRING(cuArray3DCreate)) == 0) {
        return (void *)cuArray3DCreate;
    } else if(strcmp(symbol, SYMBOL_STRING(cuArrayCreate)) == 0) {
        return (void *)cuArrayCreate;
    } else if(strcmp(symbol, SYMBOL_STRING(cuArrayDestroy)) == 0) {
        return (void *)cuArrayDestroy;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAlloc)) == 0) {
        return (void *)cuMemAlloc;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocHost)) == 0) {
        return (void *)cuMemAllocHost;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocManaged)) == 0) {
        return (void *)cuMemAllocManaged;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocPitch)) == 0) {
        return (void *)cuMemAllocPitch;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemFree)) == 0) {
        return (void *)cuMemFree;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemFreeHost)) == 0) {
        return (void *)cuMemFreeHost;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemHostAlloc)) == 0) {
        return (void *)cuMemHostAlloc;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy)) == 0) {
        return (void *)cuMemcpy;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy2D)) == 0) {
        return (void *)cuMemcpy2D;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy2DAsync)) == 0) {
        return (void *)cuMemcpy2DAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy2DUnaligned)) == 0) {
        return (void *)cuMemcpy2DUnaligned;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy3D)) == 0) {
        return (void *)cuMemcpy3D;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy3DAsync)) == 0) {
        return (void *)cuMemcpy3DAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy3DPeer)) == 0) {
        return (void *)cuMemcpy3DPeer;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy3DPeerAsync)) == 0) {
        return (void *)cuMemcpy3DPeerAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyAsync)) == 0) {
        return (void *)cuMemcpyAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyAtoA)) == 0) {
        return (void *)cuMemcpyAtoA;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyAtoD)) == 0) {
        return (void *)cuMemcpyAtoD;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyAtoH)) == 0) {
        return (void *)cuMemcpyAtoH;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyAtoHAsync)) == 0) {
        return (void *)cuMemcpyAtoHAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoA)) == 0) {
        return (void *)cuMemcpyDtoA;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoD)) == 0) {
        return (void *)cuMemcpyDtoD;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoDAsync)) == 0) {
        return (void *)cuMemcpyDtoDAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoH)) == 0) {
        return (void *)cuMemcpyDtoH;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoHAsync)) == 0) {
        return (void *)cuMemcpyDtoHAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyHtoA)) == 0) {
        return (void *)cuMemcpyHtoA;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyHtoAAsync)) == 0) {
        return (void *)cuMemcpyHtoAAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyHtoD)) == 0) {
        return (void *)cuMemcpyHtoD;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyHtoDAsync)) == 0) {
        return (void *)cuMemcpyHtoDAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyPeer)) == 0) {
        return (void *)cuMemcpyPeer;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyPeerAsync)) == 0) {
        return (void *)cuMemcpyPeerAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD16)) == 0) {
        return (void *)cuMemsetD16;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD16Async)) == 0) {
        return (void *)cuMemsetD16Async;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D16)) == 0) {
        return (void *)cuMemsetD2D16;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D16Async)) == 0) {
        return (void *)cuMemsetD2D16Async;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D32)) == 0) {
        return (void *)cuMemsetD2D32;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D32Async)) == 0) {
        return (void *)cuMemsetD2D32Async;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D8)) == 0) {
        return (void *)cuMemsetD2D8;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D8Async)) == 0) {
        return (void *)cuMemsetD2D8Async;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD32)) == 0) {
        return (void *)cuMemsetD32;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD32Async)) == 0) {
        return (void *)cuMemsetD32Async;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD8)) == 0) {
        return (void *)cuMemsetD8;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD8Async)) == 0) {
        return (void *)cuMemsetD8Async;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMipmappedArrayCreate)) == 0) {
        return (void *)cuMipmappedArrayCreate;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMipmappedArrayDestroy)) == 0) {
        return (void *)cuMipmappedArrayDestroy;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAddressFree)) == 0) {
        return (void *)cuMemAddressFree;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAddressReserve)) == 0) {
        return (void *)cuMemAddressReserve;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemCreate)) == 0) {
        return (void *)cuMemCreate;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemRelease)) == 0) {
        return (void *)cuMemRelease;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocAsync)) == 0) {
        return (void *)cuMemAllocAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocFromPoolAsync)) == 0) {
        return (void *)cuMemAllocFromPoolAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemFreeAsync)) == 0) {
        return (void *)cuMemFreeAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemPoolCreate)) == 0) {
        return (void *)cuMemPoolCreate;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemPoolDestroy)) == 0) {
        return (void *)cuMemPoolDestroy;
    } else if(strcmp(symbol, SYMBOL_STRING(cuLaunchCooperativeKernel)) == 0) {
        return (void *)cuLaunchCooperativeKernel;
    } else if(strcmp(symbol, SYMBOL_STRING(cuLaunchHostFunc)) == 0) {
        return (void *)cuLaunchHostFunc;
    } else if(strcmp(symbol, SYMBOL_STRING(cuLaunchKernel)) == 0) {
        return (void *)cuLaunchKernel;
    }
#endif /* _CUDA_HOOK_ENABLE */

    return actualDlsym(handle, symbol);
}
