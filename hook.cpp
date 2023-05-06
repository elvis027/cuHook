#define __USE_GNU
#include <iostream>
#include <string>
#include <cstring>
#include <dlfcn.h>

#include <cuda.h>
#include <cudnn.h>

#include "hook.hpp"
#include "logging.hpp"

using namespace std::string_literals;
using std::string;

void *libdl_handle;
void *libcuda_handle;
void *libcudnn_handle;

logging::log hook_log("hook_log.txt");
logging::log trace_dump("trace.txt");

__attribute__((constructor))
void hook_init()
{
    libdl_handle = dlopen("libdl.so", RTLD_LAZY);
    libcuda_handle = dlopen("libcuda.so", RTLD_LAZY);
    libcudnn_handle = dlopen("libcudnn.so", RTLD_LAZY);
}

void *actual_dlsym(void *handle, const char *symbol)
{
    typedef decltype(&dlsym) func_type;
    void *dlsym_func = dlvsym(libdl_handle, "dlsym", "GLIBC_2.2.5");

    void *actual_func = ((func_type)dlsym_func)(handle, symbol);
    if(!actual_func)
        hook_log.error("dlsym load function failed: "s + string(symbol));
    else
        hook_log.debug("dlsym load function: "s + string(symbol));

    return actual_func;
}

void *dlsym(void *handle, const char *symbol)
{
    hook_log.debug("dlsym hook function: "s + string(symbol));

    if(strncmp(symbol, "cu", 2) != 0)
        return actual_dlsym(handle, symbol);

    /* Hook functions for cuda version < 11.3 */
    if(strcmp(symbol, SYMBOL_STRING(cuGetProcAddress)) == 0) {
        return reinterpret_cast<void *>(cuGetProcAddress);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemAlloc)) == 0) {
        return reinterpret_cast<void *>(cuMemAlloc);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocManaged)) == 0) {
        return reinterpret_cast<void *>(cuMemAllocManaged);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocPitch)) == 0) {
        return reinterpret_cast<void *>(cuMemAllocPitch);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemFree)) == 0) {
        return reinterpret_cast<void *>(cuMemFree);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy)) == 0) {
        return reinterpret_cast<void *>(cuMemcpy);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyAsync)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyAsync);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoD)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyDtoD);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoDAsync)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyDtoDAsync);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoH)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyDtoH);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoHAsync)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyDtoHAsync);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyHtoD)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyHtoD);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyHtoDAsync)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyHtoDAsync);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyPeer)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyPeer);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyPeerAsync)) == 0) {
        return reinterpret_cast<void *>(cuMemcpyPeerAsync);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD16)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD16);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD16Async)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD16Async);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D16)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD2D16);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D16Async)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD2D16Async);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D32)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD2D32);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D32Async)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD2D32Async);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D8)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD2D8);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD2D8Async)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD2D8Async);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD32)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD32);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD32Async)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD32Async);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD8)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD8);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemsetD8Async)) == 0) {
        return reinterpret_cast<void *>(cuMemsetD8Async);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocAsync)) == 0) {
        return reinterpret_cast<void *>(cuMemAllocAsync);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuMemFreeAsync)) == 0) {
        return reinterpret_cast<void *>(cuMemFreeAsync);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuLaunchCooperativeKernel)) == 0) {
        return reinterpret_cast<void *>(cuLaunchCooperativeKernel);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuLaunchHostFunc)) == 0) {
        return reinterpret_cast<void *>(cuLaunchHostFunc);
    }
    else if(strcmp(symbol, SYMBOL_STRING(cuLaunchKernel)) == 0) {
        return reinterpret_cast<void *>(cuLaunchKernel);
    }

    return actual_dlsym(handle, symbol);
}
