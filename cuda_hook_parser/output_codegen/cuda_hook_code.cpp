// hook.cpp
void *dlsym(void *handle, const char *symbol)
{
    /* Hook functions for cuda version < 11.3 */
    if(strcmp(symbol, SYMBOL_STRING(cuGetProcAddress)) == 0) {
        return (void *)cuGetProcAddress;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAlloc)) == 0) {
        return (void *)cuMemAlloc;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocManaged)) == 0) {
        return (void *)cuMemAllocManaged;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocPitch)) == 0) {
        return (void *)cuMemAllocPitch;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemFree)) == 0) {
        return (void *)cuMemFree;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpy)) == 0) {
        return (void *)cuMemcpy;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyAsync)) == 0) {
        return (void *)cuMemcpyAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoD)) == 0) {
        return (void *)cuMemcpyDtoD;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoDAsync)) == 0) {
        return (void *)cuMemcpyDtoDAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoH)) == 0) {
        return (void *)cuMemcpyDtoH;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemcpyDtoHAsync)) == 0) {
        return (void *)cuMemcpyDtoHAsync;
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
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemAllocAsync)) == 0) {
        return (void *)cuMemAllocAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuMemFreeAsync)) == 0) {
        return (void *)cuMemFreeAsync;
    } else if(strcmp(symbol, SYMBOL_STRING(cuLaunchCooperativeKernel)) == 0) {
        return (void *)cuLaunchCooperativeKernel;
    } else if(strcmp(symbol, SYMBOL_STRING(cuLaunchHostFunc)) == 0) {
        return (void *)cuLaunchHostFunc;
    } else if(strcmp(symbol, SYMBOL_STRING(cuLaunchKernel)) == 0) {
        return (void *)cuLaunchKernel;
    }
}

// cuda_hook.h
enum cudaHookSymbols {
    CU_GET_PROC_ADDRESS,
    CU_MEM_ALLOC,
    CU_MEM_ALLOC_MANAGED,
    CU_MEM_ALLOC_PITCH,
    CU_MEM_FREE,
    CU_MEMCPY,
    CU_MEMCPY_ASYNC,
    CU_MEMCPY_DTO_D,
    CU_MEMCPY_DTO_D_ASYNC,
    CU_MEMCPY_DTO_H,
    CU_MEMCPY_DTO_H_ASYNC,
    CU_MEMCPY_HTO_D,
    CU_MEMCPY_HTO_D_ASYNC,
    CU_MEMCPY_PEER,
    CU_MEMCPY_PEER_ASYNC,
    CU_MEMSET_D16,
    CU_MEMSET_D16_ASYNC,
    CU_MEMSET_D2D16,
    CU_MEMSET_D2D16_ASYNC,
    CU_MEMSET_D2D32,
    CU_MEMSET_D2D32_ASYNC,
    CU_MEMSET_D2D8,
    CU_MEMSET_D2D8_ASYNC,
    CU_MEMSET_D32,
    CU_MEMSET_D32_ASYNC,
    CU_MEMSET_D8,
    CU_MEMSET_D8_ASYNC,
    CU_MEM_ALLOC_ASYNC,
    CU_MEM_FREE_ASYNC,
    CU_LAUNCH_COOPERATIVE_KERNEL,
    CU_LAUNCH_HOST_FUNC,
    CU_LAUNCH_KERNEL,
    NUM_CUDA_HOOK_SYMBOLS
};

// cuda_hook.cpp
/* ****************************** replace posthook of cuGetProcAddress() ****************************** */
/* cuGetProcAddress() is the entry of cuda api functions for cuda version >= 11.3 */
CUresult cuGetProcAddress_posthook(
    const char *symbol,
    void **pfn,
    int cudaVersion,
    cuuint64_t flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] symbol %s, cudaVersion %d, flags %lu\n", __func__, symbol, cudaVersion, flags);
    /* Hook functions for cuda version >= 11.3 */
    if(strcmp(symbol, "cuGetProcAddress") == 0) {
        cuda_hook_info.func_actual[CU_GET_PROC_ADDRESS] = *pfn;
        *pfn = (void *)cuGetProcAddress;
    }
    else if(strcmp(symbol, "cuMemAlloc") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC] = *pfn;
        *pfn = (void *)cuMemAlloc;
    }
    else if(strcmp(symbol, "cuMemAllocManaged") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_MANAGED] = *pfn;
        *pfn = (void *)cuMemAllocManaged;
    }
    else if(strcmp(symbol, "cuMemAllocPitch") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_PITCH] = *pfn;
        *pfn = (void *)cuMemAllocPitch;
    }
    else if(strcmp(symbol, "cuMemFree") == 0) {
        cuda_hook_info.func_actual[CU_MEM_FREE] = *pfn;
        *pfn = (void *)cuMemFree;
    }
    else if(strcmp(symbol, "cuMemcpy") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY] = *pfn;
        *pfn = (void *)cuMemcpy;
    }
    else if(strcmp(symbol, "cuMemcpyAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyAsync;
    }
    else if(strcmp(symbol, "cuMemcpyDtoD") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_D] = *pfn;
        *pfn = (void *)cuMemcpyDtoD;
    }
    else if(strcmp(symbol, "cuMemcpyDtoDAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_D_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyDtoDAsync;
    }
    else if(strcmp(symbol, "cuMemcpyDtoH") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_H] = *pfn;
        *pfn = (void *)cuMemcpyDtoH;
    }
    else if(strcmp(symbol, "cuMemcpyDtoHAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_H_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyDtoHAsync;
    }
    else if(strcmp(symbol, "cuMemcpyHtoD") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_D] = *pfn;
        *pfn = (void *)cuMemcpyHtoD;
    }
    else if(strcmp(symbol, "cuMemcpyHtoDAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_D_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyHtoDAsync;
    }
    else if(strcmp(symbol, "cuMemcpyPeer") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_PEER] = *pfn;
        *pfn = (void *)cuMemcpyPeer;
    }
    else if(strcmp(symbol, "cuMemcpyPeerAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_PEER_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyPeerAsync;
    }
    else if(strcmp(symbol, "cuMemsetD16") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D16] = *pfn;
        *pfn = (void *)cuMemsetD16;
    }
    else if(strcmp(symbol, "cuMemsetD16Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D16_ASYNC] = *pfn;
        *pfn = (void *)cuMemsetD16Async;
    }
    else if(strcmp(symbol, "cuMemsetD2D16") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D16] = *pfn;
        *pfn = (void *)cuMemsetD2D16;
    }
    else if(strcmp(symbol, "cuMemsetD2D16Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D16_ASYNC] = *pfn;
        *pfn = (void *)cuMemsetD2D16Async;
    }
    else if(strcmp(symbol, "cuMemsetD2D32") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D32] = *pfn;
        *pfn = (void *)cuMemsetD2D32;
    }
    else if(strcmp(symbol, "cuMemsetD2D32Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D32_ASYNC] = *pfn;
        *pfn = (void *)cuMemsetD2D32Async;
    }
    else if(strcmp(symbol, "cuMemsetD2D8") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D8] = *pfn;
        *pfn = (void *)cuMemsetD2D8;
    }
    else if(strcmp(symbol, "cuMemsetD2D8Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D2D8_ASYNC] = *pfn;
        *pfn = (void *)cuMemsetD2D8Async;
    }
    else if(strcmp(symbol, "cuMemsetD32") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D32] = *pfn;
        *pfn = (void *)cuMemsetD32;
    }
    else if(strcmp(symbol, "cuMemsetD32Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D32_ASYNC] = *pfn;
        *pfn = (void *)cuMemsetD32Async;
    }
    else if(strcmp(symbol, "cuMemsetD8") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D8] = *pfn;
        *pfn = (void *)cuMemsetD8;
    }
    else if(strcmp(symbol, "cuMemsetD8Async") == 0) {
        cuda_hook_info.func_actual[CU_MEMSET_D8_ASYNC] = *pfn;
        *pfn = (void *)cuMemsetD8Async;
    }
    else if(strcmp(symbol, "cuMemAllocAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_ASYNC] = *pfn;
        *pfn = (void *)cuMemAllocAsync;
    }
    else if(strcmp(symbol, "cuMemFreeAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEM_FREE_ASYNC] = *pfn;
        *pfn = (void *)cuMemFreeAsync;
    }
    else if(strcmp(symbol, "cuLaunchCooperativeKernel") == 0) {
        cuda_hook_info.func_actual[CU_LAUNCH_COOPERATIVE_KERNEL] = *pfn;
        *pfn = (void *)cuLaunchCooperativeKernel;
    }
    else if(strcmp(symbol, "cuLaunchHostFunc") == 0) {
        cuda_hook_info.func_actual[CU_LAUNCH_HOST_FUNC] = *pfn;
        *pfn = (void *)cuLaunchHostFunc;
    }
    else if(strcmp(symbol, "cuLaunchKernel") == 0) {
        cuda_hook_info.func_actual[CU_LAUNCH_KERNEL] = *pfn;
        *pfn = (void *)cuLaunchKernel;
    }
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}
/* ****************************** replace posthook of cuGetProcAddress() ****************************** */

/* prehook, proxy, posthook functions start */
CUresult cuGetProcAddress_prehook(
    const char *symbol,
    void **pfn,
    int cudaVersion,
    cuuint64_t flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_proxy(
    const char *symbol,
    void **pfn,
    int cudaVersion,
    cuuint64_t flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_posthook(
    const char *symbol,
    void **pfn,
    int cudaVersion,
    cuuint64_t flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuGetProcAddress\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_prehook(
    CUdeviceptr *dptr,
    size_t bytesize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_proxy(
    CUdeviceptr *dptr,
    size_t bytesize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_posthook(
    CUdeviceptr *dptr,
    size_t bytesize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemAlloc\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged_prehook(
    CUdeviceptr *dptr,
    size_t bytesize,
    unsigned int flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged_proxy(
    CUdeviceptr *dptr,
    size_t bytesize,
    unsigned int flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged_posthook(
    CUdeviceptr *dptr,
    size_t bytesize,
    unsigned int flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemAllocManaged\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch_prehook(
    CUdeviceptr *dptr,
    size_t *pPitch,
    size_t WidthInBytes,
    size_t Height,
    unsigned int ElementSizeBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch_proxy(
    CUdeviceptr *dptr,
    size_t *pPitch,
    size_t WidthInBytes,
    size_t Height,
    unsigned int ElementSizeBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch_posthook(
    CUdeviceptr *dptr,
    size_t *pPitch,
    size_t WidthInBytes,
    size_t Height,
    unsigned int ElementSizeBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemAllocPitch\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFree_prehook(
    CUdeviceptr dptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFree_proxy(
    CUdeviceptr dptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFree_posthook(
    CUdeviceptr dptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemFree\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy_prehook(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy_proxy(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy_posthook(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpy\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync_prehook(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync_proxy(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync_posthook(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyAsync\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_prehook(
    CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_proxy(
    CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_posthook(
    CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyDtoD\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoDAsync_prehook(
    CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoDAsync_proxy(
    CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoDAsync_posthook(
    CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyDtoDAsync\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_prehook(
    void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_proxy(
    void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_posthook(
    void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyDtoH\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync_prehook(
    void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync_proxy(
    void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync_posthook(
    void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyDtoHAsync\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_prehook(
    CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_proxy(
    CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_posthook(
    CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyHtoD\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoDAsync_prehook(
    CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoDAsync_proxy(
    CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoDAsync_posthook(
    CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyHtoDAsync\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeer_prehook(
    CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeer_proxy(
    CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeer_posthook(
    CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyPeer\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeerAsync_prehook(
    CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeerAsync_proxy(
    CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyPeerAsync_posthook(
    CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemcpyPeerAsync\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16_prehook(
    CUdeviceptr dstDevice,
    unsigned short us,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16_proxy(
    CUdeviceptr dstDevice,
    unsigned short us,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16_posthook(
    CUdeviceptr dstDevice,
    unsigned short us,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD16\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16Async_prehook(
    CUdeviceptr dstDevice,
    unsigned short us,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16Async_proxy(
    CUdeviceptr dstDevice,
    unsigned short us,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16Async_posthook(
    CUdeviceptr dstDevice,
    unsigned short us,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD16Async\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16_prehook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16_proxy(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16_posthook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD2D16\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16Async_prehook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16Async_proxy(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16Async_posthook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD2D16Async\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32_prehook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32_proxy(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32_posthook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD2D32\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32Async_prehook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32Async_proxy(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32Async_posthook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD2D32Async\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8_prehook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8_proxy(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8_posthook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD2D8\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8Async_prehook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8Async_proxy(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8Async_posthook(
    CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD2D8Async\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32_prehook(
    CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32_proxy(
    CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32_posthook(
    CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD32\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32Async_prehook(
    CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32Async_proxy(
    CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32Async_posthook(
    CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD32Async\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8_prehook(
    CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8_proxy(
    CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8_posthook(
    CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD8\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async_prehook(
    CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async_proxy(
    CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async_posthook(
    CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemsetD8Async\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync_prehook(
    CUdeviceptr *dptr,
    size_t bytesize,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync_proxy(
    CUdeviceptr *dptr,
    size_t bytesize,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocAsync_posthook(
    CUdeviceptr *dptr,
    size_t bytesize,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemAllocAsync\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeAsync_prehook(
    CUdeviceptr dptr,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeAsync_proxy(
    CUdeviceptr dptr,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeAsync_posthook(
    CUdeviceptr dptr,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuMemFreeAsync\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel_prehook(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel_proxy(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel_posthook(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuLaunchCooperativeKernel\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchHostFunc_prehook(
    CUstream hStream,
    CUhostFn fn,
    void *userData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchHostFunc_proxy(
    CUstream hStream,
    CUhostFn fn,
    void *userData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchHostFunc_posthook(
    CUstream hStream,
    CUhostFn fn,
    void *userData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuLaunchHostFunc\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel_prehook(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel_proxy(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel_posthook(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cuLaunchKernel\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}
/* prehook, proxy, posthook functions end */

static void cuda_hook_init()
{
    cuda_hook_info.func_prehook[CU_GET_PROC_ADDRESS] =
        (void *)cuGetProcAddress_prehook;
    cuda_hook_info.func_proxy[CU_GET_PROC_ADDRESS] =
        (void *)cuGetProcAddress_proxy;
    cuda_hook_info.func_posthook[CU_GET_PROC_ADDRESS] =
        (void *)cuGetProcAddress_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC] =
        (void *)cuMemAlloc_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC] =
        (void *)cuMemAlloc_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC] =
        (void *)cuMemAlloc_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_MANAGED] =
        (void *)cuMemAllocManaged_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_MANAGED] =
        (void *)cuMemAllocManaged_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_MANAGED] =
        (void *)cuMemAllocManaged_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_PITCH] =
        (void *)cuMemAllocPitch_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_PITCH] =
        (void *)cuMemAllocPitch_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_PITCH] =
        (void *)cuMemAllocPitch_posthook;
    cuda_hook_info.func_prehook[CU_MEM_FREE] =
        (void *)cuMemFree_prehook;
    cuda_hook_info.func_proxy[CU_MEM_FREE] =
        (void *)cuMemFree_proxy;
    cuda_hook_info.func_posthook[CU_MEM_FREE] =
        (void *)cuMemFree_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY] =
        (void *)cuMemcpy_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY] =
        (void *)cuMemcpy_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY] =
        (void *)cuMemcpy_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_ASYNC] =
        (void *)cuMemcpyAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_ASYNC] =
        (void *)cuMemcpyAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_ASYNC] =
        (void *)cuMemcpyAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_D] =
        (void *)cuMemcpyDtoD_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_D] =
        (void *)cuMemcpyDtoD_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_D] =
        (void *)cuMemcpyDtoD_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_D_ASYNC] =
        (void *)cuMemcpyDtoDAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_D_ASYNC] =
        (void *)cuMemcpyDtoDAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_D_ASYNC] =
        (void *)cuMemcpyDtoDAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_H] =
        (void *)cuMemcpyDtoH_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_H] =
        (void *)cuMemcpyDtoH_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_H] =
        (void *)cuMemcpyDtoH_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_H_ASYNC] =
        (void *)cuMemcpyDtoHAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_H_ASYNC] =
        (void *)cuMemcpyDtoHAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_H_ASYNC] =
        (void *)cuMemcpyDtoHAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_HTO_D] =
        (void *)cuMemcpyHtoD_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_HTO_D] =
        (void *)cuMemcpyHtoD_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_HTO_D] =
        (void *)cuMemcpyHtoD_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_HTO_D_ASYNC] =
        (void *)cuMemcpyHtoDAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_HTO_D_ASYNC] =
        (void *)cuMemcpyHtoDAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_HTO_D_ASYNC] =
        (void *)cuMemcpyHtoDAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_PEER] =
        (void *)cuMemcpyPeer_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_PEER] =
        (void *)cuMemcpyPeer_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_PEER] =
        (void *)cuMemcpyPeer_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_PEER_ASYNC] =
        (void *)cuMemcpyPeerAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_PEER_ASYNC] =
        (void *)cuMemcpyPeerAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_PEER_ASYNC] =
        (void *)cuMemcpyPeerAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D16] =
        (void *)cuMemsetD16_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D16] =
        (void *)cuMemsetD16_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D16] =
        (void *)cuMemsetD16_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D16_ASYNC] =
        (void *)cuMemsetD16Async_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D16_ASYNC] =
        (void *)cuMemsetD16Async_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D16_ASYNC] =
        (void *)cuMemsetD16Async_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D2D16] =
        (void *)cuMemsetD2D16_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D2D16] =
        (void *)cuMemsetD2D16_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D2D16] =
        (void *)cuMemsetD2D16_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D2D16_ASYNC] =
        (void *)cuMemsetD2D16Async_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D2D16_ASYNC] =
        (void *)cuMemsetD2D16Async_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D2D16_ASYNC] =
        (void *)cuMemsetD2D16Async_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D2D32] =
        (void *)cuMemsetD2D32_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D2D32] =
        (void *)cuMemsetD2D32_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D2D32] =
        (void *)cuMemsetD2D32_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D2D32_ASYNC] =
        (void *)cuMemsetD2D32Async_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D2D32_ASYNC] =
        (void *)cuMemsetD2D32Async_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D2D32_ASYNC] =
        (void *)cuMemsetD2D32Async_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D2D8] =
        (void *)cuMemsetD2D8_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D2D8] =
        (void *)cuMemsetD2D8_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D2D8] =
        (void *)cuMemsetD2D8_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D2D8_ASYNC] =
        (void *)cuMemsetD2D8Async_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D2D8_ASYNC] =
        (void *)cuMemsetD2D8Async_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D2D8_ASYNC] =
        (void *)cuMemsetD2D8Async_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D32] =
        (void *)cuMemsetD32_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D32] =
        (void *)cuMemsetD32_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D32] =
        (void *)cuMemsetD32_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D32_ASYNC] =
        (void *)cuMemsetD32Async_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D32_ASYNC] =
        (void *)cuMemsetD32Async_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D32_ASYNC] =
        (void *)cuMemsetD32Async_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D8] =
        (void *)cuMemsetD8_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D8] =
        (void *)cuMemsetD8_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D8] =
        (void *)cuMemsetD8_posthook;
    cuda_hook_info.func_prehook[CU_MEMSET_D8_ASYNC] =
        (void *)cuMemsetD8Async_prehook;
    cuda_hook_info.func_proxy[CU_MEMSET_D8_ASYNC] =
        (void *)cuMemsetD8Async_proxy;
    cuda_hook_info.func_posthook[CU_MEMSET_D8_ASYNC] =
        (void *)cuMemsetD8Async_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_ASYNC] =
        (void *)cuMemAllocAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_ASYNC] =
        (void *)cuMemAllocAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_ASYNC] =
        (void *)cuMemAllocAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEM_FREE_ASYNC] =
        (void *)cuMemFreeAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEM_FREE_ASYNC] =
        (void *)cuMemFreeAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEM_FREE_ASYNC] =
        (void *)cuMemFreeAsync_posthook;
    cuda_hook_info.func_prehook[CU_LAUNCH_COOPERATIVE_KERNEL] =
        (void *)cuLaunchCooperativeKernel_prehook;
    cuda_hook_info.func_proxy[CU_LAUNCH_COOPERATIVE_KERNEL] =
        (void *)cuLaunchCooperativeKernel_proxy;
    cuda_hook_info.func_posthook[CU_LAUNCH_COOPERATIVE_KERNEL] =
        (void *)cuLaunchCooperativeKernel_posthook;
    cuda_hook_info.func_prehook[CU_LAUNCH_HOST_FUNC] =
        (void *)cuLaunchHostFunc_prehook;
    cuda_hook_info.func_proxy[CU_LAUNCH_HOST_FUNC] =
        (void *)cuLaunchHostFunc_proxy;
    cuda_hook_info.func_posthook[CU_LAUNCH_HOST_FUNC] =
        (void *)cuLaunchHostFunc_posthook;
    cuda_hook_info.func_prehook[CU_LAUNCH_KERNEL] =
        (void *)cuLaunchKernel_prehook;
    cuda_hook_info.func_proxy[CU_LAUNCH_KERNEL] =
        (void *)cuLaunchKernel_proxy;
    cuda_hook_info.func_posthook[CU_LAUNCH_KERNEL] =
        (void *)cuLaunchKernel_posthook;
}

/* hook function start */
CUDA_HOOK_GEN(
    CU_GET_PROC_ADDRESS,
    ,
    cuGetProcAddress,
    (const char *symbol,
    void **pfn,
    int cudaVersion,
    cuuint64_t flags),
    symbol, pfn, cudaVersion, flags)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC,
    ,
    cuMemAlloc,
    (CUdeviceptr *dptr,
    size_t bytesize),
    dptr, bytesize)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_MANAGED,
    ,
    cuMemAllocManaged,
    (CUdeviceptr *dptr,
    size_t bytesize,
    unsigned int flags),
    dptr, bytesize, flags)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_PITCH,
    ,
    cuMemAllocPitch,
    (CUdeviceptr *dptr,
    size_t *pPitch,
    size_t WidthInBytes,
    size_t Height,
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
    (CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount),
    dst, src, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_ASYNC,
    ,
    cuMemcpyAsync,
    (CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount,
    CUstream hStream),
    dst, src, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_D,
    ,
    cuMemcpyDtoD,
    (CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount),
    dstDevice, srcDevice, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_D_ASYNC,
    ,
    cuMemcpyDtoDAsync,
    (CUdeviceptr dstDevice,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream),
    dstDevice, srcDevice, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_H,
    ,
    cuMemcpyDtoH,
    (void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount),
    dstHost, srcDevice, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_H_ASYNC,
    ,
    cuMemcpyDtoHAsync,
    (void *dstHost,
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream),
    dstHost, srcDevice, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_HTO_D,
    ,
    cuMemcpyHtoD,
    (CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount),
    dstDevice, srcHost, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_HTO_D_ASYNC,
    ,
    cuMemcpyHtoDAsync,
    (CUdeviceptr dstDevice,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream),
    dstDevice, srcHost, ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_PEER,
    ,
    cuMemcpyPeer,
    (CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount),
    dstDevice, dstContext, srcDevice, srcContext,
    ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_PEER_ASYNC,
    ,
    cuMemcpyPeerAsync,
    (CUdeviceptr dstDevice,
    CUcontext dstContext,
    CUdeviceptr srcDevice,
    CUcontext srcContext,
    size_t ByteCount,
    CUstream hStream),
    dstDevice, dstContext, srcDevice, srcContext,
    ByteCount, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D16,
    ,
    cuMemsetD16,
    (CUdeviceptr dstDevice,
    unsigned short us,
    size_t N),
    dstDevice, us, N)

CUDA_HOOK_GEN(
    CU_MEMSET_D16_ASYNC,
    ,
    cuMemsetD16Async,
    (CUdeviceptr dstDevice,
    unsigned short us,
    size_t N,
    CUstream hStream),
    dstDevice, us, N, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D16,
    ,
    cuMemsetD2D16,
    (CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height),
    dstDevice, dstPitch, us, Width,
    Height)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D16_ASYNC,
    ,
    cuMemsetD2D16Async,
    (CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned short us,
    size_t Width,
    size_t Height,
    CUstream hStream),
    dstDevice, dstPitch, us, Width,
    Height, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D32,
    ,
    cuMemsetD2D32,
    (CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height),
    dstDevice, dstPitch, ui, Width,
    Height)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D32_ASYNC,
    ,
    cuMemsetD2D32Async,
    (CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned int ui,
    size_t Width,
    size_t Height,
    CUstream hStream),
    dstDevice, dstPitch, ui, Width,
    Height, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D8,
    ,
    cuMemsetD2D8,
    (CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height),
    dstDevice, dstPitch, uc, Width,
    Height)

CUDA_HOOK_GEN(
    CU_MEMSET_D2D8_ASYNC,
    ,
    cuMemsetD2D8Async,
    (CUdeviceptr dstDevice,
    size_t dstPitch,
    unsigned char uc,
    size_t Width,
    size_t Height,
    CUstream hStream),
    dstDevice, dstPitch, uc, Width,
    Height, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D32,
    ,
    cuMemsetD32,
    (CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N),
    dstDevice, ui, N)

CUDA_HOOK_GEN(
    CU_MEMSET_D32_ASYNC,
    ,
    cuMemsetD32Async,
    (CUdeviceptr dstDevice,
    unsigned int ui,
    size_t N,
    CUstream hStream),
    dstDevice, ui, N, hStream)

CUDA_HOOK_GEN(
    CU_MEMSET_D8,
    ,
    cuMemsetD8,
    (CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N),
    dstDevice, uc, N)

CUDA_HOOK_GEN(
    CU_MEMSET_D8_ASYNC,
    ,
    cuMemsetD8Async,
    (CUdeviceptr dstDevice,
    unsigned char uc,
    size_t N,
    CUstream hStream),
    dstDevice, uc, N, hStream)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_ASYNC,
    ,
    cuMemAllocAsync,
    (CUdeviceptr *dptr,
    size_t bytesize,
    CUstream hStream),
    dptr, bytesize, hStream)

CUDA_HOOK_GEN(
    CU_MEM_FREE_ASYNC,
    ,
    cuMemFreeAsync,
    (CUdeviceptr dptr,
    CUstream hStream),
    dptr, hStream)

CUDA_HOOK_GEN(
    CU_LAUNCH_COOPERATIVE_KERNEL,
    ,
    cuLaunchCooperativeKernel,
    (CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams),
    f, gridDimX, gridDimY, gridDimZ,
    blockDimX, blockDimY, blockDimZ, sharedMemBytes,
    hStream, kernelParams)

CUDA_HOOK_GEN(
    CU_LAUNCH_HOST_FUNC,
    ,
    cuLaunchHostFunc,
    (CUstream hStream,
    CUhostFn fn,
    void *userData),
    hStream, fn, userData)

CUDA_HOOK_GEN(
    CU_LAUNCH_KERNEL,
    ,
    cuLaunchKernel,
    (CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra),
    f, gridDimX, gridDimY, gridDimZ,
    blockDimX, blockDimY, blockDimZ, sharedMemBytes,
    hStream, kernelParams, extra)
/* hook function end */
