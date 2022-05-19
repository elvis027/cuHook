#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#include <cuda.h>

#include "hook.h"
#include "cuda_hook.h"
#include "debug.h"

#ifdef _CUDA_HOOK_ENABLE

static struct cudaHookInfo cuda_hook_info;
static pthread_once_t cuda_hook_init_done = PTHREAD_ONCE_INIT;

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
    else if(strcmp(symbol, "cuArray3DCreate") == 0) {
        cuda_hook_info.func_actual[CU_ARRAY_3D_CREATE] = *pfn;
        *pfn = (void *)cuArray3DCreate;
    }
    else if(strcmp(symbol, "cuArrayCreate") == 0) {
        cuda_hook_info.func_actual[CU_ARRAY_CREATE] = *pfn;
        *pfn = (void *)cuArrayCreate;
    }
    else if(strcmp(symbol, "cuArrayDestroy") == 0) {
        cuda_hook_info.func_actual[CU_ARRAY_DESTROY] = *pfn;
        *pfn = (void *)cuArrayDestroy;
    }
    else if(strcmp(symbol, "cuMemAlloc") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC] = *pfn;
        *pfn = (void *)cuMemAlloc;
    }
    else if(strcmp(symbol, "cuMemAllocHost") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_HOST] = *pfn;
        *pfn = (void *)cuMemAllocHost;
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
    else if(strcmp(symbol, "cuMemFreeHost") == 0) {
        cuda_hook_info.func_actual[CU_MEM_FREE_HOST] = *pfn;
        *pfn = (void *)cuMemFreeHost;
    }
    else if(strcmp(symbol, "cuMemHostAlloc") == 0) {
        cuda_hook_info.func_actual[CU_MEM_HOST_ALLOC] = *pfn;
        *pfn = (void *)cuMemHostAlloc;
    }
    else if(strcmp(symbol, "cuMemcpy") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY] = *pfn;
        *pfn = (void *)cuMemcpy;
    }
    else if(strcmp(symbol, "cuMemcpy2D") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_2D] = *pfn;
        *pfn = (void *)cuMemcpy2D;
    }
    else if(strcmp(symbol, "cuMemcpy2DAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_2D_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpy2DAsync;
    }
    else if(strcmp(symbol, "cuMemcpy2DUnaligned") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_2D_UNALIGNED] = *pfn;
        *pfn = (void *)cuMemcpy2DUnaligned;
    }
    else if(strcmp(symbol, "cuMemcpy3D") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_3D] = *pfn;
        *pfn = (void *)cuMemcpy3D;
    }
    else if(strcmp(symbol, "cuMemcpy3DAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_3D_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpy3DAsync;
    }
    else if(strcmp(symbol, "cuMemcpy3DPeer") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_3D_PEER] = *pfn;
        *pfn = (void *)cuMemcpy3DPeer;
    }
    else if(strcmp(symbol, "cuMemcpy3DPeerAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_3D_PEER_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpy3DPeerAsync;
    }
    else if(strcmp(symbol, "cuMemcpyAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyAsync;
    }
    else if(strcmp(symbol, "cuMemcpyAtoA") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_ATO_A] = *pfn;
        *pfn = (void *)cuMemcpyAtoA;
    }
    else if(strcmp(symbol, "cuMemcpyAtoD") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_ATO_D] = *pfn;
        *pfn = (void *)cuMemcpyAtoD;
    }
    else if(strcmp(symbol, "cuMemcpyAtoH") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_ATO_H] = *pfn;
        *pfn = (void *)cuMemcpyAtoH;
    }
    else if(strcmp(symbol, "cuMemcpyAtoHAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_ATO_H_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyAtoHAsync;
    }
    else if(strcmp(symbol, "cuMemcpyDtoA") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_DTO_A] = *pfn;
        *pfn = (void *)cuMemcpyDtoA;
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
    else if(strcmp(symbol, "cuMemcpyHtoA") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_A] = *pfn;
        *pfn = (void *)cuMemcpyHtoA;
    }
    else if(strcmp(symbol, "cuMemcpyHtoAAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEMCPY_HTO_A_ASYNC] = *pfn;
        *pfn = (void *)cuMemcpyHtoAAsync;
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
    else if(strcmp(symbol, "cuMipmappedArrayCreate") == 0) {
        cuda_hook_info.func_actual[CU_MIPMAPPED_ARRAY_CREATE] = *pfn;
        *pfn = (void *)cuMipmappedArrayCreate;
    }
    else if(strcmp(symbol, "cuMipmappedArrayDestroy") == 0) {
        cuda_hook_info.func_actual[CU_MIPMAPPED_ARRAY_DESTROY] = *pfn;
        *pfn = (void *)cuMipmappedArrayDestroy;
    }
    else if(strcmp(symbol, "cuMemAddressFree") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ADDRESS_FREE] = *pfn;
        *pfn = (void *)cuMemAddressFree;
    }
    else if(strcmp(symbol, "cuMemAddressReserve") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ADDRESS_RESERVE] = *pfn;
        *pfn = (void *)cuMemAddressReserve;
    }
    else if(strcmp(symbol, "cuMemCreate") == 0) {
        cuda_hook_info.func_actual[CU_MEM_CREATE] = *pfn;
        *pfn = (void *)cuMemCreate;
    }
    else if(strcmp(symbol, "cuMemRelease") == 0) {
        cuda_hook_info.func_actual[CU_MEM_RELEASE] = *pfn;
        *pfn = (void *)cuMemRelease;
    }
    else if(strcmp(symbol, "cuMemAllocAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_ASYNC] = *pfn;
        *pfn = (void *)cuMemAllocAsync;
    }
    else if(strcmp(symbol, "cuMemAllocFromPoolAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEM_ALLOC_FROM_POOL_ASYNC] = *pfn;
        *pfn = (void *)cuMemAllocFromPoolAsync;
    }
    else if(strcmp(symbol, "cuMemFreeAsync") == 0) {
        cuda_hook_info.func_actual[CU_MEM_FREE_ASYNC] = *pfn;
        *pfn = (void *)cuMemFreeAsync;
    }
    else if(strcmp(symbol, "cuMemPoolCreate") == 0) {
        cuda_hook_info.func_actual[CU_MEM_POOL_CREATE] = *pfn;
        *pfn = (void *)cuMemPoolCreate;
    }
    else if(strcmp(symbol, "cuMemPoolDestroy") == 0) {
        cuda_hook_info.func_actual[CU_MEM_POOL_DESTROY] = *pfn;
        *pfn = (void *)cuMemPoolDestroy;
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

CUresult cuArray3DCreate_prehook(
    CUarray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArray3DCreate_proxy(
    CUarray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArray3DCreate_posthook(
    CUarray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArrayCreate_prehook(
    CUarray *pHandle,
    const CUDA_ARRAY_DESCRIPTOR *pAllocateArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArrayCreate_proxy(
    CUarray *pHandle,
    const CUDA_ARRAY_DESCRIPTOR *pAllocateArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArrayCreate_posthook(
    CUarray *pHandle,
    const CUDA_ARRAY_DESCRIPTOR *pAllocateArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArrayDestroy_prehook(
    CUarray hArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArrayDestroy_proxy(
    CUarray hArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuArrayDestroy_posthook(
    CUarray hArray
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost_prehook(
    void **pp,
    size_t bytesize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost_proxy(
    void **pp,
    size_t bytesize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost_posthook(
    void **pp,
    size_t bytesize
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost_prehook(
    void *p
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost_proxy(
    void *p
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost_posthook(
    void *p
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemHostAlloc_prehook(
    void **pp,
    size_t bytesize,
    unsigned int Flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemHostAlloc_proxy(
    void **pp,
    size_t bytesize,
    unsigned int Flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemHostAlloc_posthook(
    void **pp,
    size_t bytesize,
    unsigned int Flags
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2D_prehook(
    const CUDA_MEMCPY2D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2D_proxy(
    const CUDA_MEMCPY2D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2D_posthook(
    const CUDA_MEMCPY2D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DAsync_prehook(
    const CUDA_MEMCPY2D *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DAsync_proxy(
    const CUDA_MEMCPY2D *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DAsync_posthook(
    const CUDA_MEMCPY2D *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DUnaligned_prehook(
    const CUDA_MEMCPY2D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DUnaligned_proxy(
    const CUDA_MEMCPY2D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DUnaligned_posthook(
    const CUDA_MEMCPY2D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3D_prehook(
    const CUDA_MEMCPY3D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3D_proxy(
    const CUDA_MEMCPY3D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3D_posthook(
    const CUDA_MEMCPY3D *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DAsync_prehook(
    const CUDA_MEMCPY3D *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DAsync_proxy(
    const CUDA_MEMCPY3D *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DAsync_posthook(
    const CUDA_MEMCPY3D *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DPeer_prehook(
    const CUDA_MEMCPY3D_PEER *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DPeer_proxy(
    const CUDA_MEMCPY3D_PEER *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DPeer_posthook(
    const CUDA_MEMCPY3D_PEER *pCopy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DPeerAsync_prehook(
    const CUDA_MEMCPY3D_PEER *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DPeerAsync_proxy(
    const CUDA_MEMCPY3D_PEER *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DPeerAsync_posthook(
    const CUDA_MEMCPY3D_PEER *pCopy,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoA_prehook(
    CUarray dstArray,
    size_t dstOffset,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoA_proxy(
    CUarray dstArray,
    size_t dstOffset,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoA_posthook(
    CUarray dstArray,
    size_t dstOffset,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoD_prehook(
    CUdeviceptr dstDevice,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoD_proxy(
    CUdeviceptr dstDevice,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoD_posthook(
    CUdeviceptr dstDevice,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoH_prehook(
    void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoH_proxy(
    void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoH_posthook(
    void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoHAsync_prehook(
    void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoHAsync_proxy(
    void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAtoHAsync_posthook(
    void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoA_prehook(
    CUarray dstArray,
    size_t dstOffset,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoA_proxy(
    CUarray dstArray,
    size_t dstOffset,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoA_posthook(
    CUarray dstArray,
    size_t dstOffset,
    CUdeviceptr srcDevice,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoA_prehook(
    CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoA_proxy(
    CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoA_posthook(
    CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoAAsync_prehook(
    CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoAAsync_proxy(
    CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoAAsync_posthook(
    CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayCreate_prehook(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayCreate_proxy(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayCreate_posthook(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayDestroy_prehook(
    CUmipmappedArray hMipmappedArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayDestroy_proxy(
    CUmipmappedArray hMipmappedArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayDestroy_posthook(
    CUmipmappedArray hMipmappedArray
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree_prehook(
    CUdeviceptr ptr,
    size_t size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree_proxy(
    CUdeviceptr ptr,
    size_t size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree_posthook(
    CUdeviceptr ptr,
    size_t size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve_prehook(
    CUdeviceptr *ptr,
    size_t size,
    size_t alignment,
    CUdeviceptr addr,
    unsigned long long flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve_proxy(
    CUdeviceptr *ptr,
    size_t size,
    size_t alignment,
    CUdeviceptr addr,
    unsigned long long flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve_posthook(
    CUdeviceptr *ptr,
    size_t size,
    size_t alignment,
    CUdeviceptr addr,
    unsigned long long flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemCreate_prehook(
    CUmemGenericAllocationHandle *handle,
    size_t size,
    const CUmemAllocationProp *prop,
    unsigned long long flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemCreate_proxy(
    CUmemGenericAllocationHandle *handle,
    size_t size,
    const CUmemAllocationProp *prop,
    unsigned long long flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemCreate_posthook(
    CUmemGenericAllocationHandle *handle,
    size_t size,
    const CUmemAllocationProp *prop,
    unsigned long long flags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemRelease_prehook(
    CUmemGenericAllocationHandle handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemRelease_proxy(
    CUmemGenericAllocationHandle handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemRelease_posthook(
    CUmemGenericAllocationHandle handle
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocFromPoolAsync_prehook(
    CUdeviceptr *dptr,
    size_t bytesize,
    CUmemoryPool pool,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocFromPoolAsync_proxy(
    CUdeviceptr *dptr,
    size_t bytesize,
    CUmemoryPool pool,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocFromPoolAsync_posthook(
    CUdeviceptr *dptr,
    size_t bytesize,
    CUmemoryPool pool,
    CUstream hStream
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolCreate_prehook(
    CUmemoryPool *pool,
    const CUmemPoolProps *poolProps
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolCreate_proxy(
    CUmemoryPool *pool,
    const CUmemPoolProps *poolProps
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolCreate_posthook(
    CUmemoryPool *pool,
    const CUmemPoolProps *poolProps
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy_prehook(
    CUmemoryPool pool
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy_proxy(
    CUmemoryPool pool
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy_posthook(
    CUmemoryPool pool
) {
    DEBUG("[%s] Enter func\n", __func__);
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
    cuda_hook_info.func_prehook[CU_ARRAY_3D_CREATE] =
        (void *)cuArray3DCreate_prehook;
    cuda_hook_info.func_proxy[CU_ARRAY_3D_CREATE] =
        (void *)cuArray3DCreate_proxy;
    cuda_hook_info.func_posthook[CU_ARRAY_3D_CREATE] =
        (void *)cuArray3DCreate_posthook;
    cuda_hook_info.func_prehook[CU_ARRAY_CREATE] =
        (void *)cuArrayCreate_prehook;
    cuda_hook_info.func_proxy[CU_ARRAY_CREATE] =
        (void *)cuArrayCreate_proxy;
    cuda_hook_info.func_posthook[CU_ARRAY_CREATE] =
        (void *)cuArrayCreate_posthook;
    cuda_hook_info.func_prehook[CU_ARRAY_DESTROY] =
        (void *)cuArrayDestroy_prehook;
    cuda_hook_info.func_proxy[CU_ARRAY_DESTROY] =
        (void *)cuArrayDestroy_proxy;
    cuda_hook_info.func_posthook[CU_ARRAY_DESTROY] =
        (void *)cuArrayDestroy_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC] =
        (void *)cuMemAlloc_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC] =
        (void *)cuMemAlloc_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC] =
        (void *)cuMemAlloc_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_HOST] =
        (void *)cuMemAllocHost_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_HOST] =
        (void *)cuMemAllocHost_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_HOST] =
        (void *)cuMemAllocHost_posthook;
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
    cuda_hook_info.func_prehook[CU_MEM_FREE_HOST] =
        (void *)cuMemFreeHost_prehook;
    cuda_hook_info.func_proxy[CU_MEM_FREE_HOST] =
        (void *)cuMemFreeHost_proxy;
    cuda_hook_info.func_posthook[CU_MEM_FREE_HOST] =
        (void *)cuMemFreeHost_posthook;
    cuda_hook_info.func_prehook[CU_MEM_HOST_ALLOC] =
        (void *)cuMemHostAlloc_prehook;
    cuda_hook_info.func_proxy[CU_MEM_HOST_ALLOC] =
        (void *)cuMemHostAlloc_proxy;
    cuda_hook_info.func_posthook[CU_MEM_HOST_ALLOC] =
        (void *)cuMemHostAlloc_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY] =
        (void *)cuMemcpy_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY] =
        (void *)cuMemcpy_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY] =
        (void *)cuMemcpy_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_2D] =
        (void *)cuMemcpy2D_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_2D] =
        (void *)cuMemcpy2D_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_2D] =
        (void *)cuMemcpy2D_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_2D_ASYNC] =
        (void *)cuMemcpy2DAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_2D_ASYNC] =
        (void *)cuMemcpy2DAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_2D_ASYNC] =
        (void *)cuMemcpy2DAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_2D_UNALIGNED] =
        (void *)cuMemcpy2DUnaligned_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_2D_UNALIGNED] =
        (void *)cuMemcpy2DUnaligned_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_2D_UNALIGNED] =
        (void *)cuMemcpy2DUnaligned_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_3D] =
        (void *)cuMemcpy3D_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_3D] =
        (void *)cuMemcpy3D_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_3D] =
        (void *)cuMemcpy3D_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_3D_ASYNC] =
        (void *)cuMemcpy3DAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_3D_ASYNC] =
        (void *)cuMemcpy3DAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_3D_ASYNC] =
        (void *)cuMemcpy3DAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_3D_PEER] =
        (void *)cuMemcpy3DPeer_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_3D_PEER] =
        (void *)cuMemcpy3DPeer_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_3D_PEER] =
        (void *)cuMemcpy3DPeer_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_3D_PEER_ASYNC] =
        (void *)cuMemcpy3DPeerAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_3D_PEER_ASYNC] =
        (void *)cuMemcpy3DPeerAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_3D_PEER_ASYNC] =
        (void *)cuMemcpy3DPeerAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_ASYNC] =
        (void *)cuMemcpyAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_ASYNC] =
        (void *)cuMemcpyAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_ASYNC] =
        (void *)cuMemcpyAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_ATO_A] =
        (void *)cuMemcpyAtoA_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_ATO_A] =
        (void *)cuMemcpyAtoA_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_ATO_A] =
        (void *)cuMemcpyAtoA_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_ATO_D] =
        (void *)cuMemcpyAtoD_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_ATO_D] =
        (void *)cuMemcpyAtoD_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_ATO_D] =
        (void *)cuMemcpyAtoD_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_ATO_H] =
        (void *)cuMemcpyAtoH_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_ATO_H] =
        (void *)cuMemcpyAtoH_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_ATO_H] =
        (void *)cuMemcpyAtoH_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_ATO_H_ASYNC] =
        (void *)cuMemcpyAtoHAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_ATO_H_ASYNC] =
        (void *)cuMemcpyAtoHAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_ATO_H_ASYNC] =
        (void *)cuMemcpyAtoHAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_DTO_A] =
        (void *)cuMemcpyDtoA_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_DTO_A] =
        (void *)cuMemcpyDtoA_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_DTO_A] =
        (void *)cuMemcpyDtoA_posthook;
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
    cuda_hook_info.func_prehook[CU_MEMCPY_HTO_A] =
        (void *)cuMemcpyHtoA_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_HTO_A] =
        (void *)cuMemcpyHtoA_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_HTO_A] =
        (void *)cuMemcpyHtoA_posthook;
    cuda_hook_info.func_prehook[CU_MEMCPY_HTO_A_ASYNC] =
        (void *)cuMemcpyHtoAAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEMCPY_HTO_A_ASYNC] =
        (void *)cuMemcpyHtoAAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEMCPY_HTO_A_ASYNC] =
        (void *)cuMemcpyHtoAAsync_posthook;
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
    cuda_hook_info.func_prehook[CU_MIPMAPPED_ARRAY_CREATE] =
        (void *)cuMipmappedArrayCreate_prehook;
    cuda_hook_info.func_proxy[CU_MIPMAPPED_ARRAY_CREATE] =
        (void *)cuMipmappedArrayCreate_proxy;
    cuda_hook_info.func_posthook[CU_MIPMAPPED_ARRAY_CREATE] =
        (void *)cuMipmappedArrayCreate_posthook;
    cuda_hook_info.func_prehook[CU_MIPMAPPED_ARRAY_DESTROY] =
        (void *)cuMipmappedArrayDestroy_prehook;
    cuda_hook_info.func_proxy[CU_MIPMAPPED_ARRAY_DESTROY] =
        (void *)cuMipmappedArrayDestroy_proxy;
    cuda_hook_info.func_posthook[CU_MIPMAPPED_ARRAY_DESTROY] =
        (void *)cuMipmappedArrayDestroy_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ADDRESS_FREE] =
        (void *)cuMemAddressFree_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ADDRESS_FREE] =
        (void *)cuMemAddressFree_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ADDRESS_FREE] =
        (void *)cuMemAddressFree_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ADDRESS_RESERVE] =
        (void *)cuMemAddressReserve_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ADDRESS_RESERVE] =
        (void *)cuMemAddressReserve_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ADDRESS_RESERVE] =
        (void *)cuMemAddressReserve_posthook;
    cuda_hook_info.func_prehook[CU_MEM_CREATE] =
        (void *)cuMemCreate_prehook;
    cuda_hook_info.func_proxy[CU_MEM_CREATE] =
        (void *)cuMemCreate_proxy;
    cuda_hook_info.func_posthook[CU_MEM_CREATE] =
        (void *)cuMemCreate_posthook;
    cuda_hook_info.func_prehook[CU_MEM_RELEASE] =
        (void *)cuMemRelease_prehook;
    cuda_hook_info.func_proxy[CU_MEM_RELEASE] =
        (void *)cuMemRelease_proxy;
    cuda_hook_info.func_posthook[CU_MEM_RELEASE] =
        (void *)cuMemRelease_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_ASYNC] =
        (void *)cuMemAllocAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_ASYNC] =
        (void *)cuMemAllocAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_ASYNC] =
        (void *)cuMemAllocAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEM_ALLOC_FROM_POOL_ASYNC] =
        (void *)cuMemAllocFromPoolAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEM_ALLOC_FROM_POOL_ASYNC] =
        (void *)cuMemAllocFromPoolAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEM_ALLOC_FROM_POOL_ASYNC] =
        (void *)cuMemAllocFromPoolAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEM_FREE_ASYNC] =
        (void *)cuMemFreeAsync_prehook;
    cuda_hook_info.func_proxy[CU_MEM_FREE_ASYNC] =
        (void *)cuMemFreeAsync_proxy;
    cuda_hook_info.func_posthook[CU_MEM_FREE_ASYNC] =
        (void *)cuMemFreeAsync_posthook;
    cuda_hook_info.func_prehook[CU_MEM_POOL_CREATE] =
        (void *)cuMemPoolCreate_prehook;
    cuda_hook_info.func_proxy[CU_MEM_POOL_CREATE] =
        (void *)cuMemPoolCreate_proxy;
    cuda_hook_info.func_posthook[CU_MEM_POOL_CREATE] =
        (void *)cuMemPoolCreate_posthook;
    cuda_hook_info.func_prehook[CU_MEM_POOL_DESTROY] =
        (void *)cuMemPoolDestroy_prehook;
    cuda_hook_info.func_proxy[CU_MEM_POOL_DESTROY] =
        (void *)cuMemPoolDestroy_proxy;
    cuda_hook_info.func_posthook[CU_MEM_POOL_DESTROY] =
        (void *)cuMemPoolDestroy_posthook;
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
    CU_ARRAY_3D_CREATE,
    ,
    cuArray3DCreate,
    (CUarray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray),
    pHandle, pAllocateArray)

CUDA_HOOK_GEN(
    CU_ARRAY_CREATE,
    ,
    cuArrayCreate,
    (CUarray *pHandle,
    const CUDA_ARRAY_DESCRIPTOR *pAllocateArray),
    pHandle, pAllocateArray)

CUDA_HOOK_GEN(
    CU_ARRAY_DESTROY,
    ,
    cuArrayDestroy,
    (CUarray hArray),
    hArray)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC,
    ,
    cuMemAlloc,
    (CUdeviceptr *dptr,
    size_t bytesize),
    dptr, bytesize)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_HOST,
    ,
    cuMemAllocHost,
    (void **pp,
    size_t bytesize),
    pp, bytesize)

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
    CU_MEM_FREE_HOST,
    ,
    cuMemFreeHost,
    (void *p),
    p)

CUDA_HOOK_GEN(
    CU_MEM_HOST_ALLOC,
    ,
    cuMemHostAlloc,
    (void **pp,
    size_t bytesize,
    unsigned int Flags),
    pp, bytesize, Flags)

CUDA_HOOK_GEN(
    CU_MEMCPY,
    ,
    cuMemcpy,
    (CUdeviceptr dst,
    CUdeviceptr src,
    size_t ByteCount),
    dst, src, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_2D,
    ,
    cuMemcpy2D,
    (const CUDA_MEMCPY2D *pCopy),
    pCopy)

CUDA_HOOK_GEN(
    CU_MEMCPY_2D_ASYNC,
    ,
    cuMemcpy2DAsync,
    (const CUDA_MEMCPY2D *pCopy,
    CUstream hStream),
    pCopy, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_2D_UNALIGNED,
    ,
    cuMemcpy2DUnaligned,
    (const CUDA_MEMCPY2D *pCopy),
    pCopy)

CUDA_HOOK_GEN(
    CU_MEMCPY_3D,
    ,
    cuMemcpy3D,
    (const CUDA_MEMCPY3D *pCopy),
    pCopy)

CUDA_HOOK_GEN(
    CU_MEMCPY_3D_ASYNC,
    ,
    cuMemcpy3DAsync,
    (const CUDA_MEMCPY3D *pCopy,
    CUstream hStream),
    pCopy, hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_3D_PEER,
    ,
    cuMemcpy3DPeer,
    (const CUDA_MEMCPY3D_PEER *pCopy),
    pCopy)

CUDA_HOOK_GEN(
    CU_MEMCPY_3D_PEER_ASYNC,
    ,
    cuMemcpy3DPeerAsync,
    (const CUDA_MEMCPY3D_PEER *pCopy,
    CUstream hStream),
    pCopy, hStream)

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
    CU_MEMCPY_ATO_A,
    ,
    cuMemcpyAtoA,
    (CUarray dstArray,
    size_t dstOffset,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount),
    dstArray, dstOffset, srcArray, srcOffset,
    ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_ATO_D,
    ,
    cuMemcpyAtoD,
    (CUdeviceptr dstDevice,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount),
    dstDevice, srcArray, srcOffset, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_ATO_H,
    ,
    cuMemcpyAtoH,
    (void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount),
    dstHost, srcArray, srcOffset, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_ATO_H_ASYNC,
    ,
    cuMemcpyAtoHAsync,
    (void *dstHost,
    CUarray srcArray,
    size_t srcOffset,
    size_t ByteCount,
    CUstream hStream),
    dstHost, srcArray, srcOffset, ByteCount,
    hStream)

CUDA_HOOK_GEN(
    CU_MEMCPY_DTO_A,
    ,
    cuMemcpyDtoA,
    (CUarray dstArray,
    size_t dstOffset,
    CUdeviceptr srcDevice,
    size_t ByteCount),
    dstArray, dstOffset, srcDevice, ByteCount)

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
    CU_MEMCPY_HTO_A,
    ,
    cuMemcpyHtoA,
    (CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount),
    dstArray, dstOffset, srcHost, ByteCount)

CUDA_HOOK_GEN(
    CU_MEMCPY_HTO_A_ASYNC,
    ,
    cuMemcpyHtoAAsync,
    (CUarray dstArray,
    size_t dstOffset,
    const void *srcHost,
    size_t ByteCount,
    CUstream hStream),
    dstArray, dstOffset, srcHost, ByteCount,
    hStream)

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
    CU_MIPMAPPED_ARRAY_CREATE,
    ,
    cuMipmappedArrayCreate,
    (CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels),
    pHandle, pMipmappedArrayDesc, numMipmapLevels)

CUDA_HOOK_GEN(
    CU_MIPMAPPED_ARRAY_DESTROY,
    ,
    cuMipmappedArrayDestroy,
    (CUmipmappedArray hMipmappedArray),
    hMipmappedArray)

CUDA_HOOK_GEN(
    CU_MEM_ADDRESS_FREE,
    ,
    cuMemAddressFree,
    (CUdeviceptr ptr,
    size_t size),
    ptr, size)

CUDA_HOOK_GEN(
    CU_MEM_ADDRESS_RESERVE,
    ,
    cuMemAddressReserve,
    (CUdeviceptr *ptr,
    size_t size,
    size_t alignment,
    CUdeviceptr addr,
    unsigned long long flags),
    ptr, size, alignment, addr,
    flags)

CUDA_HOOK_GEN(
    CU_MEM_CREATE,
    ,
    cuMemCreate,
    (CUmemGenericAllocationHandle *handle,
    size_t size,
    const CUmemAllocationProp *prop,
    unsigned long long flags),
    handle, size, prop, flags)

CUDA_HOOK_GEN(
    CU_MEM_RELEASE,
    ,
    cuMemRelease,
    (CUmemGenericAllocationHandle handle),
    handle)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_ASYNC,
    ,
    cuMemAllocAsync,
    (CUdeviceptr *dptr,
    size_t bytesize,
    CUstream hStream),
    dptr, bytesize, hStream)

CUDA_HOOK_GEN(
    CU_MEM_ALLOC_FROM_POOL_ASYNC,
    ,
    cuMemAllocFromPoolAsync,
    (CUdeviceptr *dptr,
    size_t bytesize,
    CUmemoryPool pool,
    CUstream hStream),
    dptr, bytesize, pool, hStream)

CUDA_HOOK_GEN(
    CU_MEM_FREE_ASYNC,
    ,
    cuMemFreeAsync,
    (CUdeviceptr dptr,
    CUstream hStream),
    dptr, hStream)

CUDA_HOOK_GEN(
    CU_MEM_POOL_CREATE,
    ,
    cuMemPoolCreate,
    (CUmemoryPool *pool,
    const CUmemPoolProps *poolProps),
    pool, poolProps)

CUDA_HOOK_GEN(
    CU_MEM_POOL_DESTROY,
    ,
    cuMemPoolDestroy,
    (CUmemoryPool pool),
    pool)

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

#endif /* _CUDA_HOOK_ENABLE */
