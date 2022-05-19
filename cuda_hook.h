#ifndef _CUDA_HOOK_H_
#define _CUDA_HOOK_H_

enum cudaHookSymbols {
    CU_GET_PROC_ADDRESS,
    CU_ARRAY_3D_CREATE,
    CU_ARRAY_CREATE,
    CU_ARRAY_DESTROY,
    CU_MEM_ALLOC,
    CU_MEM_ALLOC_HOST,
    CU_MEM_ALLOC_MANAGED,
    CU_MEM_ALLOC_PITCH,
    CU_MEM_FREE,
    CU_MEM_FREE_HOST,
    CU_MEM_HOST_ALLOC,
    CU_MEMCPY,
    CU_MEMCPY_2D,
    CU_MEMCPY_2D_ASYNC,
    CU_MEMCPY_2D_UNALIGNED,
    CU_MEMCPY_3D,
    CU_MEMCPY_3D_ASYNC,
    CU_MEMCPY_3D_PEER,
    CU_MEMCPY_3D_PEER_ASYNC,
    CU_MEMCPY_ASYNC,
    CU_MEMCPY_ATO_A,
    CU_MEMCPY_ATO_D,
    CU_MEMCPY_ATO_H,
    CU_MEMCPY_ATO_H_ASYNC,
    CU_MEMCPY_DTO_A,
    CU_MEMCPY_DTO_D,
    CU_MEMCPY_DTO_D_ASYNC,
    CU_MEMCPY_DTO_H,
    CU_MEMCPY_DTO_H_ASYNC,
    CU_MEMCPY_HTO_A,
    CU_MEMCPY_HTO_A_ASYNC,
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
    CU_MIPMAPPED_ARRAY_CREATE,
    CU_MIPMAPPED_ARRAY_DESTROY,
    CU_MEM_ADDRESS_FREE,
    CU_MEM_ADDRESS_RESERVE,
    CU_MEM_CREATE,
    CU_MEM_RELEASE,
    CU_MEM_ALLOC_ASYNC,
    CU_MEM_ALLOC_FROM_POOL_ASYNC,
    CU_MEM_FREE_ASYNC,
    CU_MEM_POOL_CREATE,
    CU_MEM_POOL_DESTROY,
    CU_LAUNCH_COOPERATIVE_KERNEL,
    CU_LAUNCH_HOST_FUNC,
    CU_LAUNCH_KERNEL,
    NUM_CUDA_HOOK_SYMBOLS = 61
};

struct cudaHookInfo {
    int hook_proxy_enable;
    void *func_prehook[NUM_CUDA_HOOK_SYMBOLS];
    void *func_proxy[NUM_CUDA_HOOK_SYMBOLS];    /* hook_proxy_enable = 1 */
    void *func_actual[NUM_CUDA_HOOK_SYMBOLS];   /* hook_proxy_enable = 0 */
    void *func_posthook[NUM_CUDA_HOOK_SYMBOLS];

    cudaHookInfo() {
        hook_proxy_enable = 0;

#ifdef _CUDA_HOOK_PROXY_ENABLE
        hook_proxy_enable = 1;
#endif /* _CUDA_HOOK_PROXY_ENABLE */
    }
};

#define CUDA_HOOK_GEN(hooksymbol, deprecated, funcname, params, ...)                    \
    deprecated CUresult CUDAAPI funcname params                                         \
    {                                                                                   \
        DEBUG("[%s] Enter func\n", __func__);                                           \
                                                                                        \
        typedef decltype(&funcname) funcType;                                           \
        CUresult result;                                                                \
        void *actualFunc;                                                               \
                                                                                        \
        pthread_once(&cuda_hook_init_done, cuda_hook_init);                             \
                                                                                        \
        /* prehook */                                                                   \
        if((actualFunc = cuda_hook_info.func_prehook[hooksymbol]))                      \
            DRIVER_API_CALL(((funcType)actualFunc)(__VA_ARGS__));                       \
                                                                                        \
        /* hook */                                                                      \
        if(cuda_hook_info.hook_proxy_enable && cuda_hook_info.func_proxy[hooksymbol])   \
            actualFunc = cuda_hook_info.func_proxy[hooksymbol];                         \
        else if(!(actualFunc = cuda_hook_info.func_actual[hooksymbol])) {               \
            actualFunc = actualDlsym(libcudaHandle, SYMBOL_STRING(funcname));           \
            cuda_hook_info.func_actual[hooksymbol] = actualFunc;                        \
        }                                                                               \
        result = ((funcType)actualFunc)(__VA_ARGS__);                                   \
                                                                                        \
        /* posthook */                                                                  \
        if((actualFunc = cuda_hook_info.func_posthook[hooksymbol]))                     \
            DRIVER_API_CALL(((funcType)actualFunc)(__VA_ARGS__));                       \
                                                                                        \
        DEBUG("[%s] Leave func\n", __func__);                                           \
                                                                                        \
        return result;                                                                  \
    }

#endif /* _CUDA_HOOK_H_ */
