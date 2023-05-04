#ifndef _CUDA_HOOK_HPP_
#define _CUDA_HOOK_HPP_

enum cuda_hook_symbols
{
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
    NUM_CUDA_HOOK_SYMBOLS = 32
};

struct cudaHookInfo
{
    int hook_effect_enable;
    void *func_prehook[NUM_CUDA_HOOK_SYMBOLS];   /* hook_effect_enable = 1 */
    void *func_proxy[NUM_CUDA_HOOK_SYMBOLS];     /* hook_effect_enable = 1 */
    void *func_actual[NUM_CUDA_HOOK_SYMBOLS];    /* hook_effect_enable = 0 */
    void *func_posthook[NUM_CUDA_HOOK_SYMBOLS];  /* hook_effect_enable = 1 */

    cudaHookInfo(void)
    {
        hook_effect_enable = 0;
#ifdef _CUDA_HOOK_EFFECT_ENABLE
        hook_effect_enable = 1;
#endif
    }
};

#define CUDA_HOOK_GEN(hooksymbol, deprecated, funcname, params, ...)                        \
    deprecated CUresult CUDAAPI funcname params                                             \
    {                                                                                       \
        hook_log.debug("Enter function: "s + string(__func__));                             \
                                                                                            \
        typedef decltype(&funcname) func_type;                                              \
        CUresult result;                                                                    \
        void *actual_func;                                                                  \
                                                                                            \
        pthread_once(&cuda_hook_init_done, cuda_hook_init);                                 \
                                                                                            \
        /* prehook */                                                                       \
        if(cuda_hook_info.hook_effect_enable && cuda_hook_info.func_prehook[hooksymbol]) {  \
            actual_func = cuda_hook_info.func_prehook[hooksymbol];                          \
            ((func_type)actual_func)(__VA_ARGS__);                                          \
        }                                                                                   \
                                                                                            \
        /* hook */                                                                          \
        if(cuda_hook_info.hook_effect_enable && cuda_hook_info.func_proxy[hooksymbol])      \
            actual_func = cuda_hook_info.func_proxy[hooksymbol];                            \
        else if(!(actual_func = cuda_hook_info.func_actual[hooksymbol])) {                  \
            actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(funcname));            \
            cuda_hook_info.func_actual[hooksymbol] = actual_func;                           \
        }                                                                                   \
        result = ((func_type)actual_func)(__VA_ARGS__);                                     \
                                                                                            \
        /* posthook */                                                                      \
        if(cuda_hook_info.hook_effect_enable && cuda_hook_info.func_posthook[hooksymbol]) { \
            actual_func = cuda_hook_info.func_posthook[hooksymbol];                         \
            ((func_type)actual_func)(__VA_ARGS__);                                          \
        }                                                                                   \
                                                                                            \
        hook_log.debug("Leave function: "s + string(__func__));                             \
                                                                                            \
        return result;                                                                      \
    }

#endif /* _CUDA_HOOK_HPP_ */
