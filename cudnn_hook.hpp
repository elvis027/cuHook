#ifndef _CUDNN_HOOK_HPP_
#define _CUDNN_HOOK_HPP_

enum cudnn_hook_symbols
{
    CUDNN_CREATE,
    CUDNN_DESTROY,
    CUDNN_QUERY_RUNTIME_ERROR,
    CUDNN_GET_PROPERTY,
    CUDNN_SET_STREAM,
    CUDNN_GET_STREAM,
    CUDNN_CREATE_TENSOR_DESCRIPTOR,
    CUDNN_SET_TENSOR_4D_DESCRIPTOR,
    CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX,
    CUDNN_GET_TENSOR_4D_DESCRIPTOR,
    CUDNN_SET_TENSOR_ND_DESCRIPTOR,
    CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX,
    CUDNN_GET_TENSOR_ND_DESCRIPTOR,
    CUDNN_GET_TENSOR_SIZE_IN_BYTES,
    CUDNN_DESTROY_TENSOR_DESCRIPTOR,
    CUDNN_INIT_TRANSFORM_DEST,
    CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR,
    CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR,
    CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR,
    CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR,
    CUDNN_TRANSFORM_TENSOR,
    CUDNN_TRANSFORM_TENSOR_EX,
    CUDNN_ADD_TENSOR,
    CUDNN_CREATE_OP_TENSOR_DESCRIPTOR,
    CUDNN_SET_OP_TENSOR_DESCRIPTOR,
    CUDNN_GET_OP_TENSOR_DESCRIPTOR,
    CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR,
    CUDNN_OP_TENSOR,
    CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR,
    CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR,
    CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR,
    CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR,
    CUDNN_GET_REDUCTION_INDICES_SIZE,
    CUDNN_GET_REDUCTION_WORKSPACE_SIZE,
    CUDNN_REDUCE_TENSOR,
    CUDNN_SET_TENSOR,
    CUDNN_SCALE_TENSOR,
    CUDNN_CREATE_FILTER_DESCRIPTOR,
    CUDNN_SET_FILTER_4D_DESCRIPTOR,
    CUDNN_GET_FILTER_4D_DESCRIPTOR,
    CUDNN_SET_FILTER_ND_DESCRIPTOR,
    CUDNN_GET_FILTER_ND_DESCRIPTOR,
    CUDNN_GET_FILTER_SIZE_IN_BYTES,
    CUDNN_TRANSFORM_FILTER,
    CUDNN_DESTROY_FILTER_DESCRIPTOR,
    CUDNN_SOFTMAX_FORWARD,
    CUDNN_CREATE_POOLING_DESCRIPTOR,
    CUDNN_SET_POOLING_2D_DESCRIPTOR,
    CUDNN_GET_POOLING_2D_DESCRIPTOR,
    CUDNN_SET_POOLING_ND_DESCRIPTOR,
    CUDNN_GET_POOLING_ND_DESCRIPTOR,
    CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM,
    CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM,
    CUDNN_DESTROY_POOLING_DESCRIPTOR,
    CUDNN_POOLING_FORWARD,
    CUDNN_CREATE_ACTIVATION_DESCRIPTOR,
    CUDNN_SET_ACTIVATION_DESCRIPTOR,
    CUDNN_GET_ACTIVATION_DESCRIPTOR,
    CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA,
    CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA,
    CUDNN_DESTROY_ACTIVATION_DESCRIPTOR,
    CUDNN_ACTIVATION_FORWARD,
    CUDNN_CREATE_LRN_DESCRIPTOR,
    CUDNN_SET_LRN_DESCRIPTOR,
    CUDNN_GET_LRN_DESCRIPTOR,
    CUDNN_DESTROY_LRN_DESCRIPTOR,
    CUDNN_LRN_CROSS_CHANNEL_FORWARD,
    CUDNN_DIVISIVE_NORMALIZATION_FORWARD,
    CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR,
    CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE,
    CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR,
    CUDNN_NORMALIZATION_FORWARD_INFERENCE,
    CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR,
    CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR,
    CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR,
    CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD,
    CUDNN_SPATIAL_TF_SAMPLER_FORWARD,
    CUDNN_CREATE_DROPOUT_DESCRIPTOR,
    CUDNN_DESTROY_DROPOUT_DESCRIPTOR,
    CUDNN_DROPOUT_GET_STATES_SIZE,
    CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE,
    CUDNN_SET_DROPOUT_DESCRIPTOR,
    CUDNN_RESTORE_DROPOUT_DESCRIPTOR,
    CUDNN_GET_DROPOUT_DESCRIPTOR,
    CUDNN_DROPOUT_FORWARD,
    CUDNN_CREATE_ALGORITHM_DESCRIPTOR,
    CUDNN_SET_ALGORITHM_DESCRIPTOR,
    CUDNN_GET_ALGORITHM_DESCRIPTOR,
    CUDNN_COPY_ALGORITHM_DESCRIPTOR,
    CUDNN_DESTROY_ALGORITHM_DESCRIPTOR,
    CUDNN_CREATE_ALGORITHM_PERFORMANCE,
    CUDNN_SET_ALGORITHM_PERFORMANCE,
    CUDNN_GET_ALGORITHM_PERFORMANCE,
    CUDNN_DESTROY_ALGORITHM_PERFORMANCE,
    CUDNN_GET_ALGORITHM_SPACE_SIZE,
    CUDNN_SAVE_ALGORITHM,
    CUDNN_RESTORE_ALGORITHM,
    CUDNN_SET_CALLBACK,
    CUDNN_GET_CALLBACK,
    CUDNN_OPS_INFER_VERSION_CHECK,
    CUDNN_SOFTMAX_BACKWARD,
    CUDNN_POOLING_BACKWARD,
    CUDNN_ACTIVATION_BACKWARD,
    CUDNN_LRN_CROSS_CHANNEL_BACKWARD,
    CUDNN_DIVISIVE_NORMALIZATION_BACKWARD,
    CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE,
    CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE,
    CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE,
    CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING,
    CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX,
    CUDNN_BATCH_NORMALIZATION_BACKWARD,
    CUDNN_BATCH_NORMALIZATION_BACKWARD_EX,
    CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE,
    CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE,
    CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE,
    CUDNN_NORMALIZATION_FORWARD_TRAINING,
    CUDNN_NORMALIZATION_BACKWARD,
    CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD,
    CUDNN_SPATIAL_TF_SAMPLER_BACKWARD,
    CUDNN_DROPOUT_BACKWARD,
    CUDNN_OPS_TRAIN_VERSION_CHECK,
    CUDNN_CREATE_RNN_DESCRIPTOR,
    CUDNN_DESTROY_RNN_DESCRIPTOR,
    CUDNN_SET_RNN_DESCRIPTOR_V8,
    CUDNN_GET_RNN_DESCRIPTOR_V8,
    CUDNN_SET_RNN_DESCRIPTOR_V6,
    CUDNN_GET_RNN_DESCRIPTOR_V6,
    CUDNN_SET_RNN_MATRIX_MATH_TYPE,
    CUDNN_GET_RNN_MATRIX_MATH_TYPE,
    CUDNN_SET_RNN_BIAS_MODE,
    CUDNN_GET_RNN_BIAS_MODE,
    CUDNN_RNN_SET_CLIP_V8,
    CUDNN_RNN_GET_CLIP_V8,
    CUDNN_RNN_SET_CLIP,
    CUDNN_RNN_GET_CLIP,
    CUDNN_SET_RNN_PROJECTION_LAYERS,
    CUDNN_GET_RNN_PROJECTION_LAYERS,
    CUDNN_CREATE_PERSISTENT_RNN_PLAN,
    CUDNN_DESTROY_PERSISTENT_RNN_PLAN,
    CUDNN_SET_PERSISTENT_RNN_PLAN,
    CUDNN_BUILD_RNN_DYNAMIC,
    CUDNN_GET_RNN_WORKSPACE_SIZE,
    CUDNN_GET_RNN_TRAINING_RESERVE_SIZE,
    CUDNN_GET_RNN_TEMP_SPACE_SIZES,
    CUDNN_GET_RNN_PARAMS_SIZE,
    CUDNN_GET_RNN_WEIGHT_SPACE_SIZE,
    CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS,
    CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS,
    CUDNN_GET_RNN_WEIGHT_PARAMS,
    CUDNN_RNN_FORWARD_INFERENCE,
    CUDNN_SET_RNN_PADDING_MODE,
    CUDNN_GET_RNN_PADDING_MODE,
    CUDNN_CREATE_RNN_DATA_DESCRIPTOR,
    CUDNN_DESTROY_RNN_DATA_DESCRIPTOR,
    CUDNN_SET_RNN_DATA_DESCRIPTOR,
    CUDNN_GET_RNN_DATA_DESCRIPTOR,
    CUDNN_RNN_FORWARD_INFERENCE_EX,
    CUDNN_RNN_FORWARD,
    CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR,
    CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT,
    CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX,
    CUDNN_CREATE_SEQ_DATA_DESCRIPTOR,
    CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR,
    CUDNN_SET_SEQ_DATA_DESCRIPTOR,
    CUDNN_GET_SEQ_DATA_DESCRIPTOR,
    CUDNN_CREATE_ATTN_DESCRIPTOR,
    CUDNN_DESTROY_ATTN_DESCRIPTOR,
    CUDNN_SET_ATTN_DESCRIPTOR,
    CUDNN_GET_ATTN_DESCRIPTOR,
    CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS,
    CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS,
    CUDNN_MULTI_HEAD_ATTN_FORWARD,
    CUDNN_ADV_INFER_VERSION_CHECK,
    CUDNN_RNN_FORWARD_TRAINING,
    CUDNN_RNN_BACKWARD_DATA,
    CUDNN_RNN_BACKWARD_DATA_V8,
    CUDNN_RNN_BACKWARD_WEIGHTS,
    CUDNN_RNN_BACKWARD_WEIGHTS_V8,
    CUDNN_RNN_FORWARD_TRAINING_EX,
    CUDNN_RNN_BACKWARD_DATA_EX,
    CUDNN_RNN_BACKWARD_WEIGHTS_EX,
    CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT,
    CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX,
    CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT,
    CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX,
    CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT,
    CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX,
    CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA,
    CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS,
    CUDNN_CREATE_CTC_LOSS_DESCRIPTOR,
    CUDNN_SET_CTC_LOSS_DESCRIPTOR,
    CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX,
    CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8,
    CUDNN_GET_CTC_LOSS_DESCRIPTOR,
    CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX,
    CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8,
    CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR,
    CUDNN_CTC_LOSS,
    CUDNN_CTC_LOSS_V8,
    CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE,
    CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8,
    CUDNN_ADV_TRAIN_VERSION_CHECK,
    CUDNN_CREATE_CONVOLUTION_DESCRIPTOR,
    CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR,
    CUDNN_SET_CONVOLUTION_MATH_TYPE,
    CUDNN_GET_CONVOLUTION_MATH_TYPE,
    CUDNN_SET_CONVOLUTION_GROUP_COUNT,
    CUDNN_GET_CONVOLUTION_GROUP_COUNT,
    CUDNN_SET_CONVOLUTION_REORDER_TYPE,
    CUDNN_GET_CONVOLUTION_REORDER_TYPE,
    CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR,
    CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR,
    CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR,
    CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR,
    CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM,
    CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM,
    CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT,
    CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7,
    CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM,
    CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX,
    CUDNN_IM_2_COL,
    CUDNN_REORDER_FILTER_AND_BIAS,
    CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE,
    CUDNN_CONVOLUTION_FORWARD,
    CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD,
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT,
    CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM,
    CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX,
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7,
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE,
    CUDNN_CONVOLUTION_BACKWARD_DATA,
    CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS,
    CUDNN_CNN_INFER_VERSION_CHECK,
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT,
    CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM,
    CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX,
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7,
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE,
    CUDNN_CONVOLUTION_BACKWARD_FILTER,
    CUDNN_CONVOLUTION_BACKWARD_BIAS,
    CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK,
    CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK,
    CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE,
    CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE,
    CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK,
    CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK,
    CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE,
    CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE,
    CUDNN_CREATE_FUSED_OPS_PLAN,
    CUDNN_DESTROY_FUSED_OPS_PLAN,
    CUDNN_MAKE_FUSED_OPS_PLAN,
    CUDNN_FUSED_OPS_EXECUTE,
    CUDNN_CNN_TRAIN_VERSION_CHECK,
    CUDNN_BACKEND_CREATE_DESCRIPTOR,
    CUDNN_BACKEND_DESTROY_DESCRIPTOR,
    CUDNN_BACKEND_INITIALIZE,
    CUDNN_BACKEND_FINALIZE,
    CUDNN_BACKEND_SET_ATTRIBUTE,
    CUDNN_BACKEND_GET_ATTRIBUTE,
    CUDNN_BACKEND_EXECUTE,
    NUM_CUDNN_HOOK_SYMBOLS = 260
};

struct cudnnHookInfo
{
    int hook_effect_enable;
    void *func_prehook[NUM_CUDNN_HOOK_SYMBOLS];   /* hook_effect_enable = 1 */
    void *func_proxy[NUM_CUDNN_HOOK_SYMBOLS];     /* hook_effect_enable = 1 */
    void *func_actual[NUM_CUDNN_HOOK_SYMBOLS];    /* hook_effect_enable = 0 */
    void *func_posthook[NUM_CUDNN_HOOK_SYMBOLS];  /* hook_effect_enable = 1 */

    cudnnHookInfo()
    {
        hook_effect_enable = 0;
#ifdef _CUDNN_HOOK_EFFECT_ENABLE
        hook_effect_enable = 1;
#endif
    }
};

#define CUDNN_HANDLE_HOOK_GEN(hooksymbol, deprecated, funcname, params, handle, ...)            \
    deprecated cudnnStatus_t CUDNNWINAPI funcname params                                        \
    {                                                                                           \
        hook_log.debug("Enter function: "s + string(__func__));                                 \
                                                                                                \
        typedef decltype(&funcname) func_type;                                                  \
        cudnnStatus_t result;                                                                   \
        void *actual_func;                                                                      \
                                                                                                \
        pthread_once(&cudnn_hook_init_done, cudnn_hook_init);                                   \
                                                                                                \
        /* prehook */                                                                           \
        if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_prehook[hooksymbol]) {    \
            actual_func = cudnn_hook_info.func_prehook[hooksymbol];                             \
            ((func_type)actual_func)(handle, __VA_ARGS__);                                      \
        }                                                                                       \
                                                                                                \
        /* hook */                                                                              \
        if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_proxy[hooksymbol])        \
            actual_func = cudnn_hook_info.func_proxy[hooksymbol];                               \
        else if(!(actual_func = cudnn_hook_info.func_actual[hooksymbol])) {                     \
            actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(funcname));               \
            cudnn_hook_info.func_actual[hooksymbol] = actual_func;                              \
        }                                                                                       \
        result = ((func_type)actual_func)(handle, __VA_ARGS__);                                 \
                                                                                                \
        /* posthook */                                                                          \
        if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_posthook[hooksymbol]) {   \
            actual_func = cudnn_hook_info.func_posthook[hooksymbol];                            \
            ((func_type)actual_func)(handle, __VA_ARGS__);                                      \
        }                                                                                       \
                                                                                                \
        hook_log.debug("Leave function: "s + string(__func__));                                 \
                                                                                                \
        return result;                                                                          \
    }

#define CUDNN_HOOK_GEN(hooksymbol, deprecated, funcname, params, ...)                           \
    deprecated cudnnStatus_t CUDNNWINAPI funcname params                                        \
    {                                                                                           \
        hook_log.debug("Enter function: "s + string(__func__));                                 \
                                                                                                \
        typedef decltype(&funcname) func_type;                                                  \
        cudnnStatus_t result;                                                                   \
        void *actual_func;                                                                      \
                                                                                                \
        pthread_once(&cudnn_hook_init_done, cudnn_hook_init);                                   \
                                                                                                \
        /* prehook */                                                                           \
        if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_prehook[hooksymbol]) {    \
            actual_func = cudnn_hook_info.func_prehook[hooksymbol];                             \
            ((func_type)actual_func)(__VA_ARGS__);                                              \
        }                                                                                       \
                                                                                                \
        /* hook */                                                                              \
        if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_proxy[hooksymbol])        \
            actual_func = cudnn_hook_info.func_proxy[hooksymbol];                               \
        else if(!(actual_func = cudnn_hook_info.func_actual[hooksymbol])) {                     \
            actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(funcname));               \
            cudnn_hook_info.func_actual[hooksymbol] = actual_func;                              \
        }                                                                                       \
        result = ((func_type)actual_func)(__VA_ARGS__);                                         \
                                                                                                \
        /* posthook */                                                                          \
        if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_posthook[hooksymbol]) {   \
            actual_func = cudnn_hook_info.func_posthook[hooksymbol];                            \
            ((func_type)actual_func)(__VA_ARGS__);                                              \
        }                                                                                       \
                                                                                                \
        hook_log.debug("Leave function: "s + string(__func__));                                 \
                                                                                                \
        return result;                                                                          \
    }

#endif /* _CUDNN_HOOK_HPP_ */
