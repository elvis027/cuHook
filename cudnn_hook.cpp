#include <iostream>
#include <cstring>
#include <pthread.h>

#include <cudnn.h>

#include "cudnn_hook.hpp"
#include "hook.hpp"
#include "logging.hpp"

using namespace std::string_literals;
using std::string;
using std::to_string;

typedef unsigned long long ull;

static struct cudnnHookInfo cudnn_hook_info;
static pthread_once_t cudnn_hook_init_done = PTHREAD_ONCE_INIT;

/* prehook, proxy, posthook functions start */
cudnnStatus_t cudnnCreate_prehook(
    cudnnHandle_t *handle)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreate_proxy(
    cudnnHandle_t *handle)
{
    typedef decltype(&cudnnCreate) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreate));
        cudnn_hook_info.func_actual[CUDNN_CREATE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle);
}

cudnnStatus_t cudnnCreate_posthook(
    cudnnHandle_t *handle)
{
    trace_dump.dump("cudnnCreate");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy_prehook(
    cudnnHandle_t handle)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy_proxy(
    cudnnHandle_t handle)
{
    typedef decltype(&cudnnDestroy) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroy));
        cudnn_hook_info.func_actual[CUDNN_DESTROY] = actual_func;
    }
    return ((func_type)actual_func)(
        handle);
}

cudnnStatus_t cudnnDestroy_posthook(
    cudnnHandle_t handle)
{
    trace_dump.dump("cudnnDestroy");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnQueryRuntimeError_prehook(
    cudnnHandle_t handle, cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnQueryRuntimeError_proxy(
    cudnnHandle_t handle, cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag)
{
    typedef decltype(&cudnnQueryRuntimeError) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_QUERY_RUNTIME_ERROR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnQueryRuntimeError));
        cudnn_hook_info.func_actual[CUDNN_QUERY_RUNTIME_ERROR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rstatus, mode, tag);
}

cudnnStatus_t cudnnQueryRuntimeError_posthook(
    cudnnHandle_t handle, cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag)
{
    trace_dump.dump("cudnnQueryRuntimeError");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream_prehook(
    cudnnHandle_t handle, cudaStream_t streamId)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream_proxy(
    cudnnHandle_t handle, cudaStream_t streamId)
{
    typedef decltype(&cudnnSetStream) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_STREAM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetStream));
        cudnn_hook_info.func_actual[CUDNN_SET_STREAM] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, streamId);
}

cudnnStatus_t cudnnSetStream_posthook(
    cudnnHandle_t handle, cudaStream_t streamId)
{
    trace_dump.dump("cudnnSetStream");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream_prehook(
    cudnnHandle_t handle, cudaStream_t *streamId)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream_proxy(
    cudnnHandle_t handle, cudaStream_t *streamId)
{
    typedef decltype(&cudnnGetStream) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_STREAM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetStream));
        cudnn_hook_info.func_actual[CUDNN_GET_STREAM] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, streamId);
}

cudnnStatus_t cudnnGetStream_posthook(
    cudnnHandle_t handle, cudaStream_t *streamId)
{
    trace_dump.dump("cudnnGetStream");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensor_prehook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensor_proxy(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    typedef decltype(&cudnnTransformTensor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_TRANSFORM_TENSOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnTransformTensor));
        cudnn_hook_info.func_actual[CUDNN_TRANSFORM_TENSOR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, alpha, xDesc, x,
        beta, yDesc, y);
}

cudnnStatus_t cudnnTransformTensor_posthook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    trace_dump.dump("cudnnTransformTensor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensorEx_prehook(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnTensorDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnTensorDescriptor_t destDesc, void *destData)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensorEx_proxy(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnTensorDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnTensorDescriptor_t destDesc, void *destData)
{
    typedef decltype(&cudnnTransformTensorEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_TRANSFORM_TENSOR_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnTransformTensorEx));
        cudnn_hook_info.func_actual[CUDNN_TRANSFORM_TENSOR_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, transDesc, alpha, srcDesc,
        srcData, beta, destDesc, destData);
}

cudnnStatus_t cudnnTransformTensorEx_posthook(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnTensorDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnTensorDescriptor_t destDesc, void *destData)
{
    trace_dump.dump("cudnnTransformTensorEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAddTensor_prehook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAddTensor_proxy(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C)
{
    typedef decltype(&cudnnAddTensor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_ADD_TENSOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnAddTensor));
        cudnn_hook_info.func_actual[CUDNN_ADD_TENSOR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, alpha, aDesc, A,
        beta, cDesc, C);
}

cudnnStatus_t cudnnAddTensor_posthook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C)
{
    trace_dump.dump("cudnnAddTensor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpTensor_prehook(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *alpha2,
    const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpTensor_proxy(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *alpha2,
    const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C)
{
    typedef decltype(&cudnnOpTensor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_OP_TENSOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnOpTensor));
        cudnn_hook_info.func_actual[CUDNN_OP_TENSOR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, opTensorDesc, alpha1, aDesc,
        A, alpha2, bDesc, B,
        beta, cDesc, C);
}

cudnnStatus_t cudnnOpTensor_posthook(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *alpha2,
    const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C)
{
    trace_dump.dump("cudnnOpTensor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionIndicesSize_prehook(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionIndicesSize_proxy(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetReductionIndicesSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_REDUCTION_INDICES_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetReductionIndicesSize));
        cudnn_hook_info.func_actual[CUDNN_GET_REDUCTION_INDICES_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, reduceTensorDesc, aDesc, cDesc,
        sizeInBytes);
}

cudnnStatus_t cudnnGetReductionIndicesSize_posthook(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetReductionIndicesSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize_prehook(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize_proxy(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetReductionWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_REDUCTION_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetReductionWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_REDUCTION_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, reduceTensorDesc, aDesc, cDesc,
        sizeInBytes);
}

cudnnStatus_t cudnnGetReductionWorkspaceSize_posthook(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetReductionWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor_prehook(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes,
    void *workspace, size_t workspaceSizeInBytes,
    const void *alpha, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor_proxy(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes,
    void *workspace, size_t workspaceSizeInBytes,
    const void *alpha, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C)
{
    typedef decltype(&cudnnReduceTensor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_REDUCE_TENSOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnReduceTensor));
        cudnn_hook_info.func_actual[CUDNN_REDUCE_TENSOR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, reduceTensorDesc, indices, indicesSizeInBytes,
        workspace, workspaceSizeInBytes, alpha, aDesc,
        A, beta, cDesc, C);
}

cudnnStatus_t cudnnReduceTensor_posthook(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes,
    void *workspace, size_t workspaceSizeInBytes,
    const void *alpha, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C)
{
    trace_dump.dump("cudnnReduceTensor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *valuePtr)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *valuePtr)
{
    typedef decltype(&cudnnSetTensor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_TENSOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetTensor));
        cudnn_hook_info.func_actual[CUDNN_SET_TENSOR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, yDesc, y, valuePtr);
}

cudnnStatus_t cudnnSetTensor_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *valuePtr)
{
    trace_dump.dump("cudnnSetTensor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnScaleTensor_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *alpha)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnScaleTensor_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *alpha)
{
    typedef decltype(&cudnnScaleTensor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SCALE_TENSOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnScaleTensor));
        cudnn_hook_info.func_actual[CUDNN_SCALE_TENSOR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, yDesc, y, alpha);
}

cudnnStatus_t cudnnScaleTensor_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *alpha)
{
    trace_dump.dump("cudnnScaleTensor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformFilter_prehook(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnFilterDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnFilterDescriptor_t destDesc, void *destData)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformFilter_proxy(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnFilterDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnFilterDescriptor_t destDesc, void *destData)
{
    typedef decltype(&cudnnTransformFilter) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_TRANSFORM_FILTER])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnTransformFilter));
        cudnn_hook_info.func_actual[CUDNN_TRANSFORM_FILTER] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, transDesc, alpha, srcDesc,
        srcData, beta, destDesc, destData);
}

cudnnStatus_t cudnnTransformFilter_posthook(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnFilterDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnFilterDescriptor_t destDesc, void *destData)
{
    trace_dump.dump("cudnnTransformFilter");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxForward_prehook(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxForward_proxy(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    typedef decltype(&cudnnSoftmaxForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SOFTMAX_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSoftmaxForward));
        cudnn_hook_info.func_actual[CUDNN_SOFTMAX_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, algo, mode, alpha,
        xDesc, x, beta, yDesc,
        y);
}

cudnnStatus_t cudnnSoftmaxForward_posthook(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    trace_dump.dump("cudnnSoftmaxForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward_prehook(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward_proxy(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    typedef decltype(&cudnnPoolingForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_POOLING_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnPoolingForward));
        cudnn_hook_info.func_actual[CUDNN_POOLING_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, poolingDesc, alpha, xDesc,
        x, beta, yDesc, y);
}

cudnnStatus_t cudnnPoolingForward_posthook(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    trace_dump.dump("cudnnPoolingForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationForward_prehook(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationForward_proxy(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    typedef decltype(&cudnnActivationForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_ACTIVATION_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnActivationForward));
        cudnn_hook_info.func_actual[CUDNN_ACTIVATION_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, activationDesc, alpha, xDesc,
        x, beta, yDesc, y);
}

cudnnStatus_t cudnnActivationForward_posthook(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    trace_dump.dump("cudnnActivationForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelForward_prehook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelForward_proxy(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    typedef decltype(&cudnnLRNCrossChannelForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_LRN_CROSS_CHANNEL_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnLRNCrossChannelForward));
        cudnn_hook_info.func_actual[CUDNN_LRN_CROSS_CHANNEL_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, normDesc, lrnMode, alpha,
        xDesc, x, beta, yDesc,
        y);
}

cudnnStatus_t cudnnLRNCrossChannelForward_posthook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    trace_dump.dump("cudnnLRNCrossChannelForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationForward_prehook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, void *temp,
    void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationForward_proxy(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, void *temp,
    void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    typedef decltype(&cudnnDivisiveNormalizationForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DIVISIVE_NORMALIZATION_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDivisiveNormalizationForward));
        cudnn_hook_info.func_actual[CUDNN_DIVISIVE_NORMALIZATION_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, normDesc, mode, alpha,
        xDesc, x, means, temp,
        temp2, beta, yDesc, y);
}

cudnnStatus_t cudnnDivisiveNormalizationForward_posthook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, void *temp,
    void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    trace_dump.dump("cudnnDivisiveNormalizationForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon)
{
    typedef decltype(&cudnnBatchNormalizationForwardInference) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBatchNormalizationForwardInference));
        cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, alpha, beta,
        xDesc, x, yDesc, y,
        bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
        estimatedVariance, epsilon);
}

cudnnStatus_t cudnnBatchNormalizationForwardInference_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon)
{
    trace_dump.dump("cudnnBatchNormalizationForwardInference");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardInference_prehook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean, const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc,
    void *y, double epsilon,
    int groupCnt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardInference_proxy(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean, const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc,
    void *y, double epsilon,
    int groupCnt)
{
    typedef decltype(&cudnnNormalizationForwardInference) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_NORMALIZATION_FORWARD_INFERENCE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnNormalizationForwardInference));
        cudnn_hook_info.func_actual[CUDNN_NORMALIZATION_FORWARD_INFERENCE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, normOps, algo,
        alpha, beta, xDesc, x,
        normScaleBiasDesc, normScale, normBias, normMeanVarDesc,
        estimatedMean, estimatedVariance, zDesc, z,
        activationDesc, yDesc, y, epsilon,
        groupCnt);
}

cudnnStatus_t cudnnNormalizationForwardInference_posthook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean, const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc,
    void *y, double epsilon,
    int groupCnt)
{
    trace_dump.dump("cudnnNormalizationForwardInference");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward_prehook(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward_proxy(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid)
{
    typedef decltype(&cudnnSpatialTfGridGeneratorForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSpatialTfGridGeneratorForward));
        cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, stDesc, theta, grid);
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward_posthook(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid)
{
    trace_dump.dump("cudnnSpatialTfGridGeneratorForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerForward_prehook(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *grid,
    const void *beta, cudnnTensorDescriptor_t yDesc,
    void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerForward_proxy(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *grid,
    const void *beta, cudnnTensorDescriptor_t yDesc,
    void *y)
{
    typedef decltype(&cudnnSpatialTfSamplerForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_SAMPLER_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSpatialTfSamplerForward));
        cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_SAMPLER_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, stDesc, alpha, xDesc,
        x, grid, beta, yDesc,
        y);
}

cudnnStatus_t cudnnSpatialTfSamplerForward_posthook(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *grid,
    const void *beta, cudnnTensorDescriptor_t yDesc,
    void *y)
{
    trace_dump.dump("cudnnSpatialTfSamplerForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetStatesSize_prehook(
    cudnnHandle_t handle, size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetStatesSize_proxy(
    cudnnHandle_t handle, size_t *sizeInBytes)
{
    typedef decltype(&cudnnDropoutGetStatesSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DROPOUT_GET_STATES_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDropoutGetStatesSize));
        cudnn_hook_info.func_actual[CUDNN_DROPOUT_GET_STATES_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, sizeInBytes);
}

cudnnStatus_t cudnnDropoutGetStatesSize_posthook(
    cudnnHandle_t handle, size_t *sizeInBytes)
{
    trace_dump.dump("cudnnDropoutGetStatesSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutForward_prehook(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc, const void *x,
    const cudnnTensorDescriptor_t ydesc, void *y,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutForward_proxy(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc, const void *x,
    const cudnnTensorDescriptor_t ydesc, void *y,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnDropoutForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DROPOUT_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDropoutForward));
        cudnn_hook_info.func_actual[CUDNN_DROPOUT_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, dropoutDesc, xdesc, x,
        ydesc, y, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnDropoutForward_posthook(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc, const void *x,
    const cudnnTensorDescriptor_t ydesc, void *y,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnDropoutForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize_prehook(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize_proxy(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes)
{
    typedef decltype(&cudnnGetAlgorithmSpaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_ALGORITHM_SPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetAlgorithmSpaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_ALGORITHM_SPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, algoDesc, algoSpaceSizeInBytes);
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize_posthook(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes)
{
    trace_dump.dump("cudnnGetAlgorithmSpaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSaveAlgorithm_prehook(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace, size_t algoSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSaveAlgorithm_proxy(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace, size_t algoSpaceSizeInBytes)
{
    typedef decltype(&cudnnSaveAlgorithm) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SAVE_ALGORITHM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSaveAlgorithm));
        cudnn_hook_info.func_actual[CUDNN_SAVE_ALGORITHM] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
}

cudnnStatus_t cudnnSaveAlgorithm_posthook(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace, size_t algoSpaceSizeInBytes)
{
    trace_dump.dump("cudnnSaveAlgorithm");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreAlgorithm_prehook(
    cudnnHandle_t handle, void *algoSpace,
    size_t algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t algoDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreAlgorithm_proxy(
    cudnnHandle_t handle, void *algoSpace,
    size_t algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t algoDesc)
{
    typedef decltype(&cudnnRestoreAlgorithm) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RESTORE_ALGORITHM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRestoreAlgorithm));
        cudnn_hook_info.func_actual[CUDNN_RESTORE_ALGORITHM] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
}

cudnnStatus_t cudnnRestoreAlgorithm_posthook(
    cudnnHandle_t handle, void *algoSpace,
    size_t algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t algoDesc)
{
    trace_dump.dump("cudnnRestoreAlgorithm");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxBackward_prehook(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxBackward_proxy(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    typedef decltype(&cudnnSoftmaxBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SOFTMAX_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSoftmaxBackward));
        cudnn_hook_info.func_actual[CUDNN_SOFTMAX_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, algo, mode, alpha,
        yDesc, y, dyDesc, dy,
        beta, dxDesc, dx);
}

cudnnStatus_t cudnnSoftmaxBackward_posthook(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    trace_dump.dump("cudnnSoftmaxBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward_prehook(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward_proxy(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx)
{
    typedef decltype(&cudnnPoolingBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_POOLING_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnPoolingBackward));
        cudnn_hook_info.func_actual[CUDNN_POOLING_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, poolingDesc, alpha, yDesc,
        y, dyDesc, dy, xDesc,
        x, beta, dxDesc, dx);
}

cudnnStatus_t cudnnPoolingBackward_posthook(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx)
{
    trace_dump.dump("cudnnPoolingBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward_prehook(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward_proxy(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx)
{
    typedef decltype(&cudnnActivationBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_ACTIVATION_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnActivationBackward));
        cudnn_hook_info.func_actual[CUDNN_ACTIVATION_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, activationDesc, alpha, yDesc,
        y, dyDesc, dy, xDesc,
        x, beta, dxDesc, dx);
}

cudnnStatus_t cudnnActivationBackward_posthook(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx)
{
    trace_dump.dump("cudnnActivationBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelBackward_prehook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelBackward_proxy(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    typedef decltype(&cudnnLRNCrossChannelBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_LRN_CROSS_CHANNEL_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnLRNCrossChannelBackward));
        cudnn_hook_info.func_actual[CUDNN_LRN_CROSS_CHANNEL_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, normDesc, lrnMode, alpha,
        yDesc, y, dyDesc, dy,
        xDesc, x, beta, dxDesc,
        dx);
}

cudnnStatus_t cudnnLRNCrossChannelBackward_posthook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    trace_dump.dump("cudnnLRNCrossChannelBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward_prehook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, const void *dy,
    void *temp, void *temp2,
    const void *beta, const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx, void *dMeans)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward_proxy(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, const void *dy,
    void *temp, void *temp2,
    const void *beta, const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx, void *dMeans)
{
    typedef decltype(&cudnnDivisiveNormalizationBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDivisiveNormalizationBackward));
        cudnn_hook_info.func_actual[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, normDesc, mode, alpha,
        xDesc, x, means, dy,
        temp, temp2, beta, dXdMeansDesc,
        dx, dMeans);
}

cudnnStatus_t cudnnDivisiveNormalizationBackward_posthook(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, const void *dy,
    void *temp, void *temp2,
    const void *beta, const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx, void *dMeans)
{
    trace_dump.dump("cudnnDivisiveNormalizationBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, bnOps, xDesc,
        zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc,
        sizeInBytes);
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc, const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc, const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetBatchNormalizationBackwardExWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetBatchNormalizationBackwardExWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, bnOps, xDesc,
        yDesc, dyDesc, dzDesc, dxDesc,
        dBnScaleBiasDesc, activationDesc, sizeInBytes);
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc, const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetBatchNormalizationBackwardExWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetBatchNormalizationTrainingExReserveSpaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetBatchNormalizationTrainingExReserveSpaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, bnOps, activationDesc,
        xDesc, sizeInBytes);
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance)
{
    typedef decltype(&cudnnBatchNormalizationForwardTraining) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBatchNormalizationForwardTraining));
        cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, alpha, beta,
        xDesc, x, yDesc, y,
        bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor,
        resultRunningMean, resultRunningVariance, epsilon, resultSaveMean,
        resultSaveInvVariance);
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance)
{
    trace_dump.dump("cudnnBatchNormalizationForwardTraining");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale, const void *bnBias,
    double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale, const void *bnBias,
    double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnBatchNormalizationForwardTrainingEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBatchNormalizationForwardTrainingEx));
        cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, bnOps, alpha,
        beta, xDesc, xData, zDesc,
        zData, yDesc, yData, bnScaleBiasMeanVarDesc,
        bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
        resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance,
        activationDesc, workspace, workSpaceSizeInBytes, reserveSpace,
        reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale, const void *bnBias,
    double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnBatchNormalizationForwardTrainingEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackward_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
    void *dBnScaleResult, void *dBnBiasResult,
    double epsilon, const void *savedMean,
    const void *savedInvVariance)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackward_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
    void *dBnScaleResult, void *dBnBiasResult,
    double epsilon, const void *savedMean,
    const void *savedInvVariance)
{
    typedef decltype(&cudnnBatchNormalizationBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBatchNormalizationBackward));
        cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, alphaDataDiff, betaDataDiff,
        alphaParamDiff, betaParamDiff, xDesc, x,
        dyDesc, dy, dxDesc, dx,
        dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult,
        epsilon, savedMean, savedInvVariance);
}

cudnnStatus_t cudnnBatchNormalizationBackward_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
    void *dBnScaleResult, void *dBnBiasResult,
    double epsilon, const void *savedMean,
    const void *savedInvVariance)
{
    trace_dump.dump("cudnnBatchNormalizationBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx_prehook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t yDesc,
    const void *yData, const cudnnTensorDescriptor_t dyDesc,
    const void *dyData, const cudnnTensorDescriptor_t dzDesc,
    void *dzData, const cudnnTensorDescriptor_t dxDesc,
    void *dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData, const void *bnBiasData,
    void *dBnScaleData, void *dBnBiasData,
    double epsilon, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx_proxy(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t yDesc,
    const void *yData, const cudnnTensorDescriptor_t dyDesc,
    const void *dyData, const cudnnTensorDescriptor_t dzDesc,
    void *dzData, const cudnnTensorDescriptor_t dxDesc,
    void *dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData, const void *bnBiasData,
    void *dBnScaleData, void *dBnBiasData,
    double epsilon, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnBatchNormalizationBackwardEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBatchNormalizationBackwardEx));
        cudnn_hook_info.func_actual[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, bnOps, alphaDataDiff,
        betaDataDiff, alphaParamDiff, betaParamDiff, xDesc,
        xData, yDesc, yData, dyDesc,
        dyData, dzDesc, dzData, dxDesc,
        dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData,
        dBnScaleData, dBnBiasData, epsilon, savedMean,
        savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes,
        reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx_posthook(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t yDesc,
    const void *yData, const cudnnTensorDescriptor_t dyDesc,
    const void *dyData, const cudnnTensorDescriptor_t dzDesc,
    void *dzData, const cudnnTensorDescriptor_t dxDesc,
    void *dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData, const void *bnBiasData,
    void *dBnScaleData, void *dBnBiasData,
    double epsilon, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnBatchNormalizationBackwardEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize_prehook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize_proxy(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt)
{
    typedef decltype(&cudnnGetNormalizationForwardTrainingWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetNormalizationForwardTrainingWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, normOps, algo,
        xDesc, zDesc, yDesc, normScaleBiasDesc,
        activationDesc, normMeanVarDesc, sizeInBytes, groupCnt);
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize_posthook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt)
{
    trace_dump.dump("cudnnGetNormalizationForwardTrainingWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize_prehook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize_proxy(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt)
{
    typedef decltype(&cudnnGetNormalizationBackwardWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetNormalizationBackwardWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, normOps, algo,
        xDesc, yDesc, dyDesc, dzDesc,
        dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc,
        sizeInBytes, groupCnt);
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize_posthook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt)
{
    trace_dump.dump("cudnnGetNormalizationBackwardWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize_prehook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes, int groupCnt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize_proxy(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes, int groupCnt)
{
    typedef decltype(&cudnnGetNormalizationTrainingReserveSpaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetNormalizationTrainingReserveSpaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, normOps, algo,
        activationDesc, xDesc, sizeInBytes, groupCnt);
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize_posthook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes, int groupCnt)
{
    trace_dump.dump("cudnnGetNormalizationTrainingReserveSpaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardTraining_prehook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes, int groupCnt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardTraining_proxy(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes, int groupCnt)
{
    typedef decltype(&cudnnNormalizationForwardTraining) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_NORMALIZATION_FORWARD_TRAINING])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnNormalizationForwardTraining));
        cudnn_hook_info.func_actual[CUDNN_NORMALIZATION_FORWARD_TRAINING] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, normOps, algo,
        alpha, beta, xDesc, xData,
        normScaleBiasDesc, normScale, normBias, exponentialAverageFactor,
        normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon,
        resultSaveMean, resultSaveInvVariance, activationDesc, zDesc,
        zData, yDesc, yData, workspace,
        workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt);
}

cudnnStatus_t cudnnNormalizationForwardTraining_posthook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes, int groupCnt)
{
    trace_dump.dump("cudnnNormalizationForwardTraining");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationBackward_prehook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc, const void *normScaleData,
    const void *normBiasData, void *dNormScaleData,
    void *dNormBiasData, double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes,
    int groupCnt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationBackward_proxy(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc, const void *normScaleData,
    const void *normBiasData, void *dNormScaleData,
    void *dNormBiasData, double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes,
    int groupCnt)
{
    typedef decltype(&cudnnNormalizationBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_NORMALIZATION_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnNormalizationBackward));
        cudnn_hook_info.func_actual[CUDNN_NORMALIZATION_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, mode, normOps, algo,
        alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
        xDesc, xData, yDesc, yData,
        dyDesc, dyData, dzDesc, dzData,
        dxDesc, dxData, dNormScaleBiasDesc, normScaleData,
        normBiasData, dNormScaleData, dNormBiasData, epsilon,
        normMeanVarDesc, savedMean, savedInvVariance, activationDesc,
        workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes,
        groupCnt);
}

cudnnStatus_t cudnnNormalizationBackward_posthook(
    cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc, const void *normScaleData,
    const void *normBiasData, void *dNormScaleData,
    void *dNormBiasData, double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes,
    int groupCnt)
{
    trace_dump.dump("cudnnNormalizationBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward_prehook(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward_proxy(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta)
{
    typedef decltype(&cudnnSpatialTfGridGeneratorBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSpatialTfGridGeneratorBackward));
        cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, stDesc, dgrid, dtheta);
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward_posthook(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta)
{
    trace_dump.dump("cudnnSpatialTfGridGeneratorBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward_prehook(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid,
    const void *betaDgrid, void *dgrid)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward_proxy(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid,
    const void *betaDgrid, void *dgrid)
{
    typedef decltype(&cudnnSpatialTfSamplerBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSpatialTfSamplerBackward));
        cudnn_hook_info.func_actual[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, stDesc, alpha, xDesc,
        x, beta, dxDesc, dx,
        alphaDgrid, dyDesc, dy, grid,
        betaDgrid, dgrid);
}

cudnnStatus_t cudnnSpatialTfSamplerBackward_posthook(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid,
    const void *betaDgrid, void *dgrid)
{
    trace_dump.dump("cudnnSpatialTfSamplerBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutBackward_prehook(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc, const void *dy,
    const cudnnTensorDescriptor_t dxdesc, void *dx,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutBackward_proxy(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc, const void *dy,
    const cudnnTensorDescriptor_t dxdesc, void *dx,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnDropoutBackward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DROPOUT_BACKWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDropoutBackward));
        cudnn_hook_info.func_actual[CUDNN_DROPOUT_BACKWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, dropoutDesc, dydesc, dy,
        dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnDropoutBackward_posthook(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc, const void *dy,
    const cudnnTensorDescriptor_t dxdesc, void *dx,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnDropoutBackward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize, const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize, const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec)
{
    typedef decltype(&cudnnSetRNNDescriptor_v6) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_DESCRIPTOR_V6])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNDescriptor_v6));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_DESCRIPTOR_V6] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, hiddenSize, numLayers,
        dropoutDesc, inputMode, direction, cellMode,
        algo, mathPrec);
}

cudnnStatus_t cudnnSetRNNDescriptor_v6_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize, const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec)
{
    trace_dump.dump("cudnnSetRNNDescriptor_v6");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize, int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc, cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction, cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize, int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc, cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction, cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec)
{
    typedef decltype(&cudnnGetRNNDescriptor_v6) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_DESCRIPTOR_V6])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNDescriptor_v6));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_DESCRIPTOR_V6] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, hiddenSize, numLayers,
        dropoutDesc, inputMode, direction, cellMode,
        algo, mathPrec);
}

cudnnStatus_t cudnnGetRNNDescriptor_v6_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize, int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc, cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction, cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec)
{
    trace_dump.dump("cudnnGetRNNDescriptor_v6");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt,
    double lclip, double rclip)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt,
    double lclip, double rclip)
{
    typedef decltype(&cudnnRNNSetClip) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_SET_CLIP])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNSetClip));
        cudnn_hook_info.func_actual[CUDNN_RNN_SET_CLIP] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, clipMode, clipNanOpt,
        lclip, rclip);
}

cudnnStatus_t cudnnRNNSetClip_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt,
    double lclip, double rclip)
{
    trace_dump.dump("cudnnRNNSetClip");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt,
    double *lclip, double *rclip)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt,
    double *lclip, double *rclip)
{
    typedef decltype(&cudnnRNNGetClip) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_GET_CLIP])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNGetClip));
        cudnn_hook_info.func_actual[CUDNN_RNN_GET_CLIP] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, clipMode, clipNanOpt,
        lclip, rclip);
}

cudnnStatus_t cudnnRNNGetClip_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt,
    double *lclip, double *rclip)
{
    trace_dump.dump("cudnnRNNGetClip");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNProjectionLayers_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize, const int outProjSize)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNProjectionLayers_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize, const int outProjSize)
{
    typedef decltype(&cudnnSetRNNProjectionLayers) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_PROJECTION_LAYERS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNProjectionLayers));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_PROJECTION_LAYERS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, recProjSize, outProjSize);
}

cudnnStatus_t cudnnSetRNNProjectionLayers_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize, const int outProjSize)
{
    trace_dump.dump("cudnnSetRNNProjectionLayers");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNProjectionLayers_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize, int *outProjSize)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNProjectionLayers_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize, int *outProjSize)
{
    typedef decltype(&cudnnGetRNNProjectionLayers) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_PROJECTION_LAYERS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNProjectionLayers));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_PROJECTION_LAYERS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, recProjSize, outProjSize);
}

cudnnStatus_t cudnnGetRNNProjectionLayers_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize, int *outProjSize)
{
    trace_dump.dump("cudnnGetRNNProjectionLayers");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBuildRNNDynamic_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int miniBatch)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBuildRNNDynamic_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int miniBatch)
{
    typedef decltype(&cudnnBuildRNNDynamic) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BUILD_RNN_DYNAMIC])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBuildRNNDynamic));
        cudnn_hook_info.func_actual[CUDNN_BUILD_RNN_DYNAMIC] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, miniBatch);
}

cudnnStatus_t cudnnBuildRNNDynamic_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int miniBatch)
{
    trace_dump.dump("cudnnBuildRNNDynamic");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetRNNWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        sizeInBytes);
}

cudnnStatus_t cudnnGetRNNWorkspaceSize_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetRNNWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetRNNTrainingReserveSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNTrainingReserveSize));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        sizeInBytes);
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetRNNTrainingReserveSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode, cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize, size_t *reserveSpaceSize)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode, cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize, size_t *reserveSpaceSize)
{
    typedef decltype(&cudnnGetRNNTempSpaceSizes) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_TEMP_SPACE_SIZES])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNTempSpaceSizes));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_TEMP_SPACE_SIZES] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, fMode, xDesc,
        workSpaceSize, reserveSpaceSize);
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode, cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize, size_t *reserveSpaceSize)
{
    trace_dump.dump("cudnnGetRNNTempSpaceSizes");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNParamsSize_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
    cudnnDataType_t dataType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNParamsSize_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
    cudnnDataType_t dataType)
{
    typedef decltype(&cudnnGetRNNParamsSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_PARAMS_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNParamsSize));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_PARAMS_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, xDesc, sizeInBytes,
        dataType);
}

cudnnStatus_t cudnnGetRNNParamsSize_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
    cudnnDataType_t dataType)
{
    trace_dump.dump("cudnnGetRNNParamsSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize)
{
    typedef decltype(&cudnnGetRNNWeightSpaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNWeightSpaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, weightSpaceSize);
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize)
{
    trace_dump.dump("cudnnGetRNNWeightSpaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat)
{
    typedef decltype(&cudnnGetRNNLinLayerMatrixParams) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNLinLayerMatrixParams));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, pseudoLayer, xDesc,
        wDesc, w, linLayerID, linLayerMatDesc,
        linLayerMat);
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat)
{
    trace_dump.dump("cudnnGetRNNLinLayerMatrixParams");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias)
{
    typedef decltype(&cudnnGetRNNLinLayerBiasParams) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNLinLayerBiasParams));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, pseudoLayer, xDesc,
        wDesc, w, linLayerID, linLayerBiasDesc,
        linLayerBias);
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias)
{
    trace_dump.dump("cudnnGetRNNLinLayerBiasParams");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightParams_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer, size_t weightSpaceSize,
    const void *weightSpace, int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc, void **mAddr,
    cudnnTensorDescriptor_t bDesc, void **bAddr)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightParams_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer, size_t weightSpaceSize,
    const void *weightSpace, int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc, void **mAddr,
    cudnnTensorDescriptor_t bDesc, void **bAddr)
{
    typedef decltype(&cudnnGetRNNWeightParams) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_WEIGHT_PARAMS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNWeightParams));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_WEIGHT_PARAMS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, pseudoLayer, weightSpaceSize,
        weightSpace, linLayerID, mDesc, mAddr,
        bDesc, bAddr);
}

cudnnStatus_t cudnnGetRNNWeightParams_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer, size_t weightSpaceSize,
    const void *weightSpace, int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc, void **mAddr,
    cudnnTensorDescriptor_t bDesc, void **bAddr)
{
    trace_dump.dump("cudnnGetRNNWeightParams");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNForwardInference) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_INFERENCE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNForwardInference));
        cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_INFERENCE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        x, hxDesc, hx, cxDesc,
        cx, wDesc, w, yDesc,
        y, hyDesc, hy, cyDesc,
        cy, workSpace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNForwardInference_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNForwardInference");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInferenceEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInferenceEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNForwardInferenceEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_INFERENCE_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNForwardInferenceEx));
        cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_INFERENCE_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, xDesc, x,
        hxDesc, hx, cxDesc, cx,
        wDesc, w, yDesc, y,
        hyDesc, hy, cyDesc, cy,
        kDesc, keys, cDesc, cAttn,
        iDesc, iAttn, qDesc, queries,
        workSpace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNForwardInferenceEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNForwardInferenceEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForward_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnRNNDataDescriptor_t yDesc, void *y,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    void *hy, cudnnTensorDescriptor_t cDesc,
    const void *cx, void *cy,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForward_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnRNNDataDescriptor_t yDesc, void *y,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    void *hy, cudnnTensorDescriptor_t cDesc,
    const void *cx, void *cy,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    typedef decltype(&cudnnRNNForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNForward));
        cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, fwdMode, devSeqLengths,
        xDesc, x, yDesc, y,
        hDesc, hx, hy, cDesc,
        cx, cy, weightSpaceSize, weightSpace,
        workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);
}

cudnnStatus_t cudnnRNNForward_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnRNNDataDescriptor_t yDesc, void *y,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    void *hy, cudnnTensorDescriptor_t cDesc,
    const void *cx, void *cy,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    trace_dump.dump("cudnnRNNForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc)
{
    typedef decltype(&cudnnSetRNNAlgorithmDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNAlgorithmDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, algoDesc);
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc)
{
    trace_dump.dump("cudnnSetRNNAlgorithmDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    typedef decltype(&cudnnGetRNNForwardInferenceAlgorithmMaxCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNForwardInferenceAlgorithmMaxCount));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, count);
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    trace_dump.dump("cudnnGetRNNForwardInferenceAlgorithmMaxCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes)
{
    typedef decltype(&cudnnFindRNNForwardInferenceAlgorithmEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindRNNForwardInferenceAlgorithmEx));
        cudnn_hook_info.func_actual[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        x, hxDesc, hx, cxDesc,
        cx, wDesc, w, yDesc,
        y, hyDesc, hy, cyDesc,
        cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
        perfResults, workspace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes)
{
    trace_dump.dump("cudnnFindRNNForwardInferenceAlgorithmEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers_prehook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers_proxy(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnGetMultiHeadAttnBuffers) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetMultiHeadAttnBuffers));
        cudnn_hook_info.func_actual[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes,
        reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers_posthook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnGetMultiHeadAttnBuffers");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights_prehook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
    const void *weights, cudnnTensorDescriptor_t wDesc,
    void **wAddr)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights_proxy(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
    const void *weights, cudnnTensorDescriptor_t wDesc,
    void **wAddr)
{
    typedef decltype(&cudnnGetMultiHeadAttnWeights) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetMultiHeadAttnWeights));
        cudnn_hook_info.func_actual[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, attnDesc, wKind, weightSizeInBytes,
        weights, wDesc, wAddr);
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights_posthook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
    const void *weights, cudnnTensorDescriptor_t wDesc,
    void **wAddr)
{
    trace_dump.dump("cudnnGetMultiHeadAttnWeights");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnForward_prehook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    int currIdx, const int loWinIdx[],
    const int hiWinIdx[], const int devSeqLengthsQO[],
    const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out,
    size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnForward_proxy(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    int currIdx, const int loWinIdx[],
    const int hiWinIdx[], const int devSeqLengthsQO[],
    const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out,
    size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace)
{
    typedef decltype(&cudnnMultiHeadAttnForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_MULTI_HEAD_ATTN_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnMultiHeadAttnForward));
        cudnn_hook_info.func_actual[CUDNN_MULTI_HEAD_ATTN_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, attnDesc, currIdx, loWinIdx,
        hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc,
        queries, residuals, kDesc, keys,
        vDesc, values, oDesc, out,
        weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace,
        reserveSpaceSizeInBytes, reserveSpace);
}

cudnnStatus_t cudnnMultiHeadAttnForward_posthook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    int currIdx, const int loWinIdx[],
    const int hiWinIdx[], const int devSeqLengthsQO[],
    const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out,
    size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace)
{
    trace_dump.dump("cudnnMultiHeadAttnForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTraining_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTraining_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNForwardTraining) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_TRAINING])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNForwardTraining));
        cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_TRAINING] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        x, hxDesc, hx, cxDesc,
        cx, wDesc, w, yDesc,
        y, hyDesc, hy, cyDesc,
        cy, workSpace, workSpaceSizeInBytes, reserveSpace,
        reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNForwardTraining_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNForwardTraining");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNBackwardData) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_DATA])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNBackwardData));
        cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_DATA] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, yDesc,
        y, dyDesc, dy, dhyDesc,
        dhy, dcyDesc, dcy, wDesc,
        w, hxDesc, hx, cxDesc,
        cx, dxDesc, dx, dhxDesc,
        dhx, dcxDesc, dcx, workSpace,
        workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNBackwardData_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNBackwardData");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_v8_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t yDesc,
    const void *y, const void *dy,
    cudnnRNNDataDescriptor_t xDesc, void *dx,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    const void *dhy, void *dhx,
    cudnnTensorDescriptor_t cDesc, const void *cx,
    const void *dcy, void *dcx,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_v8_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t yDesc,
    const void *y, const void *dy,
    cudnnRNNDataDescriptor_t xDesc, void *dx,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    const void *dhy, void *dhx,
    cudnnTensorDescriptor_t cDesc, const void *cx,
    const void *dcy, void *dcx,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    typedef decltype(&cudnnRNNBackwardData_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_DATA_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNBackwardData_v8));
        cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_DATA_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, devSeqLengths, yDesc,
        y, dy, xDesc, dx,
        hDesc, hx, dhy, dhx,
        cDesc, cx, dcy, dcx,
        weightSpaceSize, weightSpace, workSpaceSize, workSpace,
        reserveSpaceSize, reserveSpace);
}

cudnnStatus_t cudnnRNNBackwardData_v8_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t yDesc,
    const void *y, const void *dy,
    cudnnRNNDataDescriptor_t xDesc, void *dx,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    const void *dhy, void *dhx,
    cudnnTensorDescriptor_t cDesc, const void *cx,
    const void *dcy, void *dcx,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    trace_dump.dump("cudnnRNNBackwardData_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const void *workSpace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const void *workSpace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNBackwardWeights) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_WEIGHTS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNBackwardWeights));
        cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_WEIGHTS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        x, hxDesc, hx, yDesc,
        y, workSpace, workSpaceSizeInBytes, dwDesc,
        dw, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNBackwardWeights_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const void *workSpace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNBackwardWeights");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_v8_prehook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    cudnnRNNDataDescriptor_t yDesc, const void *y,
    size_t weightSpaceSize, void *dweightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_v8_proxy(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    cudnnRNNDataDescriptor_t yDesc, const void *y,
    size_t weightSpaceSize, void *dweightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    typedef decltype(&cudnnRNNBackwardWeights_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_WEIGHTS_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNBackwardWeights_v8));
        cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_WEIGHTS_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, addGrad, devSeqLengths,
        xDesc, x, hDesc, hx,
        yDesc, y, weightSpaceSize, dweightSpace,
        workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);
}

cudnnStatus_t cudnnRNNBackwardWeights_v8_posthook(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    cudnnRNNDataDescriptor_t yDesc, const void *y,
    size_t weightSpaceSize, void *dweightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace)
{
    trace_dump.dump("cudnnRNNBackwardWeights_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTrainingEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTrainingEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNForwardTrainingEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_TRAINING_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNForwardTrainingEx));
        cudnn_hook_info.func_actual[CUDNN_RNN_FORWARD_TRAINING_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, xDesc, x,
        hxDesc, hx, cxDesc, cx,
        wDesc, w, yDesc, y,
        hyDesc, hy, cyDesc, cy,
        kDesc, keys, cDesc, cAttn,
        iDesc, iAttn, qDesc, queries,
        workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNForwardTrainingEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNForwardTrainingEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardDataEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, void *dkeys,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardDataEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, void *dkeys,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNBackwardDataEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_DATA_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNBackwardDataEx));
        cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_DATA_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, yDesc, y,
        dyDesc, dy, dcDesc, dcAttn,
        dhyDesc, dhy, dcyDesc, dcy,
        wDesc, w, hxDesc, hx,
        cxDesc, cx, dxDesc, dx,
        dhxDesc, dhx, dcxDesc, dcx,
        dkDesc, dkeys, workSpace, workSpaceSizeInBytes,
        reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNBackwardDataEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, void *dkeys,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNBackwardDataEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    void *workSpace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    void *workSpace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnRNNBackwardWeightsEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_WEIGHTS_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNBackwardWeightsEx));
        cudnn_hook_info.func_actual[CUDNN_RNN_BACKWARD_WEIGHTS_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, xDesc, x,
        hxDesc, hx, yDesc, y,
        workSpace, workSpaceSizeInBytes, dwDesc, dw,
        reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNBackwardWeightsEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    void *workSpace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnRNNBackwardWeightsEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    typedef decltype(&cudnnGetRNNForwardTrainingAlgorithmMaxCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNForwardTrainingAlgorithmMaxCount));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, count);
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    trace_dump.dump("cudnnGetRNNForwardTrainingAlgorithmMaxCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnFindRNNForwardTrainingAlgorithmEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindRNNForwardTrainingAlgorithmEx));
        cudnn_hook_info.func_actual[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        x, hxDesc, hx, cxDesc,
        cx, wDesc, w, yDesc,
        y, hyDesc, hy, cyDesc,
        cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
        perfResults, workspace, workSpaceSizeInBytes, reserveSpace,
        reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnFindRNNForwardTrainingAlgorithmEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    typedef decltype(&cudnnGetRNNBackwardDataAlgorithmMaxCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNBackwardDataAlgorithmMaxCount));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, count);
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    trace_dump.dump("cudnnGetRNNBackwardDataAlgorithmMaxCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnFindRNNBackwardDataAlgorithmEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindRNNBackwardDataAlgorithmEx));
        cudnn_hook_info.func_actual[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, yDesc,
        y, dyDesc, dy, dhyDesc,
        dhy, dcyDesc, dcy, wDesc,
        w, hxDesc, hx, cxDesc,
        cx, dxDesc, dx, dhxDesc,
        dhx, dcxDesc, dcx, findIntensity,
        requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
        workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnFindRNNBackwardDataAlgorithmEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    typedef decltype(&cudnnGetRNNBackwardWeightsAlgorithmMaxCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNBackwardWeightsAlgorithmMaxCount));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, count);
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count)
{
    trace_dump.dump("cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx_prehook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx_proxy(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    typedef decltype(&cudnnFindRNNBackwardWeightsAlgorithmEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindRNNBackwardWeightsAlgorithmEx));
        cudnn_hook_info.func_actual[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, rnnDesc, seqLength, xDesc,
        x, hxDesc, hx, yDesc,
        y, findIntensity, requestedAlgoCount, returnedAlgoCount,
        perfResults, workspace, workSpaceSizeInBytes, dwDesc,
        dw, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx_posthook(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    trace_dump.dump("cudnnFindRNNBackwardWeightsAlgorithmEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData_prehook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsDQDO[], const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys, const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes,
    void *workSpace, size_t reserveSpaceSizeInBytes,
    void *reserveSpace)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData_proxy(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsDQDO[], const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys, const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes,
    void *workSpace, size_t reserveSpaceSizeInBytes,
    void *reserveSpace)
{
    typedef decltype(&cudnnMultiHeadAttnBackwardData) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnMultiHeadAttnBackwardData));
        cudnn_hook_info.func_actual[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, attnDesc, loWinIdx, hiWinIdx,
        devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout,
        dqDesc, dqueries, queries, dkDesc,
        dkeys, keys, dvDesc, dvalues,
        values, weightSizeInBytes, weights, workSpaceSizeInBytes,
        workSpace, reserveSpaceSizeInBytes, reserveSpace);
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData_posthook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsDQDO[], const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys, const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes,
    void *workSpace, size_t reserveSpaceSizeInBytes,
    void *reserveSpace)
{
    trace_dump.dump("cudnnMultiHeadAttnBackwardData");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights_prehook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const cudnnSeqDataDescriptor_t kDesc,
    const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, size_t weightSizeInBytes,
    const void *weights, void *dweights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights_proxy(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const cudnnSeqDataDescriptor_t kDesc,
    const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, size_t weightSizeInBytes,
    const void *weights, void *dweights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace)
{
    typedef decltype(&cudnnMultiHeadAttnBackwardWeights) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnMultiHeadAttnBackwardWeights));
        cudnn_hook_info.func_actual[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, attnDesc, addGrad, qDesc,
        queries, kDesc, keys, vDesc,
        values, doDesc, dout, weightSizeInBytes,
        weights, dweights, workSpaceSizeInBytes, workSpace,
        reserveSpaceSizeInBytes, reserveSpace);
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights_posthook(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const cudnnSeqDataDescriptor_t kDesc,
    const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, size_t weightSizeInBytes,
    const void *weights, void *dweights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace)
{
    trace_dump.dump("cudnnMultiHeadAttnBackwardWeights");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int hostLabels[],
    const int hostLabelLengths[], const int hostInputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace,
    size_t workSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int hostLabels[],
    const int hostLabelLengths[], const int hostInputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace,
    size_t workSpaceSizeInBytes)
{
    typedef decltype(&cudnnCTCLoss) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CTC_LOSS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCTCLoss));
        cudnn_hook_info.func_actual[CUDNN_CTC_LOSS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, probsDesc, probs, hostLabels,
        hostLabelLengths, hostInputLengths, costs, gradientsDesc,
        gradients, algo, ctcLossDesc, workspace,
        workSpaceSizeInBytes);
}

cudnnStatus_t cudnnCTCLoss_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int hostLabels[],
    const int hostLabelLengths[], const int hostInputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace,
    size_t workSpaceSizeInBytes)
{
    trace_dump.dump("cudnnCTCLoss");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_v8_prehook(
    cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int labels[],
    const int labelLengths[], const int inputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, size_t workSpaceSizeInBytes,
    void *workspace)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_v8_proxy(
    cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int labels[],
    const int labelLengths[], const int inputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, size_t workSpaceSizeInBytes,
    void *workspace)
{
    typedef decltype(&cudnnCTCLoss_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CTC_LOSS_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCTCLoss_v8));
        cudnn_hook_info.func_actual[CUDNN_CTC_LOSS_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, algo, ctcLossDesc, probsDesc,
        probs, labels, labelLengths, inputLengths,
        costs, gradientsDesc, gradients, workSpaceSizeInBytes,
        workspace);
}

cudnnStatus_t cudnnCTCLoss_v8_posthook(
    cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int labels[],
    const int labelLengths[], const int inputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, size_t workSpaceSizeInBytes,
    void *workspace)
{
    trace_dump.dump("cudnnCTCLoss_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths,
    cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths,
    cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetCTCLossWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetCTCLossWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, probsDesc, gradientsDesc, labels,
        labelLengths, inputLengths, algo, ctcLossDesc,
        sizeInBytes);
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths,
    cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetCTCLossWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8_prehook(
    cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8_proxy(
    cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetCTCLossWorkspaceSize_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetCTCLossWorkspaceSize_v8));
        cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, algo, ctcLossDesc, probsDesc,
        gradientsDesc, sizeInBytes);
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8_posthook(
    cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetCTCLossWorkspaceSize_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount_prehook(
    cudnnHandle_t handle, int *count)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount_proxy(
    cudnnHandle_t handle, int *count)
{
    typedef decltype(&cudnnGetConvolutionForwardAlgorithmMaxCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionForwardAlgorithmMaxCount));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, count);
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount_posthook(
    cudnnHandle_t handle, int *count)
{
    trace_dump.dump("cudnnGetConvolutionForwardAlgorithmMaxCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults)
{
    typedef decltype(&cudnnGetConvolutionForwardAlgorithm_v7) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionForwardAlgorithm_v7));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, srcDesc, filterDesc, convDesc,
        destDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults)
{
    trace_dump.dump("cudnnGetConvolutionForwardAlgorithm_v7");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults)
{
    typedef decltype(&cudnnFindConvolutionForwardAlgorithm) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindConvolutionForwardAlgorithm));
        cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, xDesc, wDesc, convDesc,
        yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults)
{
    trace_dump.dump("cudnnFindConvolutionForwardAlgorithm");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    typedef decltype(&cudnnFindConvolutionForwardAlgorithmEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindConvolutionForwardAlgorithmEx));
        cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, xDesc, x, wDesc,
        w, convDesc, yDesc, y,
        requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
        workSpaceSizeInBytes);
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    trace_dump.dump("cudnnFindConvolutionForwardAlgorithmEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnIm2Col_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, void *colBuffer)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnIm2Col_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, void *colBuffer)
{
    typedef decltype(&cudnnIm2Col) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_IM_2_COL])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnIm2Col));
        cudnn_hook_info.func_actual[CUDNN_IM_2_COL] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, xDesc, x, wDesc,
        convDesc, colBuffer);
}

cudnnStatus_t cudnnIm2Col_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, void *colBuffer)
{
    trace_dump.dump("cudnnIm2Col");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReorderFilterAndBias_prehook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias,
    const void *biasData, void *reorderedBiasData)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReorderFilterAndBias_proxy(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias,
    const void *biasData, void *reorderedBiasData)
{
    typedef decltype(&cudnnReorderFilterAndBias) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_REORDER_FILTER_AND_BIAS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnReorderFilterAndBias));
        cudnn_hook_info.func_actual[CUDNN_REORDER_FILTER_AND_BIAS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, filterDesc, reorderType, filterData,
        reorderedFilterData, reorderBias, biasData, reorderedBiasData);
}

cudnnStatus_t cudnnReorderFilterAndBias_posthook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias,
    const void *biasData, void *reorderedBiasData)
{
    trace_dump.dump("cudnnReorderFilterAndBias");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetConvolutionForwardWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionForwardWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, xDesc, wDesc, convDesc,
        yDesc, algo, sizeInBytes);
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetConvolutionForwardWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward_prehook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward_proxy(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    typedef decltype(&cudnnConvolutionForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnConvolutionForward));
        cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, alpha, xDesc, x,
        wDesc, w, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, yDesc,
        y);
}

cudnnStatus_t cudnnConvolutionForward_posthook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y)
{
    trace_dump.dump("cudnnConvolutionForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward_prehook(
    cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *alpha2, const cudnnTensorDescriptor_t zDesc,
    const void *z, const cudnnTensorDescriptor_t biasDesc,
    const void *bias, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward_proxy(
    cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *alpha2, const cudnnTensorDescriptor_t zDesc,
    const void *z, const cudnnTensorDescriptor_t biasDesc,
    const void *bias, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    typedef decltype(&cudnnConvolutionBiasActivationForward) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnConvolutionBiasActivationForward));
        cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, alpha1, xDesc, x,
        wDesc, w, convDesc, algo,
        workSpace, workSpaceSizeInBytes, alpha2, zDesc,
        z, biasDesc, bias, activationDesc,
        yDesc, y);
}

cudnnStatus_t cudnnConvolutionBiasActivationForward_posthook(
    cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *alpha2, const cudnnTensorDescriptor_t zDesc,
    const void *z, const cudnnTensorDescriptor_t biasDesc,
    const void *bias, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y)
{
    trace_dump.dump("cudnnConvolutionBiasActivationForward");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount_prehook(
    cudnnHandle_t handle, int *count)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount_proxy(
    cudnnHandle_t handle, int *count)
{
    typedef decltype(&cudnnGetConvolutionBackwardDataAlgorithmMaxCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionBackwardDataAlgorithmMaxCount));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, count);
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount_posthook(
    cudnnHandle_t handle, int *count)
{
    trace_dump.dump("cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm_prehook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm_proxy(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    typedef decltype(&cudnnFindConvolutionBackwardDataAlgorithm) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindConvolutionBackwardDataAlgorithm));
        cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, wDesc, dyDesc, convDesc,
        dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm_posthook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    trace_dump.dump("cudnnFindConvolutionBackwardDataAlgorithm");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx_prehook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx_proxy(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    typedef decltype(&cudnnFindConvolutionBackwardDataAlgorithmEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindConvolutionBackwardDataAlgorithmEx));
        cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, wDesc, w, dyDesc,
        dy, convDesc, dxDesc, dx,
        requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
        workSpaceSizeInBytes);
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx_posthook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    trace_dump.dump("cudnnFindConvolutionBackwardDataAlgorithmEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7_prehook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7_proxy(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    typedef decltype(&cudnnGetConvolutionBackwardDataAlgorithm_v7) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionBackwardDataAlgorithm_v7));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, filterDesc, diffDesc, convDesc,
        gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7_posthook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    trace_dump.dump("cudnnGetConvolutionBackwardDataAlgorithm_v7");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize_prehook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize_proxy(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetConvolutionBackwardDataWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionBackwardDataWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, wDesc, dyDesc, convDesc,
        dxDesc, algo, sizeInBytes);
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize_posthook(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetConvolutionBackwardDataWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData_prehook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData_proxy(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    typedef decltype(&cudnnConvolutionBackwardData) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BACKWARD_DATA])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnConvolutionBackwardData));
        cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BACKWARD_DATA] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, alpha, wDesc, w,
        dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dxDesc,
        dx);
}

cudnnStatus_t cudnnConvolutionBackwardData_posthook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx)
{
    trace_dump.dump("cudnnConvolutionBackwardData");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors_prehook(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc, cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc, cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc, cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc, cudnnTensorTransformDescriptor_t gradUnfoldTransDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors_proxy(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc, cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc, cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc, cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc, cudnnTensorTransformDescriptor_t gradUnfoldTransDesc)
{
    typedef decltype(&cudnnGetFoldedConvBackwardDataDescriptors) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetFoldedConvBackwardDataDescriptors));
        cudnn_hook_info.func_actual[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, filterDesc, diffDesc, convDesc,
        gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc,
        foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc,
        gradFoldTransDesc, gradUnfoldTransDesc);
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors_posthook(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc, cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc, cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc, cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc, cudnnTensorTransformDescriptor_t gradUnfoldTransDesc)
{
    trace_dump.dump("cudnnGetFoldedConvBackwardDataDescriptors");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_prehook(
    cudnnHandle_t handle, int *count)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_proxy(
    cudnnHandle_t handle, int *count)
{
    typedef decltype(&cudnnGetConvolutionBackwardFilterAlgorithmMaxCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, count);
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_posthook(
    cudnnHandle_t handle, int *count)
{
    trace_dump.dump("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    typedef decltype(&cudnnFindConvolutionBackwardFilterAlgorithm) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindConvolutionBackwardFilterAlgorithm));
        cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, xDesc, dyDesc, convDesc,
        dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    trace_dump.dump("cudnnFindConvolutionBackwardFilterAlgorithm");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *y, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *y, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    typedef decltype(&cudnnFindConvolutionBackwardFilterAlgorithmEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFindConvolutionBackwardFilterAlgorithmEx));
        cudnn_hook_info.func_actual[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, xDesc, x, dyDesc,
        y, convDesc, dwDesc, dw,
        requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
        workSpaceSizeInBytes);
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *y, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes)
{
    trace_dump.dump("cudnnFindConvolutionBackwardFilterAlgorithmEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    typedef decltype(&cudnnGetConvolutionBackwardFilterAlgorithm_v7) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionBackwardFilterAlgorithm_v7));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, srcDesc, diffDesc, convDesc,
        gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    trace_dump.dump("cudnnGetConvolutionBackwardFilterAlgorithm_v7");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize_prehook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize_proxy(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes)
{
    typedef decltype(&cudnnGetConvolutionBackwardFilterWorkspaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionBackwardFilterWorkspaceSize));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, xDesc, dyDesc, convDesc,
        gradDesc, algo, sizeInBytes);
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize_posthook(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes)
{
    trace_dump.dump("cudnnGetConvolutionBackwardFilterWorkspaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_prehook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnFilterDescriptor_t dwDesc,
    void *dw)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_proxy(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnFilterDescriptor_t dwDesc,
    void *dw)
{
    typedef decltype(&cudnnConvolutionBackwardFilter) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BACKWARD_FILTER])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnConvolutionBackwardFilter));
        cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BACKWARD_FILTER] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, alpha, xDesc, x,
        dyDesc, dy, convDesc, algo,
        workSpace, workSpaceSizeInBytes, beta, dwDesc,
        dw);
}

cudnnStatus_t cudnnConvolutionBackwardFilter_posthook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnFilterDescriptor_t dwDesc,
    void *dw)
{
    trace_dump.dump("cudnnConvolutionBackwardFilter");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardBias_prehook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dbDesc,
    void *db)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardBias_proxy(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dbDesc,
    void *db)
{
    typedef decltype(&cudnnConvolutionBackwardBias) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BACKWARD_BIAS])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnConvolutionBackwardBias));
        cudnn_hook_info.func_actual[CUDNN_CONVOLUTION_BACKWARD_BIAS] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, alpha, dyDesc, dy,
        beta, dbDesc, db);
}

cudnnStatus_t cudnnConvolutionBackwardBias_posthook(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dbDesc,
    void *db)
{
    trace_dump.dump("cudnnConvolutionBackwardBias");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMakeFusedOpsPlan_prehook(
    cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack, size_t *workspaceSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMakeFusedOpsPlan_proxy(
    cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack, size_t *workspaceSizeInBytes)
{
    typedef decltype(&cudnnMakeFusedOpsPlan) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_MAKE_FUSED_OPS_PLAN])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnMakeFusedOpsPlan));
        cudnn_hook_info.func_actual[CUDNN_MAKE_FUSED_OPS_PLAN] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, plan, constPack, workspaceSizeInBytes);
}

cudnnStatus_t cudnnMakeFusedOpsPlan_posthook(
    cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack, size_t *workspaceSizeInBytes)
{
    trace_dump.dump("cudnnMakeFusedOpsPlan");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFusedOpsExecute_prehook(
    cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFusedOpsExecute_proxy(
    cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack)
{
    typedef decltype(&cudnnFusedOpsExecute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_FUSED_OPS_EXECUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnFusedOpsExecute));
        cudnn_hook_info.func_actual[CUDNN_FUSED_OPS_EXECUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, plan, varPack);
}

cudnnStatus_t cudnnFusedOpsExecute_posthook(
    cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack)
{
    trace_dump.dump("cudnnFusedOpsExecute");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendExecute_prehook(
    cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendExecute_proxy(
    cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack)
{
    typedef decltype(&cudnnBackendExecute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BACKEND_EXECUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBackendExecute));
        cudnn_hook_info.func_actual[CUDNN_BACKEND_EXECUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        handle, executionPlan, variantPack);
}

cudnnStatus_t cudnnBackendExecute_posthook(
    cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack)
{
    trace_dump.dump("cudnnBackendExecute");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetProperty_prehook(
    libraryPropertyType type, int *value)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetProperty_proxy(
    libraryPropertyType type, int *value)
{
    typedef decltype(&cudnnGetProperty) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_PROPERTY])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetProperty));
        cudnn_hook_info.func_actual[CUDNN_GET_PROPERTY] = actual_func;
    }
    return ((func_type)actual_func)(
        type, value);
}

cudnnStatus_t cudnnGetProperty_posthook(
    libraryPropertyType type, int *value)
{
    trace_dump.dump("cudnnGetProperty");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorDescriptor_prehook(
    cudnnTensorDescriptor_t *tensorDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorDescriptor_proxy(
    cudnnTensorDescriptor_t *tensorDesc)
{
    typedef decltype(&cudnnCreateTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc);
}

cudnnStatus_t cudnnCreateTensorDescriptor_posthook(
    cudnnTensorDescriptor_t *tensorDesc)
{
    trace_dump.dump("cudnnCreateTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor_prehook(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int n,
    int c, int h,
    int w)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor_proxy(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int n,
    int c, int h,
    int w)
{
    typedef decltype(&cudnnSetTensor4dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_4D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetTensor4dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_4D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc, format, dataType, n,
        c, h, w);
}

cudnnStatus_t cudnnSetTensor4dDescriptor_posthook(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int n,
    int c, int h,
    int w)
{
    trace_dump.dump("cudnnSetTensor4dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx_prehook(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int n, int c,
    int h, int w,
    int nStride, int cStride,
    int hStride, int wStride)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx_proxy(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int n, int c,
    int h, int w,
    int nStride, int cStride,
    int hStride, int wStride)
{
    typedef decltype(&cudnnSetTensor4dDescriptorEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetTensor4dDescriptorEx));
        cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc, dataType, n, c,
        h, w, nStride, cStride,
        hStride, wStride);
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx_posthook(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int n, int c,
    int h, int w,
    int nStride, int cStride,
    int hStride, int wStride)
{
    trace_dump.dump("cudnnSetTensor4dDescriptorEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor_prehook(
    const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType,
    int *n, int *c,
    int *h, int *w,
    int *nStride, int *cStride,
    int *hStride, int *wStride)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor_proxy(
    const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType,
    int *n, int *c,
    int *h, int *w,
    int *nStride, int *cStride,
    int *hStride, int *wStride)
{
    typedef decltype(&cudnnGetTensor4dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_4D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetTensor4dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_4D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc, dataType, n, c,
        h, w, nStride, cStride,
        hStride, wStride);
}

cudnnStatus_t cudnnGetTensor4dDescriptor_posthook(
    const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType,
    int *n, int *c,
    int *h, int *w,
    int *nStride, int *cStride,
    int *hStride, int *wStride)
{
    trace_dump.dump("cudnnGetTensor4dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor_prehook(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const int strideA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor_proxy(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const int strideA[])
{
    typedef decltype(&cudnnSetTensorNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetTensorNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc, dataType, nbDims, dimA,
        strideA);
}

cudnnStatus_t cudnnSetTensorNdDescriptor_posthook(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const int strideA[])
{
    trace_dump.dump("cudnnSetTensorNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx_prehook(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims,
    const int dimA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx_proxy(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims,
    const int dimA[])
{
    typedef decltype(&cudnnSetTensorNdDescriptorEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetTensorNdDescriptorEx));
        cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc, format, dataType, nbDims,
        dimA);
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx_posthook(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims,
    const int dimA[])
{
    trace_dump.dump("cudnnSetTensorNdDescriptorEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor_prehook(
    const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, int *nbDims,
    int dimA[], int strideA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor_proxy(
    const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, int *nbDims,
    int dimA[], int strideA[])
{
    typedef decltype(&cudnnGetTensorNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetTensorNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc, nbDimsRequested, dataType, nbDims,
        dimA, strideA);
}

cudnnStatus_t cudnnGetTensorNdDescriptor_posthook(
    const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, int *nbDims,
    int dimA[], int strideA[])
{
    trace_dump.dump("cudnnGetTensorNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorSizeInBytes_prehook(
    const cudnnTensorDescriptor_t tensorDesc, size_t *size)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorSizeInBytes_proxy(
    const cudnnTensorDescriptor_t tensorDesc, size_t *size)
{
    typedef decltype(&cudnnGetTensorSizeInBytes) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_SIZE_IN_BYTES])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetTensorSizeInBytes));
        cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_SIZE_IN_BYTES] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc, size);
}

cudnnStatus_t cudnnGetTensorSizeInBytes_posthook(
    const cudnnTensorDescriptor_t tensorDesc, size_t *size)
{
    trace_dump.dump("cudnnGetTensorSizeInBytes");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor_prehook(
    cudnnTensorDescriptor_t tensorDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor_proxy(
    cudnnTensorDescriptor_t tensorDesc)
{
    typedef decltype(&cudnnDestroyTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        tensorDesc);
}

cudnnStatus_t cudnnDestroyTensorDescriptor_posthook(
    cudnnTensorDescriptor_t tensorDesc)
{
    trace_dump.dump("cudnnDestroyTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnInitTransformDest_prehook(
    const cudnnTensorTransformDescriptor_t transformDesc, const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnInitTransformDest_proxy(
    const cudnnTensorTransformDescriptor_t transformDesc, const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes)
{
    typedef decltype(&cudnnInitTransformDest) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_INIT_TRANSFORM_DEST])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnInitTransformDest));
        cudnn_hook_info.func_actual[CUDNN_INIT_TRANSFORM_DEST] = actual_func;
    }
    return ((func_type)actual_func)(
        transformDesc, srcDesc, destDesc, destSizeInBytes);
}

cudnnStatus_t cudnnInitTransformDest_posthook(
    const cudnnTensorTransformDescriptor_t transformDesc, const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes)
{
    trace_dump.dump("cudnnInitTransformDest");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t *transformDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t *transformDesc)
{
    typedef decltype(&cudnnCreateTensorTransformDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateTensorTransformDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        transformDesc);
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t *transformDesc)
{
    trace_dump.dump("cudnnCreateTensorTransformDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t padBeforeA[],
    const int32_t padAfterA[], const uint32_t foldA[],
    const cudnnFoldingDirection_t direction)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t padBeforeA[],
    const int32_t padAfterA[], const uint32_t foldA[],
    const cudnnFoldingDirection_t direction)
{
    typedef decltype(&cudnnSetTensorTransformDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetTensorTransformDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        transformDesc, nbDims, destFormat, padBeforeA,
        padAfterA, foldA, direction);
}

cudnnStatus_t cudnnSetTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t padBeforeA[],
    const int32_t padAfterA[], const uint32_t foldA[],
    const cudnnFoldingDirection_t direction)
{
    trace_dump.dump("cudnnSetTensorTransformDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t padBeforeA[],
    int32_t padAfterA[], uint32_t foldA[],
    cudnnFoldingDirection_t *direction)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t padBeforeA[],
    int32_t padAfterA[], uint32_t foldA[],
    cudnnFoldingDirection_t *direction)
{
    typedef decltype(&cudnnGetTensorTransformDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetTensorTransformDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        transformDesc, nbDimsRequested, destFormat, padBeforeA,
        padAfterA, foldA, direction);
}

cudnnStatus_t cudnnGetTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t padBeforeA[],
    int32_t padAfterA[], uint32_t foldA[],
    cudnnFoldingDirection_t *direction)
{
    trace_dump.dump("cudnnGetTensorTransformDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t transformDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t transformDesc)
{
    typedef decltype(&cudnnDestroyTensorTransformDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyTensorTransformDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        transformDesc);
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t transformDesc)
{
    trace_dump.dump("cudnnDestroyTensorTransformDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateOpTensorDescriptor_prehook(
    cudnnOpTensorDescriptor_t *opTensorDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateOpTensorDescriptor_proxy(
    cudnnOpTensorDescriptor_t *opTensorDesc)
{
    typedef decltype(&cudnnCreateOpTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateOpTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        opTensorDesc);
}

cudnnStatus_t cudnnCreateOpTensorDescriptor_posthook(
    cudnnOpTensorDescriptor_t *opTensorDesc)
{
    trace_dump.dump("cudnnCreateOpTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor_prehook(
    cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor_proxy(
    cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt)
{
    typedef decltype(&cudnnSetOpTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_OP_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetOpTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_OP_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
}

cudnnStatus_t cudnnSetOpTensorDescriptor_posthook(
    cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt)
{
    trace_dump.dump("cudnnSetOpTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetOpTensorDescriptor_prehook(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetOpTensorDescriptor_proxy(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt)
{
    typedef decltype(&cudnnGetOpTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_OP_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetOpTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_OP_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
}

cudnnStatus_t cudnnGetOpTensorDescriptor_posthook(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt)
{
    trace_dump.dump("cudnnGetOpTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor_prehook(
    cudnnOpTensorDescriptor_t opTensorDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor_proxy(
    cudnnOpTensorDescriptor_t opTensorDesc)
{
    typedef decltype(&cudnnDestroyOpTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyOpTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        opTensorDesc);
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor_posthook(
    cudnnOpTensorDescriptor_t opTensorDesc)
{
    trace_dump.dump("cudnnDestroyOpTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor_prehook(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor_proxy(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc)
{
    typedef decltype(&cudnnCreateReduceTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateReduceTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        reduceTensorDesc);
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor_posthook(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc)
{
    trace_dump.dump("cudnnCreateReduceTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor_prehook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor_proxy(
    cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType)
{
    typedef decltype(&cudnnSetReduceTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetReduceTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
        reduceTensorIndices, reduceTensorIndicesType);
}

cudnnStatus_t cudnnSetReduceTensorDescriptor_posthook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType)
{
    trace_dump.dump("cudnnSetReduceTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor_prehook(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor_proxy(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType)
{
    typedef decltype(&cudnnGetReduceTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetReduceTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
        reduceTensorIndices, reduceTensorIndicesType);
}

cudnnStatus_t cudnnGetReduceTensorDescriptor_posthook(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType)
{
    trace_dump.dump("cudnnGetReduceTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor_prehook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor_proxy(
    cudnnReduceTensorDescriptor_t reduceTensorDesc)
{
    typedef decltype(&cudnnDestroyReduceTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyReduceTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        reduceTensorDesc);
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor_posthook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc)
{
    trace_dump.dump("cudnnDestroyReduceTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor_prehook(
    cudnnFilterDescriptor_t *filterDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor_proxy(
    cudnnFilterDescriptor_t *filterDesc)
{
    typedef decltype(&cudnnCreateFilterDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_FILTER_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateFilterDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_FILTER_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        filterDesc);
}

cudnnStatus_t cudnnCreateFilterDescriptor_posthook(
    cudnnFilterDescriptor_t *filterDesc)
{
    trace_dump.dump("cudnnCreateFilterDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor_prehook(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int k,
    int c, int h,
    int w)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor_proxy(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int k,
    int c, int h,
    int w)
{
    typedef decltype(&cudnnSetFilter4dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_FILTER_4D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetFilter4dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_FILTER_4D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        filterDesc, dataType, format, k,
        c, h, w);
}

cudnnStatus_t cudnnSetFilter4dDescriptor_posthook(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int k,
    int c, int h,
    int w)
{
    trace_dump.dump("cudnnSetFilter4dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilter4dDescriptor_prehook(
    const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *k,
    int *c, int *h,
    int *w)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilter4dDescriptor_proxy(
    const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *k,
    int *c, int *h,
    int *w)
{
    typedef decltype(&cudnnGetFilter4dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_FILTER_4D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetFilter4dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_FILTER_4D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        filterDesc, dataType, format, k,
        c, h, w);
}

cudnnStatus_t cudnnGetFilter4dDescriptor_posthook(
    const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *k,
    int *c, int *h,
    int *w)
{
    trace_dump.dump("cudnnGetFilter4dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor_prehook(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int nbDims,
    const int filterDimA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor_proxy(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int nbDims,
    const int filterDimA[])
{
    typedef decltype(&cudnnSetFilterNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_FILTER_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetFilterNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_FILTER_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        filterDesc, dataType, format, nbDims,
        filterDimA);
}

cudnnStatus_t cudnnSetFilterNdDescriptor_posthook(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int nbDims,
    const int filterDimA[])
{
    trace_dump.dump("cudnnSetFilterNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterNdDescriptor_prehook(
    const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, cudnnTensorFormat_t *format,
    int *nbDims, int filterDimA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterNdDescriptor_proxy(
    const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, cudnnTensorFormat_t *format,
    int *nbDims, int filterDimA[])
{
    typedef decltype(&cudnnGetFilterNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_FILTER_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetFilterNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_FILTER_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        filterDesc, nbDimsRequested, dataType, format,
        nbDims, filterDimA);
}

cudnnStatus_t cudnnGetFilterNdDescriptor_posthook(
    const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, cudnnTensorFormat_t *format,
    int *nbDims, int filterDimA[])
{
    trace_dump.dump("cudnnGetFilterNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterSizeInBytes_prehook(
    const cudnnFilterDescriptor_t filterDesc, size_t *size)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterSizeInBytes_proxy(
    const cudnnFilterDescriptor_t filterDesc, size_t *size)
{
    typedef decltype(&cudnnGetFilterSizeInBytes) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_FILTER_SIZE_IN_BYTES])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetFilterSizeInBytes));
        cudnn_hook_info.func_actual[CUDNN_GET_FILTER_SIZE_IN_BYTES] = actual_func;
    }
    return ((func_type)actual_func)(
        filterDesc, size);
}

cudnnStatus_t cudnnGetFilterSizeInBytes_posthook(
    const cudnnFilterDescriptor_t filterDesc, size_t *size)
{
    trace_dump.dump("cudnnGetFilterSizeInBytes");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor_prehook(
    cudnnFilterDescriptor_t filterDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor_proxy(
    cudnnFilterDescriptor_t filterDesc)
{
    typedef decltype(&cudnnDestroyFilterDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_FILTER_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyFilterDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_FILTER_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        filterDesc);
}

cudnnStatus_t cudnnDestroyFilterDescriptor_posthook(
    cudnnFilterDescriptor_t filterDesc)
{
    trace_dump.dump("cudnnDestroyFilterDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePoolingDescriptor_prehook(
    cudnnPoolingDescriptor_t *poolingDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePoolingDescriptor_proxy(
    cudnnPoolingDescriptor_t *poolingDesc)
{
    typedef decltype(&cudnnCreatePoolingDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_POOLING_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreatePoolingDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_POOLING_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc);
}

cudnnStatus_t cudnnCreatePoolingDescriptor_posthook(
    cudnnPoolingDescriptor_t *poolingDesc)
{
    trace_dump.dump("cudnnCreatePoolingDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor_prehook(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
    int windowWidth, int verticalPadding,
    int horizontalPadding, int verticalStride,
    int horizontalStride)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor_proxy(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
    int windowWidth, int verticalPadding,
    int horizontalPadding, int verticalStride,
    int horizontalStride)
{
    typedef decltype(&cudnnSetPooling2dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_POOLING_2D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetPooling2dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_POOLING_2D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc, mode, maxpoolingNanOpt, windowHeight,
        windowWidth, verticalPadding, horizontalPadding, verticalStride,
        horizontalStride);
}

cudnnStatus_t cudnnSetPooling2dDescriptor_posthook(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
    int windowWidth, int verticalPadding,
    int horizontalPadding, int verticalStride,
    int horizontalStride)
{
    trace_dump.dump("cudnnSetPooling2dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dDescriptor_prehook(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding,
    int *horizontalPadding, int *verticalStride,
    int *horizontalStride)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dDescriptor_proxy(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding,
    int *horizontalPadding, int *verticalStride,
    int *horizontalStride)
{
    typedef decltype(&cudnnGetPooling2dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_POOLING_2D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetPooling2dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_POOLING_2D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc, mode, maxpoolingNanOpt, windowHeight,
        windowWidth, verticalPadding, horizontalPadding, verticalStride,
        horizontalStride);
}

cudnnStatus_t cudnnGetPooling2dDescriptor_posthook(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding,
    int *horizontalPadding, int *verticalStride,
    int *horizontalStride)
{
    trace_dump.dump("cudnnGetPooling2dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_prehook(
    cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[],
    const int strideA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_proxy(
    cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[],
    const int strideA[])
{
    typedef decltype(&cudnnSetPoolingNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_POOLING_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetPoolingNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_POOLING_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc, mode, maxpoolingNanOpt, nbDims,
        windowDimA, paddingA, strideA);
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_posthook(
    cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[],
    const int strideA[])
{
    trace_dump.dump("cudnnSetPoolingNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor_prehook(
    const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[],
    int paddingA[], int strideA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor_proxy(
    const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[],
    int paddingA[], int strideA[])
{
    typedef decltype(&cudnnGetPoolingNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_POOLING_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetPoolingNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_POOLING_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt,
        nbDims, windowDimA, paddingA, strideA);
}

cudnnStatus_t cudnnGetPoolingNdDescriptor_posthook(
    const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[],
    int paddingA[], int strideA[])
{
    trace_dump.dump("cudnnGetPoolingNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim_prehook(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims, int outputTensorDimA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim_proxy(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims, int outputTensorDimA[])
{
    typedef decltype(&cudnnGetPoolingNdForwardOutputDim) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetPoolingNdForwardOutputDim));
        cudnn_hook_info.func_actual[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim_posthook(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims, int outputTensorDimA[])
{
    trace_dump.dump("cudnnGetPoolingNdForwardOutputDim");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim_prehook(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int *n, int *c,
    int *h, int *w)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim_proxy(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int *n, int *c,
    int *h, int *w)
{
    typedef decltype(&cudnnGetPooling2dForwardOutputDim) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetPooling2dForwardOutputDim));
        cudnn_hook_info.func_actual[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc, inputTensorDesc, n, c,
        h, w);
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim_posthook(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int *n, int *c,
    int *h, int *w)
{
    trace_dump.dump("cudnnGetPooling2dForwardOutputDim");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor_prehook(
    cudnnPoolingDescriptor_t poolingDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor_proxy(
    cudnnPoolingDescriptor_t poolingDesc)
{
    typedef decltype(&cudnnDestroyPoolingDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_POOLING_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyPoolingDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_POOLING_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        poolingDesc);
}

cudnnStatus_t cudnnDestroyPoolingDescriptor_posthook(
    cudnnPoolingDescriptor_t poolingDesc)
{
    trace_dump.dump("cudnnDestroyPoolingDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateActivationDescriptor_prehook(
    cudnnActivationDescriptor_t *activationDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateActivationDescriptor_proxy(
    cudnnActivationDescriptor_t *activationDesc)
{
    typedef decltype(&cudnnCreateActivationDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_ACTIVATION_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateActivationDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_ACTIVATION_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        activationDesc);
}

cudnnStatus_t cudnnCreateActivationDescriptor_posthook(
    cudnnActivationDescriptor_t *activationDesc)
{
    trace_dump.dump("cudnnCreateActivationDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor_prehook(
    cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt, double coef)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor_proxy(
    cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt, double coef)
{
    typedef decltype(&cudnnSetActivationDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_ACTIVATION_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetActivationDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_ACTIVATION_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        activationDesc, mode, reluNanOpt, coef);
}

cudnnStatus_t cudnnSetActivationDescriptor_posthook(
    cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt, double coef)
{
    trace_dump.dump("cudnnSetActivationDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptor_prehook(
    const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt, double *coef)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptor_proxy(
    const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt, double *coef)
{
    typedef decltype(&cudnnGetActivationDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_ACTIVATION_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetActivationDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_ACTIVATION_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        activationDesc, mode, reluNanOpt, coef);
}

cudnnStatus_t cudnnGetActivationDescriptor_posthook(
    const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt, double *coef)
{
    trace_dump.dump("cudnnGetActivationDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta_prehook(
    cudnnActivationDescriptor_t activationDesc, double swish_beta)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta_proxy(
    cudnnActivationDescriptor_t activationDesc, double swish_beta)
{
    typedef decltype(&cudnnSetActivationDescriptorSwishBeta) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetActivationDescriptorSwishBeta));
        cudnn_hook_info.func_actual[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA] = actual_func;
    }
    return ((func_type)actual_func)(
        activationDesc, swish_beta);
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta_posthook(
    cudnnActivationDescriptor_t activationDesc, double swish_beta)
{
    trace_dump.dump("cudnnSetActivationDescriptorSwishBeta");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta_prehook(
    cudnnActivationDescriptor_t activationDesc, double *swish_beta)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta_proxy(
    cudnnActivationDescriptor_t activationDesc, double *swish_beta)
{
    typedef decltype(&cudnnGetActivationDescriptorSwishBeta) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetActivationDescriptorSwishBeta));
        cudnn_hook_info.func_actual[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA] = actual_func;
    }
    return ((func_type)actual_func)(
        activationDesc, swish_beta);
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta_posthook(
    cudnnActivationDescriptor_t activationDesc, double *swish_beta)
{
    trace_dump.dump("cudnnGetActivationDescriptorSwishBeta");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor_prehook(
    cudnnActivationDescriptor_t activationDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor_proxy(
    cudnnActivationDescriptor_t activationDesc)
{
    typedef decltype(&cudnnDestroyActivationDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyActivationDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        activationDesc);
}

cudnnStatus_t cudnnDestroyActivationDescriptor_posthook(
    cudnnActivationDescriptor_t activationDesc)
{
    trace_dump.dump("cudnnDestroyActivationDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateLRNDescriptor_prehook(
    cudnnLRNDescriptor_t *normDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateLRNDescriptor_proxy(
    cudnnLRNDescriptor_t *normDesc)
{
    typedef decltype(&cudnnCreateLRNDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_LRN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateLRNDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_LRN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        normDesc);
}

cudnnStatus_t cudnnCreateLRNDescriptor_posthook(
    cudnnLRNDescriptor_t *normDesc)
{
    trace_dump.dump("cudnnCreateLRNDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetLRNDescriptor_prehook(
    cudnnLRNDescriptor_t normDesc, unsigned lrnN,
    double lrnAlpha, double lrnBeta,
    double lrnK)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetLRNDescriptor_proxy(
    cudnnLRNDescriptor_t normDesc, unsigned lrnN,
    double lrnAlpha, double lrnBeta,
    double lrnK)
{
    typedef decltype(&cudnnSetLRNDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_LRN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetLRNDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_LRN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        normDesc, lrnN, lrnAlpha, lrnBeta,
        lrnK);
}

cudnnStatus_t cudnnSetLRNDescriptor_posthook(
    cudnnLRNDescriptor_t normDesc, unsigned lrnN,
    double lrnAlpha, double lrnBeta,
    double lrnK)
{
    trace_dump.dump("cudnnSetLRNDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetLRNDescriptor_prehook(
    cudnnLRNDescriptor_t normDesc, unsigned *lrnN,
    double *lrnAlpha, double *lrnBeta,
    double *lrnK)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetLRNDescriptor_proxy(
    cudnnLRNDescriptor_t normDesc, unsigned *lrnN,
    double *lrnAlpha, double *lrnBeta,
    double *lrnK)
{
    typedef decltype(&cudnnGetLRNDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_LRN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetLRNDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_LRN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        normDesc, lrnN, lrnAlpha, lrnBeta,
        lrnK);
}

cudnnStatus_t cudnnGetLRNDescriptor_posthook(
    cudnnLRNDescriptor_t normDesc, unsigned *lrnN,
    double *lrnAlpha, double *lrnBeta,
    double *lrnK)
{
    trace_dump.dump("cudnnGetLRNDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyLRNDescriptor_prehook(
    cudnnLRNDescriptor_t lrnDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyLRNDescriptor_proxy(
    cudnnLRNDescriptor_t lrnDesc)
{
    typedef decltype(&cudnnDestroyLRNDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_LRN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyLRNDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_LRN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        lrnDesc);
}

cudnnStatus_t cudnnDestroyLRNDescriptor_posthook(
    cudnnLRNDescriptor_t lrnDesc)
{
    trace_dump.dump("cudnnDestroyLRNDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor_prehook(
    cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor_proxy(
    cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode)
{
    typedef decltype(&cudnnDeriveBNTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDeriveBNTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        derivedBnDesc, xDesc, mode);
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor_posthook(
    cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode)
{
    trace_dump.dump("cudnnDeriveBNTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor_prehook(
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc, cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc, cudnnNormMode_t mode,
    int groupCnt)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor_proxy(
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc, cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc, cudnnNormMode_t mode,
    int groupCnt)
{
    typedef decltype(&cudnnDeriveNormTensorDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDeriveNormTensorDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode,
        groupCnt);
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor_posthook(
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc, cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc, cudnnNormMode_t mode,
    int groupCnt)
{
    trace_dump.dump("cudnnDeriveNormTensorDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor_prehook(
    cudnnSpatialTransformerDescriptor_t *stDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor_proxy(
    cudnnSpatialTransformerDescriptor_t *stDesc)
{
    typedef decltype(&cudnnCreateSpatialTransformerDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateSpatialTransformerDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        stDesc);
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor_posthook(
    cudnnSpatialTransformerDescriptor_t *stDesc)
{
    trace_dump.dump("cudnnCreateSpatialTransformerDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor_prehook(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims,
    const int dimA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor_proxy(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims,
    const int dimA[])
{
    typedef decltype(&cudnnSetSpatialTransformerNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetSpatialTransformerNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        stDesc, samplerType, dataType, nbDims,
        dimA);
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor_posthook(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims,
    const int dimA[])
{
    trace_dump.dump("cudnnSetSpatialTransformerNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor_prehook(
    cudnnSpatialTransformerDescriptor_t stDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor_proxy(
    cudnnSpatialTransformerDescriptor_t stDesc)
{
    typedef decltype(&cudnnDestroySpatialTransformerDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroySpatialTransformerDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        stDesc);
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor_posthook(
    cudnnSpatialTransformerDescriptor_t stDesc)
{
    trace_dump.dump("cudnnDestroySpatialTransformerDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t *dropoutDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t *dropoutDesc)
{
    typedef decltype(&cudnnCreateDropoutDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_DROPOUT_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateDropoutDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_DROPOUT_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        dropoutDesc);
}

cudnnStatus_t cudnnCreateDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t *dropoutDesc)
{
    trace_dump.dump("cudnnCreateDropoutDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc)
{
    typedef decltype(&cudnnDestroyDropoutDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_DROPOUT_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyDropoutDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_DROPOUT_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        dropoutDesc);
}

cudnnStatus_t cudnnDestroyDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc)
{
    trace_dump.dump("cudnnDestroyDropoutDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize_prehook(
    cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize_proxy(
    cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes)
{
    typedef decltype(&cudnnDropoutGetReserveSpaceSize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDropoutGetReserveSpaceSize));
        cudnn_hook_info.func_actual[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        xdesc, sizeInBytes);
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize_posthook(
    cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes)
{
    trace_dump.dump("cudnnDropoutGetReserveSpaceSize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed)
{
    typedef decltype(&cudnnSetDropoutDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_DROPOUT_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetDropoutDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_DROPOUT_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        dropoutDesc, handle, dropout, states,
        stateSizeInBytes, seed);
}

cudnnStatus_t cudnnSetDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed)
{
    trace_dump.dump("cudnnSetDropoutDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed)
{
    typedef decltype(&cudnnRestoreDropoutDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RESTORE_DROPOUT_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRestoreDropoutDescriptor));
        cudnn_hook_info.func_actual[CUDNN_RESTORE_DROPOUT_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        dropoutDesc, handle, dropout, states,
        stateSizeInBytes, seed);
}

cudnnStatus_t cudnnRestoreDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed)
{
    trace_dump.dump("cudnnRestoreDropoutDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float *dropout, void **states,
    unsigned long long *seed)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float *dropout, void **states,
    unsigned long long *seed)
{
    typedef decltype(&cudnnGetDropoutDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_DROPOUT_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetDropoutDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_DROPOUT_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        dropoutDesc, handle, dropout, states,
        seed);
}

cudnnStatus_t cudnnGetDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float *dropout, void **states,
    unsigned long long *seed)
{
    trace_dump.dump("cudnnGetDropoutDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor_prehook(
    cudnnAlgorithmDescriptor_t *algoDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor_proxy(
    cudnnAlgorithmDescriptor_t *algoDesc)
{
    typedef decltype(&cudnnCreateAlgorithmDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_ALGORITHM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateAlgorithmDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_ALGORITHM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        algoDesc);
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor_posthook(
    cudnnAlgorithmDescriptor_t *algoDesc)
{
    trace_dump.dump("cudnnCreateAlgorithmDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor_prehook(
    cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor_proxy(
    cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm)
{
    typedef decltype(&cudnnSetAlgorithmDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_ALGORITHM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetAlgorithmDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_ALGORITHM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        algoDesc, algorithm);
}

cudnnStatus_t cudnnSetAlgorithmDescriptor_posthook(
    cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm)
{
    trace_dump.dump("cudnnSetAlgorithmDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmDescriptor_prehook(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmDescriptor_proxy(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm)
{
    typedef decltype(&cudnnGetAlgorithmDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_ALGORITHM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetAlgorithmDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_ALGORITHM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        algoDesc, algorithm);
}

cudnnStatus_t cudnnGetAlgorithmDescriptor_posthook(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm)
{
    trace_dump.dump("cudnnGetAlgorithmDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor_prehook(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor_proxy(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest)
{
    typedef decltype(&cudnnCopyAlgorithmDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_COPY_ALGORITHM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCopyAlgorithmDescriptor));
        cudnn_hook_info.func_actual[CUDNN_COPY_ALGORITHM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        src, dest);
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor_posthook(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest)
{
    trace_dump.dump("cudnnCopyAlgorithmDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor_prehook(
    cudnnAlgorithmDescriptor_t algoDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor_proxy(
    cudnnAlgorithmDescriptor_t algoDesc)
{
    typedef decltype(&cudnnDestroyAlgorithmDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyAlgorithmDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        algoDesc);
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor_posthook(
    cudnnAlgorithmDescriptor_t algoDesc)
{
    trace_dump.dump("cudnnDestroyAlgorithmDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmPerformance_prehook(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmPerformance_proxy(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate)
{
    typedef decltype(&cudnnCreateAlgorithmPerformance) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_ALGORITHM_PERFORMANCE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateAlgorithmPerformance));
        cudnn_hook_info.func_actual[CUDNN_CREATE_ALGORITHM_PERFORMANCE] = actual_func;
    }
    return ((func_type)actual_func)(
        algoPerf, numberToCreate);
}

cudnnStatus_t cudnnCreateAlgorithmPerformance_posthook(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate)
{
    trace_dump.dump("cudnnCreateAlgorithmPerformance");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmPerformance_prehook(
    cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status, float time,
    size_t memory)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmPerformance_proxy(
    cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status, float time,
    size_t memory)
{
    typedef decltype(&cudnnSetAlgorithmPerformance) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_ALGORITHM_PERFORMANCE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetAlgorithmPerformance));
        cudnn_hook_info.func_actual[CUDNN_SET_ALGORITHM_PERFORMANCE] = actual_func;
    }
    return ((func_type)actual_func)(
        algoPerf, algoDesc, status, time,
        memory);
}

cudnnStatus_t cudnnSetAlgorithmPerformance_posthook(
    cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status, float time,
    size_t memory)
{
    trace_dump.dump("cudnnSetAlgorithmPerformance");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmPerformance_prehook(
    const cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status, float *time,
    size_t *memory)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmPerformance_proxy(
    const cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status, float *time,
    size_t *memory)
{
    typedef decltype(&cudnnGetAlgorithmPerformance) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_ALGORITHM_PERFORMANCE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetAlgorithmPerformance));
        cudnn_hook_info.func_actual[CUDNN_GET_ALGORITHM_PERFORMANCE] = actual_func;
    }
    return ((func_type)actual_func)(
        algoPerf, algoDesc, status, time,
        memory);
}

cudnnStatus_t cudnnGetAlgorithmPerformance_posthook(
    const cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status, float *time,
    size_t *memory)
{
    trace_dump.dump("cudnnGetAlgorithmPerformance");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance_prehook(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance_proxy(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy)
{
    typedef decltype(&cudnnDestroyAlgorithmPerformance) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_ALGORITHM_PERFORMANCE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyAlgorithmPerformance));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_ALGORITHM_PERFORMANCE] = actual_func;
    }
    return ((func_type)actual_func)(
        algoPerf, numberToDestroy);
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance_posthook(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy)
{
    trace_dump.dump("cudnnDestroyAlgorithmPerformance");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCallback_prehook(
    unsigned mask, void *udata,
    cudnnCallback_t fptr)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCallback_proxy(
    unsigned mask, void *udata,
    cudnnCallback_t fptr)
{
    typedef decltype(&cudnnSetCallback) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CALLBACK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetCallback));
        cudnn_hook_info.func_actual[CUDNN_SET_CALLBACK] = actual_func;
    }
    return ((func_type)actual_func)(
        mask, udata, fptr);
}

cudnnStatus_t cudnnSetCallback_posthook(
    unsigned mask, void *udata,
    cudnnCallback_t fptr)
{
    trace_dump.dump("cudnnSetCallback");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCallback_prehook(
    unsigned *mask, void **udata,
    cudnnCallback_t *fptr)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCallback_proxy(
    unsigned *mask, void **udata,
    cudnnCallback_t *fptr)
{
    typedef decltype(&cudnnGetCallback) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CALLBACK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetCallback));
        cudnn_hook_info.func_actual[CUDNN_GET_CALLBACK] = actual_func;
    }
    return ((func_type)actual_func)(
        mask, udata, fptr);
}

cudnnStatus_t cudnnGetCallback_posthook(
    unsigned *mask, void **udata,
    cudnnCallback_t *fptr)
{
    trace_dump.dump("cudnnGetCallback");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsInferVersionCheck_prehook(
)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsInferVersionCheck_proxy(
)
{
    typedef decltype(&cudnnOpsInferVersionCheck) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_OPS_INFER_VERSION_CHECK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnOpsInferVersionCheck));
        cudnn_hook_info.func_actual[CUDNN_OPS_INFER_VERSION_CHECK] = actual_func;
    }
    return ((func_type)actual_func)(
);
}

cudnnStatus_t cudnnOpsInferVersionCheck_posthook(
)
{
    trace_dump.dump("cudnnOpsInferVersionCheck");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsTrainVersionCheck_prehook(
)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsTrainVersionCheck_proxy(
)
{
    typedef decltype(&cudnnOpsTrainVersionCheck) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_OPS_TRAIN_VERSION_CHECK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnOpsTrainVersionCheck));
        cudnn_hook_info.func_actual[CUDNN_OPS_TRAIN_VERSION_CHECK] = actual_func;
    }
    return ((func_type)actual_func)(
);
}

cudnnStatus_t cudnnOpsTrainVersionCheck_posthook(
)
{
    trace_dump.dump("cudnnOpsTrainVersionCheck");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDescriptor_prehook(
    cudnnRNNDescriptor_t *rnnDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDescriptor_proxy(
    cudnnRNNDescriptor_t *rnnDesc)
{
    typedef decltype(&cudnnCreateRNNDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_RNN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateRNNDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_RNN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc);
}

cudnnStatus_t cudnnCreateRNNDescriptor_posthook(
    cudnnRNNDescriptor_t *rnnDesc)
{
    trace_dump.dump("cudnnCreateRNNDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDescriptor_prehook(
    cudnnRNNDescriptor_t rnnDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDescriptor_proxy(
    cudnnRNNDescriptor_t rnnDesc)
{
    typedef decltype(&cudnnDestroyRNNDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_RNN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyRNNDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_RNN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc);
}

cudnnStatus_t cudnnDestroyRNNDescriptor_posthook(
    cudnnRNNDescriptor_t rnnDesc)
{
    trace_dump.dump("cudnnDestroyRNNDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType, cudnnDataType_t mathPrec,
    cudnnMathType_t mathType, int32_t inputSize,
    int32_t hiddenSize, int32_t projSize,
    int32_t numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType, cudnnDataType_t mathPrec,
    cudnnMathType_t mathType, int32_t inputSize,
    int32_t hiddenSize, int32_t projSize,
    int32_t numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags)
{
    typedef decltype(&cudnnSetRNNDescriptor_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_DESCRIPTOR_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNDescriptor_v8));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_DESCRIPTOR_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, algo, cellMode, biasMode,
        dirMode, inputMode, dataType, mathPrec,
        mathType, inputSize, hiddenSize, projSize,
        numLayers, dropoutDesc, auxFlags);
}

cudnnStatus_t cudnnSetRNNDescriptor_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType, cudnnDataType_t mathPrec,
    cudnnMathType_t mathType, int32_t inputSize,
    int32_t hiddenSize, int32_t projSize,
    int32_t numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags)
{
    trace_dump.dump("cudnnSetRNNDescriptor_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode, cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode, cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType, cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType, int32_t *inputSize,
    int32_t *hiddenSize, int32_t *projSize,
    int32_t *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode, cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode, cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType, cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType, int32_t *inputSize,
    int32_t *hiddenSize, int32_t *projSize,
    int32_t *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags)
{
    typedef decltype(&cudnnGetRNNDescriptor_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_DESCRIPTOR_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNDescriptor_v8));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_DESCRIPTOR_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, algo, cellMode, biasMode,
        dirMode, inputMode, dataType, mathPrec,
        mathType, inputSize, hiddenSize, projSize,
        numLayers, dropoutDesc, auxFlags);
}

cudnnStatus_t cudnnGetRNNDescriptor_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode, cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode, cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType, cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType, int32_t *inputSize,
    int32_t *hiddenSize, int32_t *projSize,
    int32_t *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags)
{
    trace_dump.dump("cudnnGetRNNDescriptor_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNMatrixMathType_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNMatrixMathType_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType)
{
    typedef decltype(&cudnnSetRNNMatrixMathType) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_MATRIX_MATH_TYPE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNMatrixMathType));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_MATRIX_MATH_TYPE] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, mType);
}

cudnnStatus_t cudnnSetRNNMatrixMathType_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType)
{
    trace_dump.dump("cudnnSetRNNMatrixMathType");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNMatrixMathType_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNMatrixMathType_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType)
{
    typedef decltype(&cudnnGetRNNMatrixMathType) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_MATRIX_MATH_TYPE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNMatrixMathType));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_MATRIX_MATH_TYPE] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, mType);
}

cudnnStatus_t cudnnGetRNNMatrixMathType_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType)
{
    trace_dump.dump("cudnnGetRNNMatrixMathType");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNBiasMode_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNBiasMode_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode)
{
    typedef decltype(&cudnnSetRNNBiasMode) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_BIAS_MODE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNBiasMode));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_BIAS_MODE] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, biasMode);
}

cudnnStatus_t cudnnSetRNNBiasMode_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode)
{
    trace_dump.dump("cudnnSetRNNBiasMode");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBiasMode_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBiasMode_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode)
{
    typedef decltype(&cudnnGetRNNBiasMode) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_BIAS_MODE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNBiasMode));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_BIAS_MODE] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, biasMode);
}

cudnnStatus_t cudnnGetRNNBiasMode_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode)
{
    trace_dump.dump("cudnnGetRNNBiasMode");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt, double lclip,
    double rclip)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt, double lclip,
    double rclip)
{
    typedef decltype(&cudnnRNNSetClip_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_SET_CLIP_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNSetClip_v8));
        cudnn_hook_info.func_actual[CUDNN_RNN_SET_CLIP_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, clipMode, clipNanOpt, lclip,
        rclip);
}

cudnnStatus_t cudnnRNNSetClip_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt, double lclip,
    double rclip)
{
    trace_dump.dump("cudnnRNNSetClip_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt, double *lclip,
    double *rclip)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt, double *lclip,
    double *rclip)
{
    typedef decltype(&cudnnRNNGetClip_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_RNN_GET_CLIP_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnRNNGetClip_v8));
        cudnn_hook_info.func_actual[CUDNN_RNN_GET_CLIP_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, clipMode, clipNanOpt, lclip,
        rclip);
}

cudnnStatus_t cudnnRNNGetClip_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt, double *lclip,
    double *rclip)
{
    trace_dump.dump("cudnnRNNGetClip_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan_prehook(
    cudnnRNNDescriptor_t rnnDesc, const int minibatch,
    const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan_proxy(
    cudnnRNNDescriptor_t rnnDesc, const int minibatch,
    const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan)
{
    typedef decltype(&cudnnCreatePersistentRNNPlan) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_PERSISTENT_RNN_PLAN])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreatePersistentRNNPlan));
        cudnn_hook_info.func_actual[CUDNN_CREATE_PERSISTENT_RNN_PLAN] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, minibatch, dataType, plan);
}

cudnnStatus_t cudnnCreatePersistentRNNPlan_posthook(
    cudnnRNNDescriptor_t rnnDesc, const int minibatch,
    const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan)
{
    trace_dump.dump("cudnnCreatePersistentRNNPlan");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan_prehook(
    cudnnPersistentRNNPlan_t plan)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan_proxy(
    cudnnPersistentRNNPlan_t plan)
{
    typedef decltype(&cudnnDestroyPersistentRNNPlan) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_PERSISTENT_RNN_PLAN])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyPersistentRNNPlan));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_PERSISTENT_RNN_PLAN] = actual_func;
    }
    return ((func_type)actual_func)(
        plan);
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan_posthook(
    cudnnPersistentRNNPlan_t plan)
{
    trace_dump.dump("cudnnDestroyPersistentRNNPlan");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPersistentRNNPlan_prehook(
    cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPersistentRNNPlan_proxy(
    cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan)
{
    typedef decltype(&cudnnSetPersistentRNNPlan) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_PERSISTENT_RNN_PLAN])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetPersistentRNNPlan));
        cudnn_hook_info.func_actual[CUDNN_SET_PERSISTENT_RNN_PLAN] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, plan);
}

cudnnStatus_t cudnnSetPersistentRNNPlan_posthook(
    cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan)
{
    trace_dump.dump("cudnnSetPersistentRNNPlan");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNPaddingMode_prehook(
    cudnnRNNDescriptor_t rnnDesc, unsigned paddingMode)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNPaddingMode_proxy(
    cudnnRNNDescriptor_t rnnDesc, unsigned paddingMode)
{
    typedef decltype(&cudnnSetRNNPaddingMode) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_PADDING_MODE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNPaddingMode));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_PADDING_MODE] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, paddingMode);
}

cudnnStatus_t cudnnSetRNNPaddingMode_posthook(
    cudnnRNNDescriptor_t rnnDesc, unsigned paddingMode)
{
    trace_dump.dump("cudnnSetRNNPaddingMode");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNPaddingMode_prehook(
    cudnnRNNDescriptor_t rnnDesc, unsigned *paddingMode)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNPaddingMode_proxy(
    cudnnRNNDescriptor_t rnnDesc, unsigned *paddingMode)
{
    typedef decltype(&cudnnGetRNNPaddingMode) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_PADDING_MODE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNPaddingMode));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_PADDING_MODE] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDesc, paddingMode);
}

cudnnStatus_t cudnnGetRNNPaddingMode_posthook(
    cudnnRNNDescriptor_t rnnDesc, unsigned *paddingMode)
{
    trace_dump.dump("cudnnGetRNNPaddingMode");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t *rnnDataDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t *rnnDataDesc)
{
    typedef decltype(&cudnnCreateRNNDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_RNN_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateRNNDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_RNN_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDataDesc);
}

cudnnStatus_t cudnnCreateRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t *rnnDataDesc)
{
    trace_dump.dump("cudnnCreateRNNDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t rnnDataDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t rnnDataDesc)
{
    typedef decltype(&cudnnDestroyRNNDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyRNNDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDataDesc);
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t rnnDataDesc)
{
    trace_dump.dump("cudnnDestroyRNNDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout, int maxSeqLength,
    int batchSize, int vectorSize,
    const int seqLengthArray[], void *paddingFill)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout, int maxSeqLength,
    int batchSize, int vectorSize,
    const int seqLengthArray[], void *paddingFill)
{
    typedef decltype(&cudnnSetRNNDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_RNN_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetRNNDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_RNN_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDataDesc, dataType, layout, maxSeqLength,
        batchSize, vectorSize, seqLengthArray, paddingFill);
}

cudnnStatus_t cudnnSetRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout, int maxSeqLength,
    int batchSize, int vectorSize,
    const int seqLengthArray[], void *paddingFill)
{
    trace_dump.dump("cudnnSetRNNDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout, int *maxSeqLength,
    int *batchSize, int *vectorSize,
    int arrayLengthRequested, int seqLengthArray[],
    void *paddingFill)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout, int *maxSeqLength,
    int *batchSize, int *vectorSize,
    int arrayLengthRequested, int seqLengthArray[],
    void *paddingFill)
{
    typedef decltype(&cudnnGetRNNDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_RNN_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetRNNDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_RNN_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        rnnDataDesc, dataType, layout, maxSeqLength,
        batchSize, vectorSize, arrayLengthRequested, seqLengthArray,
        paddingFill);
}

cudnnStatus_t cudnnGetRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout, int *maxSeqLength,
    int *batchSize, int *vectorSize,
    int arrayLengthRequested, int seqLengthArray[],
    void *paddingFill)
{
    trace_dump.dump("cudnnGetRNNDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSeqDataDescriptor_prehook(
    cudnnSeqDataDescriptor_t *seqDataDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSeqDataDescriptor_proxy(
    cudnnSeqDataDescriptor_t *seqDataDesc)
{
    typedef decltype(&cudnnCreateSeqDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateSeqDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        seqDataDesc);
}

cudnnStatus_t cudnnCreateSeqDataDescriptor_posthook(
    cudnnSeqDataDescriptor_t *seqDataDesc)
{
    trace_dump.dump("cudnnCreateSeqDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor_prehook(
    cudnnSeqDataDescriptor_t seqDataDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor_proxy(
    cudnnSeqDataDescriptor_t seqDataDesc)
{
    typedef decltype(&cudnnDestroySeqDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroySeqDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        seqDataDesc);
}

cudnnStatus_t cudnnDestroySeqDataDescriptor_posthook(
    cudnnSeqDataDescriptor_t seqDataDesc)
{
    trace_dump.dump("cudnnDestroySeqDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSeqDataDescriptor_prehook(
    cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const cudnnSeqDataAxis_t axes[], size_t seqLengthArraySize,
    const int seqLengthArray[], void *paddingFill)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSeqDataDescriptor_proxy(
    cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const cudnnSeqDataAxis_t axes[], size_t seqLengthArraySize,
    const int seqLengthArray[], void *paddingFill)
{
    typedef decltype(&cudnnSetSeqDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_SEQ_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetSeqDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_SEQ_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        seqDataDesc, dataType, nbDims, dimA,
        axes, seqLengthArraySize, seqLengthArray, paddingFill);
}

cudnnStatus_t cudnnSetSeqDataDescriptor_posthook(
    cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const cudnnSeqDataAxis_t axes[], size_t seqLengthArraySize,
    const int seqLengthArray[], void *paddingFill)
{
    trace_dump.dump("cudnnSetSeqDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetSeqDataDescriptor_prehook(
    const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType,
    int *nbDims, int nbDimsRequested,
    int dimA[], cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize, size_t seqLengthSizeRequested,
    int seqLengthArray[], void *paddingFill)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetSeqDataDescriptor_proxy(
    const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType,
    int *nbDims, int nbDimsRequested,
    int dimA[], cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize, size_t seqLengthSizeRequested,
    int seqLengthArray[], void *paddingFill)
{
    typedef decltype(&cudnnGetSeqDataDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_SEQ_DATA_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetSeqDataDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_SEQ_DATA_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        seqDataDesc, dataType, nbDims, nbDimsRequested,
        dimA, axes, seqLengthArraySize, seqLengthSizeRequested,
        seqLengthArray, paddingFill);
}

cudnnStatus_t cudnnGetSeqDataDescriptor_posthook(
    const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType,
    int *nbDims, int nbDimsRequested,
    int dimA[], cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize, size_t seqLengthSizeRequested,
    int seqLengthArray[], void *paddingFill)
{
    trace_dump.dump("cudnnGetSeqDataDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAttnDescriptor_prehook(
    cudnnAttnDescriptor_t *attnDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAttnDescriptor_proxy(
    cudnnAttnDescriptor_t *attnDesc)
{
    typedef decltype(&cudnnCreateAttnDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_ATTN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateAttnDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_ATTN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        attnDesc);
}

cudnnStatus_t cudnnCreateAttnDescriptor_posthook(
    cudnnAttnDescriptor_t *attnDesc)
{
    trace_dump.dump("cudnnCreateAttnDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAttnDescriptor_prehook(
    cudnnAttnDescriptor_t attnDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAttnDescriptor_proxy(
    cudnnAttnDescriptor_t attnDesc)
{
    typedef decltype(&cudnnDestroyAttnDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_ATTN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyAttnDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_ATTN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        attnDesc);
}

cudnnStatus_t cudnnDestroyAttnDescriptor_posthook(
    cudnnAttnDescriptor_t attnDesc)
{
    trace_dump.dump("cudnnDestroyAttnDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAttnDescriptor_prehook(
    cudnnAttnDescriptor_t attnDesc, unsigned attnMode,
    int nHeads, double smScaler,
    cudnnDataType_t dataType, cudnnDataType_t computePrec,
    cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize,
    int kSize, int vSize,
    int qProjSize, int kProjSize,
    int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength,
    int maxBatchSize, int maxBeamSize)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAttnDescriptor_proxy(
    cudnnAttnDescriptor_t attnDesc, unsigned attnMode,
    int nHeads, double smScaler,
    cudnnDataType_t dataType, cudnnDataType_t computePrec,
    cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize,
    int kSize, int vSize,
    int qProjSize, int kProjSize,
    int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength,
    int maxBatchSize, int maxBeamSize)
{
    typedef decltype(&cudnnSetAttnDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_ATTN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetAttnDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_ATTN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        attnDesc, attnMode, nHeads, smScaler,
        dataType, computePrec, mathType, attnDropoutDesc,
        postDropoutDesc, qSize, kSize, vSize,
        qProjSize, kProjSize, vProjSize, oProjSize,
        qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
}

cudnnStatus_t cudnnSetAttnDescriptor_posthook(
    cudnnAttnDescriptor_t attnDesc, unsigned attnMode,
    int nHeads, double smScaler,
    cudnnDataType_t dataType, cudnnDataType_t computePrec,
    cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize,
    int kSize, int vSize,
    int qProjSize, int kProjSize,
    int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength,
    int maxBatchSize, int maxBeamSize)
{
    trace_dump.dump("cudnnSetAttnDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAttnDescriptor_prehook(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode,
    int *nHeads, double *smScaler,
    cudnnDataType_t *dataType, cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize,
    int *kSize, int *vSize,
    int *qProjSize, int *kProjSize,
    int *vProjSize, int *oProjSize,
    int *qoMaxSeqLength, int *kvMaxSeqLength,
    int *maxBatchSize, int *maxBeamSize)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAttnDescriptor_proxy(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode,
    int *nHeads, double *smScaler,
    cudnnDataType_t *dataType, cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize,
    int *kSize, int *vSize,
    int *qProjSize, int *kProjSize,
    int *vProjSize, int *oProjSize,
    int *qoMaxSeqLength, int *kvMaxSeqLength,
    int *maxBatchSize, int *maxBeamSize)
{
    typedef decltype(&cudnnGetAttnDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_ATTN_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetAttnDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_ATTN_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        attnDesc, attnMode, nHeads, smScaler,
        dataType, computePrec, mathType, attnDropoutDesc,
        postDropoutDesc, qSize, kSize, vSize,
        qProjSize, kProjSize, vProjSize, oProjSize,
        qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
}

cudnnStatus_t cudnnGetAttnDescriptor_posthook(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode,
    int *nHeads, double *smScaler,
    cudnnDataType_t *dataType, cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize,
    int *kSize, int *vSize,
    int *qProjSize, int *kProjSize,
    int *vProjSize, int *oProjSize,
    int *qoMaxSeqLength, int *kvMaxSeqLength,
    int *maxBatchSize, int *maxBeamSize)
{
    trace_dump.dump("cudnnGetAttnDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvInferVersionCheck_prehook(
)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvInferVersionCheck_proxy(
)
{
    typedef decltype(&cudnnAdvInferVersionCheck) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_ADV_INFER_VERSION_CHECK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnAdvInferVersionCheck));
        cudnn_hook_info.func_actual[CUDNN_ADV_INFER_VERSION_CHECK] = actual_func;
    }
    return ((func_type)actual_func)(
);
}

cudnnStatus_t cudnnAdvInferVersionCheck_posthook(
)
{
    trace_dump.dump("cudnnAdvInferVersionCheck");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t *ctcLossDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t *ctcLossDesc)
{
    typedef decltype(&cudnnCreateCTCLossDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateCTCLossDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc);
}

cudnnStatus_t cudnnCreateCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t *ctcLossDesc)
{
    trace_dump.dump("cudnnCreateCTCLossDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType)
{
    typedef decltype(&cudnnSetCTCLossDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CTC_LOSS_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetCTCLossDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_CTC_LOSS_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc, compType);
}

cudnnStatus_t cudnnSetCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType)
{
    trace_dump.dump("cudnnSetCTCLossDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode)
{
    typedef decltype(&cudnnSetCTCLossDescriptorEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetCTCLossDescriptorEx));
        cudnn_hook_info.func_actual[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc, compType, normMode, gradMode);
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode)
{
    trace_dump.dump("cudnnSetCTCLossDescriptorEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode,
    int maxLabelLength)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode,
    int maxLabelLength)
{
    typedef decltype(&cudnnSetCTCLossDescriptor_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetCTCLossDescriptor_v8));
        cudnn_hook_info.func_actual[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc, compType, normMode, gradMode,
        maxLabelLength);
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode,
    int maxLabelLength)
{
    trace_dump.dump("cudnnSetCTCLossDescriptor_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType)
{
    typedef decltype(&cudnnGetCTCLossDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetCTCLossDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc, compType);
}

cudnnStatus_t cudnnGetCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType)
{
    trace_dump.dump("cudnnGetCTCLossDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode)
{
    typedef decltype(&cudnnGetCTCLossDescriptorEx) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetCTCLossDescriptorEx));
        cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc, compType, normMode, gradMode);
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode)
{
    trace_dump.dump("cudnnGetCTCLossDescriptorEx");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode,
    int *maxLabelLength)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode,
    int *maxLabelLength)
{
    typedef decltype(&cudnnGetCTCLossDescriptor_v8) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetCTCLossDescriptor_v8));
        cudnn_hook_info.func_actual[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc, compType, normMode, gradMode,
        maxLabelLength);
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode,
    int *maxLabelLength)
{
    trace_dump.dump("cudnnGetCTCLossDescriptor_v8");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc)
{
    typedef decltype(&cudnnDestroyCTCLossDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyCTCLossDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        ctcLossDesc);
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc)
{
    trace_dump.dump("cudnnDestroyCTCLossDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvTrainVersionCheck_prehook(
)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvTrainVersionCheck_proxy(
)
{
    typedef decltype(&cudnnAdvTrainVersionCheck) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_ADV_TRAIN_VERSION_CHECK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnAdvTrainVersionCheck));
        cudnn_hook_info.func_actual[CUDNN_ADV_TRAIN_VERSION_CHECK] = actual_func;
    }
    return ((func_type)actual_func)(
);
}

cudnnStatus_t cudnnAdvTrainVersionCheck_posthook(
)
{
    trace_dump.dump("cudnnAdvTrainVersionCheck");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor_prehook(
    cudnnConvolutionDescriptor_t *convDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor_proxy(
    cudnnConvolutionDescriptor_t *convDesc)
{
    typedef decltype(&cudnnCreateConvolutionDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateConvolutionDescriptor));
        cudnn_hook_info.func_actual[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc);
}

cudnnStatus_t cudnnCreateConvolutionDescriptor_posthook(
    cudnnConvolutionDescriptor_t *convDesc)
{
    trace_dump.dump("cudnnCreateConvolutionDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor_prehook(
    cudnnConvolutionDescriptor_t convDesc)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor_proxy(
    cudnnConvolutionDescriptor_t convDesc)
{
    typedef decltype(&cudnnDestroyConvolutionDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyConvolutionDescriptor));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc);
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor_posthook(
    cudnnConvolutionDescriptor_t convDesc)
{
    trace_dump.dump("cudnnDestroyConvolutionDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType_prehook(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType_proxy(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType)
{
    typedef decltype(&cudnnSetConvolutionMathType) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_MATH_TYPE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetConvolutionMathType));
        cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_MATH_TYPE] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, mathType);
}

cudnnStatus_t cudnnSetConvolutionMathType_posthook(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType)
{
    trace_dump.dump("cudnnSetConvolutionMathType");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType_prehook(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType_proxy(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType)
{
    typedef decltype(&cudnnGetConvolutionMathType) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_MATH_TYPE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionMathType));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_MATH_TYPE] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, mathType);
}

cudnnStatus_t cudnnGetConvolutionMathType_posthook(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType)
{
    trace_dump.dump("cudnnGetConvolutionMathType");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount_prehook(
    cudnnConvolutionDescriptor_t convDesc, int groupCount)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount_proxy(
    cudnnConvolutionDescriptor_t convDesc, int groupCount)
{
    typedef decltype(&cudnnSetConvolutionGroupCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_GROUP_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetConvolutionGroupCount));
        cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_GROUP_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, groupCount);
}

cudnnStatus_t cudnnSetConvolutionGroupCount_posthook(
    cudnnConvolutionDescriptor_t convDesc, int groupCount)
{
    trace_dump.dump("cudnnSetConvolutionGroupCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount_prehook(
    cudnnConvolutionDescriptor_t convDesc, int *groupCount)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount_proxy(
    cudnnConvolutionDescriptor_t convDesc, int *groupCount)
{
    typedef decltype(&cudnnGetConvolutionGroupCount) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_GROUP_COUNT])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionGroupCount));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_GROUP_COUNT] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, groupCount);
}

cudnnStatus_t cudnnGetConvolutionGroupCount_posthook(
    cudnnConvolutionDescriptor_t convDesc, int *groupCount)
{
    trace_dump.dump("cudnnGetConvolutionGroupCount");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionReorderType_prehook(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionReorderType_proxy(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType)
{
    typedef decltype(&cudnnSetConvolutionReorderType) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_REORDER_TYPE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetConvolutionReorderType));
        cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_REORDER_TYPE] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, reorderType);
}

cudnnStatus_t cudnnSetConvolutionReorderType_posthook(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType)
{
    trace_dump.dump("cudnnSetConvolutionReorderType");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionReorderType_prehook(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionReorderType_proxy(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType)
{
    typedef decltype(&cudnnGetConvolutionReorderType) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_REORDER_TYPE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionReorderType));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_REORDER_TYPE] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, reorderType);
}

cudnnStatus_t cudnnGetConvolutionReorderType_posthook(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType)
{
    trace_dump.dump("cudnnGetConvolutionReorderType");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor_prehook(
    cudnnConvolutionDescriptor_t convDesc, int pad_h,
    int pad_w, int u,
    int v, int dilation_h,
    int dilation_w, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor_proxy(
    cudnnConvolutionDescriptor_t convDesc, int pad_h,
    int pad_w, int u,
    int v, int dilation_h,
    int dilation_w, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType)
{
    typedef decltype(&cudnnSetConvolution2dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetConvolution2dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, pad_h, pad_w, u,
        v, dilation_h, dilation_w, mode,
        computeType);
}

cudnnStatus_t cudnnSetConvolution2dDescriptor_posthook(
    cudnnConvolutionDescriptor_t convDesc, int pad_h,
    int pad_w, int u,
    int v, int dilation_h,
    int dilation_w, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType)
{
    trace_dump.dump("cudnnSetConvolution2dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor_prehook(
    const cudnnConvolutionDescriptor_t convDesc, int *pad_h,
    int *pad_w, int *u,
    int *v, int *dilation_h,
    int *dilation_w, cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor_proxy(
    const cudnnConvolutionDescriptor_t convDesc, int *pad_h,
    int *pad_w, int *u,
    int *v, int *dilation_h,
    int *dilation_w, cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType)
{
    typedef decltype(&cudnnGetConvolution2dDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolution2dDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, pad_h, pad_w, u,
        v, dilation_h, dilation_w, mode,
        computeType);
}

cudnnStatus_t cudnnGetConvolution2dDescriptor_posthook(
    const cudnnConvolutionDescriptor_t convDesc, int *pad_h,
    int *pad_w, int *u,
    int *v, int *dilation_h,
    int *dilation_w, cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType)
{
    trace_dump.dump("cudnnGetConvolution2dDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_prehook(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength,
    const int padA[], const int filterStrideA[],
    const int dilationA[], cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_proxy(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength,
    const int padA[], const int filterStrideA[],
    const int dilationA[], cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType)
{
    typedef decltype(&cudnnSetConvolutionNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetConvolutionNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, arrayLength, padA, filterStrideA,
        dilationA, mode, computeType);
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_posthook(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength,
    const int padA[], const int filterStrideA[],
    const int dilationA[], cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType)
{
    trace_dump.dump("cudnnSetConvolutionNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_prehook(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[],
    int strideA[], int dilationA[],
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_proxy(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[],
    int strideA[], int dilationA[],
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType)
{
    typedef decltype(&cudnnGetConvolutionNdDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionNdDescriptor));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, arrayLengthRequested, arrayLength, padA,
        strideA, dilationA, mode, computeType);
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_posthook(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[],
    int strideA[], int dilationA[],
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType)
{
    trace_dump.dump("cudnnGetConvolutionNdDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim_prehook(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n,
    int *c, int *h,
    int *w)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim_proxy(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n,
    int *c, int *h,
    int *w)
{
    typedef decltype(&cudnnGetConvolution2dForwardOutputDim) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolution2dForwardOutputDim));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, inputTensorDesc, filterDesc, n,
        c, h, w);
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim_posthook(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n,
    int *c, int *h,
    int *w)
{
    trace_dump.dump("cudnnGetConvolution2dForwardOutputDim");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim_prehook(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[])
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim_proxy(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[])
{
    typedef decltype(&cudnnGetConvolutionNdForwardOutputDim) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetConvolutionNdForwardOutputDim));
        cudnn_hook_info.func_actual[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM] = actual_func;
    }
    return ((func_type)actual_func)(
        convDesc, inputTensorDesc, filterDesc, nbDims,
        tensorOuputDimA);
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim_posthook(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[])
{
    trace_dump.dump("cudnnGetConvolutionNdForwardOutputDim");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnInferVersionCheck_prehook(
)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnInferVersionCheck_proxy(
)
{
    typedef decltype(&cudnnCnnInferVersionCheck) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CNN_INFER_VERSION_CHECK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCnnInferVersionCheck));
        cudnn_hook_info.func_actual[CUDNN_CNN_INFER_VERSION_CHECK] = actual_func;
    }
    return ((func_type)actual_func)(
);
}

cudnnStatus_t cudnnCnnInferVersionCheck_posthook(
)
{
    trace_dump.dump("cudnnCnnInferVersionCheck");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack_prehook(
    cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack_proxy(
    cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops)
{
    typedef decltype(&cudnnCreateFusedOpsConstParamPack) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateFusedOpsConstParamPack));
        cudnn_hook_info.func_actual[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK] = actual_func;
    }
    return ((func_type)actual_func)(
        constPack, ops);
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack_posthook(
    cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops)
{
    trace_dump.dump("cudnnCreateFusedOpsConstParamPack");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack_prehook(
    cudnnFusedOpsConstParamPack_t constPack)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack_proxy(
    cudnnFusedOpsConstParamPack_t constPack)
{
    typedef decltype(&cudnnDestroyFusedOpsConstParamPack) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyFusedOpsConstParamPack));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK] = actual_func;
    }
    return ((func_type)actual_func)(
        constPack);
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack_posthook(
    cudnnFusedOpsConstParamPack_t constPack)
{
    trace_dump.dump("cudnnDestroyFusedOpsConstParamPack");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute_prehook(
    cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute_proxy(
    cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param)
{
    typedef decltype(&cudnnSetFusedOpsConstParamPackAttribute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetFusedOpsConstParamPackAttribute));
        cudnn_hook_info.func_actual[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        constPack, paramLabel, param);
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute_posthook(
    cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param)
{
    trace_dump.dump("cudnnSetFusedOpsConstParamPackAttribute");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute_prehook(
    const cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param, int *isNULL)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute_proxy(
    const cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param, int *isNULL)
{
    typedef decltype(&cudnnGetFusedOpsConstParamPackAttribute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetFusedOpsConstParamPackAttribute));
        cudnn_hook_info.func_actual[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        constPack, paramLabel, param, isNULL);
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute_posthook(
    const cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param, int *isNULL)
{
    trace_dump.dump("cudnnGetFusedOpsConstParamPackAttribute");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack_prehook(
    cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack_proxy(
    cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops)
{
    typedef decltype(&cudnnCreateFusedOpsVariantParamPack) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateFusedOpsVariantParamPack));
        cudnn_hook_info.func_actual[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK] = actual_func;
    }
    return ((func_type)actual_func)(
        varPack, ops);
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack_posthook(
    cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops)
{
    trace_dump.dump("cudnnCreateFusedOpsVariantParamPack");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack_prehook(
    cudnnFusedOpsVariantParamPack_t varPack)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack_proxy(
    cudnnFusedOpsVariantParamPack_t varPack)
{
    typedef decltype(&cudnnDestroyFusedOpsVariantParamPack) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyFusedOpsVariantParamPack));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK] = actual_func;
    }
    return ((func_type)actual_func)(
        varPack);
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack_posthook(
    cudnnFusedOpsVariantParamPack_t varPack)
{
    trace_dump.dump("cudnnDestroyFusedOpsVariantParamPack");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute_prehook(
    cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute_proxy(
    cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr)
{
    typedef decltype(&cudnnSetFusedOpsVariantParamPackAttribute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnSetFusedOpsVariantParamPackAttribute));
        cudnn_hook_info.func_actual[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        varPack, paramLabel, ptr);
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute_posthook(
    cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr)
{
    trace_dump.dump("cudnnSetFusedOpsVariantParamPackAttribute");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute_prehook(
    const cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute_proxy(
    const cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr)
{
    typedef decltype(&cudnnGetFusedOpsVariantParamPackAttribute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnGetFusedOpsVariantParamPackAttribute));
        cudnn_hook_info.func_actual[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        varPack, paramLabel, ptr);
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute_posthook(
    const cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr)
{
    trace_dump.dump("cudnnGetFusedOpsVariantParamPackAttribute");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsPlan_prehook(
    cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsPlan_proxy(
    cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops)
{
    typedef decltype(&cudnnCreateFusedOpsPlan) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE_FUSED_OPS_PLAN])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCreateFusedOpsPlan));
        cudnn_hook_info.func_actual[CUDNN_CREATE_FUSED_OPS_PLAN] = actual_func;
    }
    return ((func_type)actual_func)(
        plan, ops);
}

cudnnStatus_t cudnnCreateFusedOpsPlan_posthook(
    cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops)
{
    trace_dump.dump("cudnnCreateFusedOpsPlan");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan_prehook(
    cudnnFusedOpsPlan_t plan)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan_proxy(
    cudnnFusedOpsPlan_t plan)
{
    typedef decltype(&cudnnDestroyFusedOpsPlan) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY_FUSED_OPS_PLAN])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnDestroyFusedOpsPlan));
        cudnn_hook_info.func_actual[CUDNN_DESTROY_FUSED_OPS_PLAN] = actual_func;
    }
    return ((func_type)actual_func)(
        plan);
}

cudnnStatus_t cudnnDestroyFusedOpsPlan_posthook(
    cudnnFusedOpsPlan_t plan)
{
    trace_dump.dump("cudnnDestroyFusedOpsPlan");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnTrainVersionCheck_prehook(
)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnTrainVersionCheck_proxy(
)
{
    typedef decltype(&cudnnCnnTrainVersionCheck) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CNN_TRAIN_VERSION_CHECK])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnCnnTrainVersionCheck));
        cudnn_hook_info.func_actual[CUDNN_CNN_TRAIN_VERSION_CHECK] = actual_func;
    }
    return ((func_type)actual_func)(
);
}

cudnnStatus_t cudnnCnnTrainVersionCheck_posthook(
)
{
    trace_dump.dump("cudnnCnnTrainVersionCheck");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendCreateDescriptor_prehook(
    cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendCreateDescriptor_proxy(
    cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor)
{
    typedef decltype(&cudnnBackendCreateDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BACKEND_CREATE_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBackendCreateDescriptor));
        cudnn_hook_info.func_actual[CUDNN_BACKEND_CREATE_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        descriptorType, descriptor);
}

cudnnStatus_t cudnnBackendCreateDescriptor_posthook(
    cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor)
{
    trace_dump.dump("cudnnBackendCreateDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor_prehook(
    cudnnBackendDescriptor_t descriptor)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor_proxy(
    cudnnBackendDescriptor_t descriptor)
{
    typedef decltype(&cudnnBackendDestroyDescriptor) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BACKEND_DESTROY_DESCRIPTOR])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBackendDestroyDescriptor));
        cudnn_hook_info.func_actual[CUDNN_BACKEND_DESTROY_DESCRIPTOR] = actual_func;
    }
    return ((func_type)actual_func)(
        descriptor);
}

cudnnStatus_t cudnnBackendDestroyDescriptor_posthook(
    cudnnBackendDescriptor_t descriptor)
{
    trace_dump.dump("cudnnBackendDestroyDescriptor");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendInitialize_prehook(
    cudnnBackendDescriptor_t descriptor)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendInitialize_proxy(
    cudnnBackendDescriptor_t descriptor)
{
    typedef decltype(&cudnnBackendInitialize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BACKEND_INITIALIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBackendInitialize));
        cudnn_hook_info.func_actual[CUDNN_BACKEND_INITIALIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        descriptor);
}

cudnnStatus_t cudnnBackendInitialize_posthook(
    cudnnBackendDescriptor_t descriptor)
{
    trace_dump.dump("cudnnBackendInitialize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendFinalize_prehook(
    cudnnBackendDescriptor_t descriptor)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendFinalize_proxy(
    cudnnBackendDescriptor_t descriptor)
{
    typedef decltype(&cudnnBackendFinalize) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BACKEND_FINALIZE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBackendFinalize));
        cudnn_hook_info.func_actual[CUDNN_BACKEND_FINALIZE] = actual_func;
    }
    return ((func_type)actual_func)(
        descriptor);
}

cudnnStatus_t cudnnBackendFinalize_posthook(
    cudnnBackendDescriptor_t descriptor)
{
    trace_dump.dump("cudnnBackendFinalize");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendSetAttribute_prehook(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t elementCount,
    const void *arrayOfElements)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendSetAttribute_proxy(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t elementCount,
    const void *arrayOfElements)
{
    typedef decltype(&cudnnBackendSetAttribute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BACKEND_SET_ATTRIBUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBackendSetAttribute));
        cudnn_hook_info.func_actual[CUDNN_BACKEND_SET_ATTRIBUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        descriptor, attributeName, attributeType, elementCount,
        arrayOfElements);
}

cudnnStatus_t cudnnBackendSetAttribute_posthook(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t elementCount,
    const void *arrayOfElements)
{
    trace_dump.dump("cudnnBackendSetAttribute");
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute_prehook(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount,
    int64_t *elementCount, void *arrayOfElements)
{
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute_proxy(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount,
    int64_t *elementCount, void *arrayOfElements)
{
    typedef decltype(&cudnnBackendGetAttribute) func_type;
    void *actual_func;
    if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_BACKEND_GET_ATTRIBUTE])) {
        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(cudnnBackendGetAttribute));
        cudnn_hook_info.func_actual[CUDNN_BACKEND_GET_ATTRIBUTE] = actual_func;
    }
    return ((func_type)actual_func)(
        descriptor, attributeName, attributeType, requestedElementCount,
        elementCount, arrayOfElements);
}

cudnnStatus_t cudnnBackendGetAttribute_posthook(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount,
    int64_t *elementCount, void *arrayOfElements)
{
    trace_dump.dump("cudnnBackendGetAttribute");
    return CUDNN_STATUS_SUCCESS;
}
/* prehook, proxy, posthook functions end */

static void cudnn_hook_init()
{
    cudnn_hook_info.func_prehook[CUDNN_CREATE] =
        reinterpret_cast<void *>(cudnnCreate_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE] =
        reinterpret_cast<void *>(cudnnCreate_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE] =
        reinterpret_cast<void *>(cudnnCreate_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY] =
        reinterpret_cast<void *>(cudnnDestroy_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY] =
        reinterpret_cast<void *>(cudnnDestroy_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY] =
        reinterpret_cast<void *>(cudnnDestroy_posthook);
    cudnn_hook_info.func_prehook[CUDNN_QUERY_RUNTIME_ERROR] =
        reinterpret_cast<void *>(cudnnQueryRuntimeError_prehook);
    cudnn_hook_info.func_proxy[CUDNN_QUERY_RUNTIME_ERROR] =
        reinterpret_cast<void *>(cudnnQueryRuntimeError_proxy);
    cudnn_hook_info.func_posthook[CUDNN_QUERY_RUNTIME_ERROR] =
        reinterpret_cast<void *>(cudnnQueryRuntimeError_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_PROPERTY] =
        reinterpret_cast<void *>(cudnnGetProperty_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_PROPERTY] =
        reinterpret_cast<void *>(cudnnGetProperty_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_PROPERTY] =
        reinterpret_cast<void *>(cudnnGetProperty_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_STREAM] =
        reinterpret_cast<void *>(cudnnSetStream_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_STREAM] =
        reinterpret_cast<void *>(cudnnSetStream_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_STREAM] =
        reinterpret_cast<void *>(cudnnSetStream_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_STREAM] =
        reinterpret_cast<void *>(cudnnGetStream_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_STREAM] =
        reinterpret_cast<void *>(cudnnGetStream_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_STREAM] =
        reinterpret_cast<void *>(cudnnGetStream_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensor4dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensor4dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensor4dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetTensor4dDescriptorEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetTensor4dDescriptorEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetTensor4dDescriptorEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensor4dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensor4dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensor4dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensorNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensorNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensorNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetTensorNdDescriptorEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetTensorNdDescriptorEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetTensorNdDescriptorEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensorNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensorNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensorNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_SIZE_IN_BYTES] =
        reinterpret_cast<void *>(cudnnGetTensorSizeInBytes_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_SIZE_IN_BYTES] =
        reinterpret_cast<void *>(cudnnGetTensorSizeInBytes_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_SIZE_IN_BYTES] =
        reinterpret_cast<void *>(cudnnGetTensorSizeInBytes_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_INIT_TRANSFORM_DEST] =
        reinterpret_cast<void *>(cudnnInitTransformDest_prehook);
    cudnn_hook_info.func_proxy[CUDNN_INIT_TRANSFORM_DEST] =
        reinterpret_cast<void *>(cudnnInitTransformDest_proxy);
    cudnn_hook_info.func_posthook[CUDNN_INIT_TRANSFORM_DEST] =
        reinterpret_cast<void *>(cudnnInitTransformDest_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateTensorTransformDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateTensorTransformDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateTensorTransformDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensorTransformDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensorTransformDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetTensorTransformDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensorTransformDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensorTransformDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetTensorTransformDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyTensorTransformDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyTensorTransformDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyTensorTransformDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_TRANSFORM_TENSOR] =
        reinterpret_cast<void *>(cudnnTransformTensor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_TRANSFORM_TENSOR] =
        reinterpret_cast<void *>(cudnnTransformTensor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_TRANSFORM_TENSOR] =
        reinterpret_cast<void *>(cudnnTransformTensor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_TRANSFORM_TENSOR_EX] =
        reinterpret_cast<void *>(cudnnTransformTensorEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_TRANSFORM_TENSOR_EX] =
        reinterpret_cast<void *>(cudnnTransformTensorEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_TRANSFORM_TENSOR_EX] =
        reinterpret_cast<void *>(cudnnTransformTensorEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_ADD_TENSOR] =
        reinterpret_cast<void *>(cudnnAddTensor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_ADD_TENSOR] =
        reinterpret_cast<void *>(cudnnAddTensor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_ADD_TENSOR] =
        reinterpret_cast<void *>(cudnnAddTensor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateOpTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateOpTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateOpTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetOpTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetOpTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetOpTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetOpTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetOpTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetOpTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyOpTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyOpTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyOpTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_OP_TENSOR] =
        reinterpret_cast<void *>(cudnnOpTensor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_OP_TENSOR] =
        reinterpret_cast<void *>(cudnnOpTensor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_OP_TENSOR] =
        reinterpret_cast<void *>(cudnnOpTensor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateReduceTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateReduceTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateReduceTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetReduceTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetReduceTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetReduceTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetReduceTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetReduceTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetReduceTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyReduceTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyReduceTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyReduceTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_REDUCTION_INDICES_SIZE] =
        reinterpret_cast<void *>(cudnnGetReductionIndicesSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_REDUCTION_INDICES_SIZE] =
        reinterpret_cast<void *>(cudnnGetReductionIndicesSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_REDUCTION_INDICES_SIZE] =
        reinterpret_cast<void *>(cudnnGetReductionIndicesSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_REDUCTION_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetReductionWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_REDUCTION_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetReductionWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_REDUCTION_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetReductionWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_REDUCE_TENSOR] =
        reinterpret_cast<void *>(cudnnReduceTensor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_REDUCE_TENSOR] =
        reinterpret_cast<void *>(cudnnReduceTensor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_REDUCE_TENSOR] =
        reinterpret_cast<void *>(cudnnReduceTensor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR] =
        reinterpret_cast<void *>(cudnnSetTensor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR] =
        reinterpret_cast<void *>(cudnnSetTensor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR] =
        reinterpret_cast<void *>(cudnnSetTensor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SCALE_TENSOR] =
        reinterpret_cast<void *>(cudnnScaleTensor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SCALE_TENSOR] =
        reinterpret_cast<void *>(cudnnScaleTensor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SCALE_TENSOR] =
        reinterpret_cast<void *>(cudnnScaleTensor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FILTER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateFilterDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FILTER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateFilterDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FILTER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateFilterDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_FILTER_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetFilter4dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_FILTER_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetFilter4dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_FILTER_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetFilter4dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_FILTER_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetFilter4dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_FILTER_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetFilter4dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_FILTER_4D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetFilter4dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_FILTER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetFilterNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_FILTER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetFilterNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_FILTER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetFilterNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_FILTER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetFilterNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_FILTER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetFilterNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_FILTER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetFilterNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_FILTER_SIZE_IN_BYTES] =
        reinterpret_cast<void *>(cudnnGetFilterSizeInBytes_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_FILTER_SIZE_IN_BYTES] =
        reinterpret_cast<void *>(cudnnGetFilterSizeInBytes_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_FILTER_SIZE_IN_BYTES] =
        reinterpret_cast<void *>(cudnnGetFilterSizeInBytes_posthook);
    cudnn_hook_info.func_prehook[CUDNN_TRANSFORM_FILTER] =
        reinterpret_cast<void *>(cudnnTransformFilter_prehook);
    cudnn_hook_info.func_proxy[CUDNN_TRANSFORM_FILTER] =
        reinterpret_cast<void *>(cudnnTransformFilter_proxy);
    cudnn_hook_info.func_posthook[CUDNN_TRANSFORM_FILTER] =
        reinterpret_cast<void *>(cudnnTransformFilter_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FILTER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyFilterDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FILTER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyFilterDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FILTER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyFilterDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SOFTMAX_FORWARD] =
        reinterpret_cast<void *>(cudnnSoftmaxForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SOFTMAX_FORWARD] =
        reinterpret_cast<void *>(cudnnSoftmaxForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SOFTMAX_FORWARD] =
        reinterpret_cast<void *>(cudnnSoftmaxForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_POOLING_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreatePoolingDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_POOLING_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreatePoolingDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_POOLING_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreatePoolingDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_POOLING_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetPooling2dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_POOLING_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetPooling2dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_POOLING_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetPooling2dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetPooling2dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetPooling2dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetPooling2dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_POOLING_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetPoolingNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_POOLING_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetPoolingNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_POOLING_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetPoolingNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetPoolingNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetPoolingNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetPoolingNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetPoolingNdForwardOutputDim_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetPoolingNdForwardOutputDim_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetPoolingNdForwardOutputDim_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetPooling2dForwardOutputDim_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetPooling2dForwardOutputDim_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetPooling2dForwardOutputDim_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_POOLING_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyPoolingDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_POOLING_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyPoolingDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_POOLING_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyPoolingDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_POOLING_FORWARD] =
        reinterpret_cast<void *>(cudnnPoolingForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_POOLING_FORWARD] =
        reinterpret_cast<void *>(cudnnPoolingForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_POOLING_FORWARD] =
        reinterpret_cast<void *>(cudnnPoolingForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateActivationDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateActivationDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateActivationDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetActivationDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetActivationDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetActivationDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetActivationDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetActivationDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetActivationDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        reinterpret_cast<void *>(cudnnSetActivationDescriptorSwishBeta_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        reinterpret_cast<void *>(cudnnSetActivationDescriptorSwishBeta_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        reinterpret_cast<void *>(cudnnSetActivationDescriptorSwishBeta_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        reinterpret_cast<void *>(cudnnGetActivationDescriptorSwishBeta_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        reinterpret_cast<void *>(cudnnGetActivationDescriptorSwishBeta_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        reinterpret_cast<void *>(cudnnGetActivationDescriptorSwishBeta_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyActivationDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyActivationDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyActivationDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_ACTIVATION_FORWARD] =
        reinterpret_cast<void *>(cudnnActivationForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_ACTIVATION_FORWARD] =
        reinterpret_cast<void *>(cudnnActivationForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_ACTIVATION_FORWARD] =
        reinterpret_cast<void *>(cudnnActivationForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateLRNDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateLRNDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateLRNDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetLRNDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetLRNDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetLRNDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetLRNDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetLRNDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetLRNDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyLRNDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyLRNDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_LRN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyLRNDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_LRN_CROSS_CHANNEL_FORWARD] =
        reinterpret_cast<void *>(cudnnLRNCrossChannelForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_LRN_CROSS_CHANNEL_FORWARD] =
        reinterpret_cast<void *>(cudnnLRNCrossChannelForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_LRN_CROSS_CHANNEL_FORWARD] =
        reinterpret_cast<void *>(cudnnLRNCrossChannelForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DIVISIVE_NORMALIZATION_FORWARD] =
        reinterpret_cast<void *>(cudnnDivisiveNormalizationForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DIVISIVE_NORMALIZATION_FORWARD] =
        reinterpret_cast<void *>(cudnnDivisiveNormalizationForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DIVISIVE_NORMALIZATION_FORWARD] =
        reinterpret_cast<void *>(cudnnDivisiveNormalizationForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDeriveBNTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDeriveBNTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDeriveBNTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardInference_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardInference_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardInference_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDeriveNormTensorDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDeriveNormTensorDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDeriveNormTensorDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_NORMALIZATION_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnNormalizationForwardInference_prehook);
    cudnn_hook_info.func_proxy[CUDNN_NORMALIZATION_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnNormalizationForwardInference_proxy);
    cudnn_hook_info.func_posthook[CUDNN_NORMALIZATION_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnNormalizationForwardInference_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateSpatialTransformerDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateSpatialTransformerDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateSpatialTransformerDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetSpatialTransformerNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetSpatialTransformerNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetSpatialTransformerNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroySpatialTransformerDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroySpatialTransformerDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroySpatialTransformerDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfGridGeneratorForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfGridGeneratorForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfGridGeneratorForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_SAMPLER_FORWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfSamplerForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_SAMPLER_FORWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfSamplerForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_SAMPLER_FORWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfSamplerForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateDropoutDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateDropoutDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateDropoutDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyDropoutDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyDropoutDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyDropoutDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_GET_STATES_SIZE] =
        reinterpret_cast<void *>(cudnnDropoutGetStatesSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_GET_STATES_SIZE] =
        reinterpret_cast<void *>(cudnnDropoutGetStatesSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_GET_STATES_SIZE] =
        reinterpret_cast<void *>(cudnnDropoutGetStatesSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnDropoutGetReserveSpaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnDropoutGetReserveSpaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnDropoutGetReserveSpaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetDropoutDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetDropoutDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetDropoutDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RESTORE_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnRestoreDropoutDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RESTORE_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnRestoreDropoutDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RESTORE_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnRestoreDropoutDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetDropoutDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetDropoutDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_DROPOUT_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetDropoutDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_FORWARD] =
        reinterpret_cast<void *>(cudnnDropoutForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_FORWARD] =
        reinterpret_cast<void *>(cudnnDropoutForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_FORWARD] =
        reinterpret_cast<void *>(cudnnDropoutForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateAlgorithmDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateAlgorithmDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateAlgorithmDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetAlgorithmDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetAlgorithmDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetAlgorithmDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetAlgorithmDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetAlgorithmDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetAlgorithmDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_COPY_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCopyAlgorithmDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_COPY_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCopyAlgorithmDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_COPY_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCopyAlgorithmDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyAlgorithmDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyAlgorithmDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyAlgorithmDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnCreateAlgorithmPerformance_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnCreateAlgorithmPerformance_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnCreateAlgorithmPerformance_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnSetAlgorithmPerformance_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnSetAlgorithmPerformance_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnSetAlgorithmPerformance_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnGetAlgorithmPerformance_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnGetAlgorithmPerformance_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnGetAlgorithmPerformance_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnDestroyAlgorithmPerformance_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnDestroyAlgorithmPerformance_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ALGORITHM_PERFORMANCE] =
        reinterpret_cast<void *>(cudnnDestroyAlgorithmPerformance_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_ALGORITHM_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetAlgorithmSpaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_ALGORITHM_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetAlgorithmSpaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_ALGORITHM_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetAlgorithmSpaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SAVE_ALGORITHM] =
        reinterpret_cast<void *>(cudnnSaveAlgorithm_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SAVE_ALGORITHM] =
        reinterpret_cast<void *>(cudnnSaveAlgorithm_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SAVE_ALGORITHM] =
        reinterpret_cast<void *>(cudnnSaveAlgorithm_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RESTORE_ALGORITHM] =
        reinterpret_cast<void *>(cudnnRestoreAlgorithm_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RESTORE_ALGORITHM] =
        reinterpret_cast<void *>(cudnnRestoreAlgorithm_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RESTORE_ALGORITHM] =
        reinterpret_cast<void *>(cudnnRestoreAlgorithm_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CALLBACK] =
        reinterpret_cast<void *>(cudnnSetCallback_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CALLBACK] =
        reinterpret_cast<void *>(cudnnSetCallback_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CALLBACK] =
        reinterpret_cast<void *>(cudnnSetCallback_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CALLBACK] =
        reinterpret_cast<void *>(cudnnGetCallback_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CALLBACK] =
        reinterpret_cast<void *>(cudnnGetCallback_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CALLBACK] =
        reinterpret_cast<void *>(cudnnGetCallback_posthook);
    cudnn_hook_info.func_prehook[CUDNN_OPS_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnOpsInferVersionCheck_prehook);
    cudnn_hook_info.func_proxy[CUDNN_OPS_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnOpsInferVersionCheck_proxy);
    cudnn_hook_info.func_posthook[CUDNN_OPS_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnOpsInferVersionCheck_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SOFTMAX_BACKWARD] =
        reinterpret_cast<void *>(cudnnSoftmaxBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SOFTMAX_BACKWARD] =
        reinterpret_cast<void *>(cudnnSoftmaxBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SOFTMAX_BACKWARD] =
        reinterpret_cast<void *>(cudnnSoftmaxBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_POOLING_BACKWARD] =
        reinterpret_cast<void *>(cudnnPoolingBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_POOLING_BACKWARD] =
        reinterpret_cast<void *>(cudnnPoolingBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_POOLING_BACKWARD] =
        reinterpret_cast<void *>(cudnnPoolingBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_ACTIVATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnActivationBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_ACTIVATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnActivationBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_ACTIVATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnActivationBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_LRN_CROSS_CHANNEL_BACKWARD] =
        reinterpret_cast<void *>(cudnnLRNCrossChannelBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_LRN_CROSS_CHANNEL_BACKWARD] =
        reinterpret_cast<void *>(cudnnLRNCrossChannelBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_LRN_CROSS_CHANNEL_BACKWARD] =
        reinterpret_cast<void *>(cudnnLRNCrossChannelBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnDivisiveNormalizationBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnDivisiveNormalizationBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnDivisiveNormalizationBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationBackwardExWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationBackwardExWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationBackwardExWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationTrainingExReserveSpaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationTrainingExReserveSpaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetBatchNormalizationTrainingExReserveSpaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardTraining_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardTraining_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardTraining_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardTrainingEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardTrainingEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX] =
        reinterpret_cast<void *>(cudnnBatchNormalizationForwardTrainingEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnBatchNormalizationBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnBatchNormalizationBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnBatchNormalizationBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX] =
        reinterpret_cast<void *>(cudnnBatchNormalizationBackwardEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX] =
        reinterpret_cast<void *>(cudnnBatchNormalizationBackwardEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX] =
        reinterpret_cast<void *>(cudnnBatchNormalizationBackwardEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationForwardTrainingWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationForwardTrainingWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationForwardTrainingWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationBackwardWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationBackwardWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationBackwardWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationTrainingReserveSpaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationTrainingReserveSpaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetNormalizationTrainingReserveSpaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_NORMALIZATION_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnNormalizationForwardTraining_prehook);
    cudnn_hook_info.func_proxy[CUDNN_NORMALIZATION_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnNormalizationForwardTraining_proxy);
    cudnn_hook_info.func_posthook[CUDNN_NORMALIZATION_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnNormalizationForwardTraining_posthook);
    cudnn_hook_info.func_prehook[CUDNN_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnNormalizationBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnNormalizationBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_NORMALIZATION_BACKWARD] =
        reinterpret_cast<void *>(cudnnNormalizationBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfGridGeneratorBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfGridGeneratorBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfGridGeneratorBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfSamplerBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfSamplerBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD] =
        reinterpret_cast<void *>(cudnnSpatialTfSamplerBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_BACKWARD] =
        reinterpret_cast<void *>(cudnnDropoutBackward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_BACKWARD] =
        reinterpret_cast<void *>(cudnnDropoutBackward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_BACKWARD] =
        reinterpret_cast<void *>(cudnnDropoutBackward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_OPS_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnOpsTrainVersionCheck_prehook);
    cudnn_hook_info.func_proxy[CUDNN_OPS_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnOpsTrainVersionCheck_proxy);
    cudnn_hook_info.func_posthook[CUDNN_OPS_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnOpsTrainVersionCheck_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_RNN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateRNNDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_RNN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateRNNDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_RNN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateRNNDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_RNN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyRNNDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_RNN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyRNNDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_RNN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyRNNDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnSetRNNDescriptor_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnSetRNNDescriptor_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnSetRNNDescriptor_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnGetRNNDescriptor_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnGetRNNDescriptor_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnGetRNNDescriptor_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_DESCRIPTOR_V6] =
        reinterpret_cast<void *>(cudnnSetRNNDescriptor_v6_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_DESCRIPTOR_V6] =
        reinterpret_cast<void *>(cudnnSetRNNDescriptor_v6_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_DESCRIPTOR_V6] =
        reinterpret_cast<void *>(cudnnSetRNNDescriptor_v6_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_DESCRIPTOR_V6] =
        reinterpret_cast<void *>(cudnnGetRNNDescriptor_v6_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_DESCRIPTOR_V6] =
        reinterpret_cast<void *>(cudnnGetRNNDescriptor_v6_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_DESCRIPTOR_V6] =
        reinterpret_cast<void *>(cudnnGetRNNDescriptor_v6_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_MATRIX_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnSetRNNMatrixMathType_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_MATRIX_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnSetRNNMatrixMathType_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_MATRIX_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnSetRNNMatrixMathType_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_MATRIX_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnGetRNNMatrixMathType_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_MATRIX_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnGetRNNMatrixMathType_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_MATRIX_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnGetRNNMatrixMathType_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_BIAS_MODE] =
        reinterpret_cast<void *>(cudnnSetRNNBiasMode_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_BIAS_MODE] =
        reinterpret_cast<void *>(cudnnSetRNNBiasMode_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_BIAS_MODE] =
        reinterpret_cast<void *>(cudnnSetRNNBiasMode_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_BIAS_MODE] =
        reinterpret_cast<void *>(cudnnGetRNNBiasMode_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_BIAS_MODE] =
        reinterpret_cast<void *>(cudnnGetRNNBiasMode_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_BIAS_MODE] =
        reinterpret_cast<void *>(cudnnGetRNNBiasMode_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_SET_CLIP_V8] =
        reinterpret_cast<void *>(cudnnRNNSetClip_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_SET_CLIP_V8] =
        reinterpret_cast<void *>(cudnnRNNSetClip_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_SET_CLIP_V8] =
        reinterpret_cast<void *>(cudnnRNNSetClip_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_GET_CLIP_V8] =
        reinterpret_cast<void *>(cudnnRNNGetClip_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_GET_CLIP_V8] =
        reinterpret_cast<void *>(cudnnRNNGetClip_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_GET_CLIP_V8] =
        reinterpret_cast<void *>(cudnnRNNGetClip_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_SET_CLIP] =
        reinterpret_cast<void *>(cudnnRNNSetClip_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_SET_CLIP] =
        reinterpret_cast<void *>(cudnnRNNSetClip_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_SET_CLIP] =
        reinterpret_cast<void *>(cudnnRNNSetClip_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_GET_CLIP] =
        reinterpret_cast<void *>(cudnnRNNGetClip_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_GET_CLIP] =
        reinterpret_cast<void *>(cudnnRNNGetClip_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_GET_CLIP] =
        reinterpret_cast<void *>(cudnnRNNGetClip_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_PROJECTION_LAYERS] =
        reinterpret_cast<void *>(cudnnSetRNNProjectionLayers_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_PROJECTION_LAYERS] =
        reinterpret_cast<void *>(cudnnSetRNNProjectionLayers_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_PROJECTION_LAYERS] =
        reinterpret_cast<void *>(cudnnSetRNNProjectionLayers_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_PROJECTION_LAYERS] =
        reinterpret_cast<void *>(cudnnGetRNNProjectionLayers_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_PROJECTION_LAYERS] =
        reinterpret_cast<void *>(cudnnGetRNNProjectionLayers_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_PROJECTION_LAYERS] =
        reinterpret_cast<void *>(cudnnGetRNNProjectionLayers_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnCreatePersistentRNNPlan_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnCreatePersistentRNNPlan_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnCreatePersistentRNNPlan_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnDestroyPersistentRNNPlan_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnDestroyPersistentRNNPlan_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnDestroyPersistentRNNPlan_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnSetPersistentRNNPlan_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnSetPersistentRNNPlan_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_PERSISTENT_RNN_PLAN] =
        reinterpret_cast<void *>(cudnnSetPersistentRNNPlan_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BUILD_RNN_DYNAMIC] =
        reinterpret_cast<void *>(cudnnBuildRNNDynamic_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BUILD_RNN_DYNAMIC] =
        reinterpret_cast<void *>(cudnnBuildRNNDynamic_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BUILD_RNN_DYNAMIC] =
        reinterpret_cast<void *>(cudnnBuildRNNDynamic_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNTrainingReserveSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNTrainingReserveSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNTrainingReserveSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_TEMP_SPACE_SIZES] =
        reinterpret_cast<void *>(cudnnGetRNNTempSpaceSizes_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_TEMP_SPACE_SIZES] =
        reinterpret_cast<void *>(cudnnGetRNNTempSpaceSizes_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_TEMP_SPACE_SIZES] =
        reinterpret_cast<void *>(cudnnGetRNNTempSpaceSizes_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_PARAMS_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNParamsSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_PARAMS_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNParamsSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_PARAMS_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNParamsSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNWeightSpaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNWeightSpaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetRNNWeightSpaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNLinLayerMatrixParams_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNLinLayerMatrixParams_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNLinLayerMatrixParams_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNLinLayerBiasParams_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNLinLayerBiasParams_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNLinLayerBiasParams_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_WEIGHT_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNWeightParams_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_WEIGHT_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNWeightParams_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_WEIGHT_PARAMS] =
        reinterpret_cast<void *>(cudnnGetRNNWeightParams_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnRNNForwardInference_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnRNNForwardInference_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_INFERENCE] =
        reinterpret_cast<void *>(cudnnRNNForwardInference_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_PADDING_MODE] =
        reinterpret_cast<void *>(cudnnSetRNNPaddingMode_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_PADDING_MODE] =
        reinterpret_cast<void *>(cudnnSetRNNPaddingMode_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_PADDING_MODE] =
        reinterpret_cast<void *>(cudnnSetRNNPaddingMode_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_PADDING_MODE] =
        reinterpret_cast<void *>(cudnnGetRNNPaddingMode_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_PADDING_MODE] =
        reinterpret_cast<void *>(cudnnGetRNNPaddingMode_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_PADDING_MODE] =
        reinterpret_cast<void *>(cudnnGetRNNPaddingMode_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateRNNDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateRNNDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateRNNDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyRNNDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyRNNDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyRNNDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetRNNDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetRNNDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetRNNDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetRNNDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetRNNDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetRNNDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_INFERENCE_EX] =
        reinterpret_cast<void *>(cudnnRNNForwardInferenceEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_INFERENCE_EX] =
        reinterpret_cast<void *>(cudnnRNNForwardInferenceEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_INFERENCE_EX] =
        reinterpret_cast<void *>(cudnnRNNForwardInferenceEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD] =
        reinterpret_cast<void *>(cudnnRNNForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD] =
        reinterpret_cast<void *>(cudnnRNNForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD] =
        reinterpret_cast<void *>(cudnnRNNForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetRNNAlgorithmDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetRNNAlgorithmDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetRNNAlgorithmDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNForwardInferenceAlgorithmMaxCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNForwardInferenceAlgorithmMaxCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNForwardInferenceAlgorithmMaxCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNForwardInferenceAlgorithmEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNForwardInferenceAlgorithmEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNForwardInferenceAlgorithmEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateSeqDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateSeqDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateSeqDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroySeqDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroySeqDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroySeqDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetSeqDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetSeqDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetSeqDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetSeqDataDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetSeqDataDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_SEQ_DATA_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetSeqDataDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateAttnDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateAttnDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateAttnDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyAttnDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyAttnDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyAttnDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetAttnDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetAttnDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetAttnDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetAttnDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetAttnDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_ATTN_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetAttnDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS] =
        reinterpret_cast<void *>(cudnnGetMultiHeadAttnBuffers_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS] =
        reinterpret_cast<void *>(cudnnGetMultiHeadAttnBuffers_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS] =
        reinterpret_cast<void *>(cudnnGetMultiHeadAttnBuffers_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS] =
        reinterpret_cast<void *>(cudnnGetMultiHeadAttnWeights_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS] =
        reinterpret_cast<void *>(cudnnGetMultiHeadAttnWeights_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS] =
        reinterpret_cast<void *>(cudnnGetMultiHeadAttnWeights_posthook);
    cudnn_hook_info.func_prehook[CUDNN_MULTI_HEAD_ATTN_FORWARD] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_MULTI_HEAD_ATTN_FORWARD] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_MULTI_HEAD_ATTN_FORWARD] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_ADV_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnAdvInferVersionCheck_prehook);
    cudnn_hook_info.func_proxy[CUDNN_ADV_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnAdvInferVersionCheck_proxy);
    cudnn_hook_info.func_posthook[CUDNN_ADV_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnAdvInferVersionCheck_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnRNNForwardTraining_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnRNNForwardTraining_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_TRAINING] =
        reinterpret_cast<void *>(cudnnRNNForwardTraining_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnRNNBackwardData_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnRNNBackwardData_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnRNNBackwardData_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_DATA_V8] =
        reinterpret_cast<void *>(cudnnRNNBackwardData_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_DATA_V8] =
        reinterpret_cast<void *>(cudnnRNNBackwardData_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_DATA_V8] =
        reinterpret_cast<void *>(cudnnRNNBackwardData_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_WEIGHTS] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeights_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_WEIGHTS] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeights_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_WEIGHTS] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeights_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_WEIGHTS_V8] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeights_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_WEIGHTS_V8] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeights_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_WEIGHTS_V8] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeights_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_TRAINING_EX] =
        reinterpret_cast<void *>(cudnnRNNForwardTrainingEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_TRAINING_EX] =
        reinterpret_cast<void *>(cudnnRNNForwardTrainingEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_TRAINING_EX] =
        reinterpret_cast<void *>(cudnnRNNForwardTrainingEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_DATA_EX] =
        reinterpret_cast<void *>(cudnnRNNBackwardDataEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_DATA_EX] =
        reinterpret_cast<void *>(cudnnRNNBackwardDataEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_DATA_EX] =
        reinterpret_cast<void *>(cudnnRNNBackwardDataEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_WEIGHTS_EX] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeightsEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_WEIGHTS_EX] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeightsEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_WEIGHTS_EX] =
        reinterpret_cast<void *>(cudnnRNNBackwardWeightsEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNForwardTrainingAlgorithmMaxCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNForwardTrainingAlgorithmMaxCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNForwardTrainingAlgorithmMaxCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNForwardTrainingAlgorithmEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNForwardTrainingAlgorithmEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNForwardTrainingAlgorithmEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNBackwardDataAlgorithmMaxCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNBackwardDataAlgorithmMaxCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNBackwardDataAlgorithmMaxCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNBackwardDataAlgorithmEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNBackwardDataAlgorithmEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNBackwardDataAlgorithmEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNBackwardWeightsAlgorithmMaxCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNBackwardWeightsAlgorithmMaxCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetRNNBackwardWeightsAlgorithmMaxCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNBackwardWeightsAlgorithmEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNBackwardWeightsAlgorithmEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindRNNBackwardWeightsAlgorithmEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnBackwardData_prehook);
    cudnn_hook_info.func_proxy[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnBackwardData_proxy);
    cudnn_hook_info.func_posthook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnBackwardData_posthook);
    cudnn_hook_info.func_prehook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnBackwardWeights_prehook);
    cudnn_hook_info.func_proxy[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnBackwardWeights_proxy);
    cudnn_hook_info.func_posthook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS] =
        reinterpret_cast<void *>(cudnnMultiHeadAttnBackwardWeights_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateCTCLossDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateCTCLossDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateCTCLossDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptorEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptorEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptorEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptor_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptor_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnSetCTCLossDescriptor_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptorEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptorEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptorEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptor_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptor_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8] =
        reinterpret_cast<void *>(cudnnGetCTCLossDescriptor_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyCTCLossDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyCTCLossDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyCTCLossDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CTC_LOSS] =
        reinterpret_cast<void *>(cudnnCTCLoss_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CTC_LOSS] =
        reinterpret_cast<void *>(cudnnCTCLoss_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CTC_LOSS] =
        reinterpret_cast<void *>(cudnnCTCLoss_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CTC_LOSS_V8] =
        reinterpret_cast<void *>(cudnnCTCLoss_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CTC_LOSS_V8] =
        reinterpret_cast<void *>(cudnnCTCLoss_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CTC_LOSS_V8] =
        reinterpret_cast<void *>(cudnnCTCLoss_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetCTCLossWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetCTCLossWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetCTCLossWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8] =
        reinterpret_cast<void *>(cudnnGetCTCLossWorkspaceSize_v8_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8] =
        reinterpret_cast<void *>(cudnnGetCTCLossWorkspaceSize_v8_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8] =
        reinterpret_cast<void *>(cudnnGetCTCLossWorkspaceSize_v8_posthook);
    cudnn_hook_info.func_prehook[CUDNN_ADV_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnAdvTrainVersionCheck_prehook);
    cudnn_hook_info.func_proxy[CUDNN_ADV_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnAdvTrainVersionCheck_proxy);
    cudnn_hook_info.func_posthook[CUDNN_ADV_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnAdvTrainVersionCheck_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateConvolutionDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateConvolutionDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnCreateConvolutionDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyConvolutionDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyConvolutionDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnDestroyConvolutionDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnSetConvolutionMathType_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnSetConvolutionMathType_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnSetConvolutionMathType_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnGetConvolutionMathType_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnGetConvolutionMathType_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_MATH_TYPE] =
        reinterpret_cast<void *>(cudnnGetConvolutionMathType_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_GROUP_COUNT] =
        reinterpret_cast<void *>(cudnnSetConvolutionGroupCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_GROUP_COUNT] =
        reinterpret_cast<void *>(cudnnSetConvolutionGroupCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_GROUP_COUNT] =
        reinterpret_cast<void *>(cudnnSetConvolutionGroupCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_GROUP_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionGroupCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_GROUP_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionGroupCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_GROUP_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionGroupCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_REORDER_TYPE] =
        reinterpret_cast<void *>(cudnnSetConvolutionReorderType_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_REORDER_TYPE] =
        reinterpret_cast<void *>(cudnnSetConvolutionReorderType_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_REORDER_TYPE] =
        reinterpret_cast<void *>(cudnnSetConvolutionReorderType_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_REORDER_TYPE] =
        reinterpret_cast<void *>(cudnnGetConvolutionReorderType_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_REORDER_TYPE] =
        reinterpret_cast<void *>(cudnnGetConvolutionReorderType_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_REORDER_TYPE] =
        reinterpret_cast<void *>(cudnnGetConvolutionReorderType_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetConvolution2dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetConvolution2dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetConvolution2dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetConvolution2dDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetConvolution2dDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetConvolution2dDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetConvolutionNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetConvolutionNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnSetConvolutionNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetConvolutionNdDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetConvolutionNdDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnGetConvolutionNdDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetConvolution2dForwardOutputDim_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetConvolution2dForwardOutputDim_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetConvolution2dForwardOutputDim_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetConvolutionNdForwardOutputDim_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetConvolutionNdForwardOutputDim_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM] =
        reinterpret_cast<void *>(cudnnGetConvolutionNdForwardOutputDim_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardAlgorithmMaxCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardAlgorithmMaxCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardAlgorithmMaxCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardAlgorithm_v7_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardAlgorithm_v7_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardAlgorithm_v7_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionForwardAlgorithm_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionForwardAlgorithm_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionForwardAlgorithm_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionForwardAlgorithmEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionForwardAlgorithmEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionForwardAlgorithmEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_IM_2_COL] =
        reinterpret_cast<void *>(cudnnIm2Col_prehook);
    cudnn_hook_info.func_proxy[CUDNN_IM_2_COL] =
        reinterpret_cast<void *>(cudnnIm2Col_proxy);
    cudnn_hook_info.func_posthook[CUDNN_IM_2_COL] =
        reinterpret_cast<void *>(cudnnIm2Col_posthook);
    cudnn_hook_info.func_prehook[CUDNN_REORDER_FILTER_AND_BIAS] =
        reinterpret_cast<void *>(cudnnReorderFilterAndBias_prehook);
    cudnn_hook_info.func_proxy[CUDNN_REORDER_FILTER_AND_BIAS] =
        reinterpret_cast<void *>(cudnnReorderFilterAndBias_proxy);
    cudnn_hook_info.func_posthook[CUDNN_REORDER_FILTER_AND_BIAS] =
        reinterpret_cast<void *>(cudnnReorderFilterAndBias_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionForwardWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_FORWARD] =
        reinterpret_cast<void *>(cudnnConvolutionForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_FORWARD] =
        reinterpret_cast<void *>(cudnnConvolutionForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_FORWARD] =
        reinterpret_cast<void *>(cudnnConvolutionForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD] =
        reinterpret_cast<void *>(cudnnConvolutionBiasActivationForward_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD] =
        reinterpret_cast<void *>(cudnnConvolutionBiasActivationForward_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD] =
        reinterpret_cast<void *>(cudnnConvolutionBiasActivationForward_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataAlgorithmMaxCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataAlgorithmMaxCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataAlgorithmMaxCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardDataAlgorithm_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardDataAlgorithm_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardDataAlgorithm_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardDataAlgorithmEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardDataAlgorithmEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardDataAlgorithmEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataAlgorithm_v7_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataAlgorithm_v7_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataAlgorithm_v7_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardDataWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardData_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardData_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BACKWARD_DATA] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardData_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS] =
        reinterpret_cast<void *>(cudnnGetFoldedConvBackwardDataDescriptors_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS] =
        reinterpret_cast<void *>(cudnnGetFoldedConvBackwardDataDescriptors_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS] =
        reinterpret_cast<void *>(cudnnGetFoldedConvBackwardDataDescriptors_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CNN_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnCnnInferVersionCheck_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CNN_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnCnnInferVersionCheck_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CNN_INFER_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnCnnInferVersionCheck_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardFilterAlgorithm_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardFilterAlgorithm_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardFilterAlgorithm_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardFilterAlgorithmEx_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardFilterAlgorithmEx_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX] =
        reinterpret_cast<void *>(cudnnFindConvolutionBackwardFilterAlgorithmEx_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterAlgorithm_v7_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterAlgorithm_v7_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterAlgorithm_v7_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterWorkspaceSize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterWorkspaceSize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE] =
        reinterpret_cast<void *>(cudnnGetConvolutionBackwardFilterWorkspaceSize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BACKWARD_FILTER] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardFilter_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BACKWARD_FILTER] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardFilter_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BACKWARD_FILTER] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardFilter_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BACKWARD_BIAS] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardBias_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BACKWARD_BIAS] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardBias_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BACKWARD_BIAS] =
        reinterpret_cast<void *>(cudnnConvolutionBackwardBias_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsConstParamPack_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsConstParamPack_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsConstParamPack_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsConstParamPack_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsConstParamPack_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsConstParamPack_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnSetFusedOpsConstParamPackAttribute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnSetFusedOpsConstParamPackAttribute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnSetFusedOpsConstParamPackAttribute_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnGetFusedOpsConstParamPackAttribute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnGetFusedOpsConstParamPackAttribute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnGetFusedOpsConstParamPackAttribute_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsVariantParamPack_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsVariantParamPack_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsVariantParamPack_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsVariantParamPack_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsVariantParamPack_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsVariantParamPack_posthook);
    cudnn_hook_info.func_prehook[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnSetFusedOpsVariantParamPackAttribute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnSetFusedOpsVariantParamPackAttribute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnSetFusedOpsVariantParamPackAttribute_posthook);
    cudnn_hook_info.func_prehook[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnGetFusedOpsVariantParamPackAttribute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnGetFusedOpsVariantParamPackAttribute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnGetFusedOpsVariantParamPackAttribute_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsPlan_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsPlan_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnCreateFusedOpsPlan_posthook);
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsPlan_prehook);
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsPlan_proxy);
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnDestroyFusedOpsPlan_posthook);
    cudnn_hook_info.func_prehook[CUDNN_MAKE_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnMakeFusedOpsPlan_prehook);
    cudnn_hook_info.func_proxy[CUDNN_MAKE_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnMakeFusedOpsPlan_proxy);
    cudnn_hook_info.func_posthook[CUDNN_MAKE_FUSED_OPS_PLAN] =
        reinterpret_cast<void *>(cudnnMakeFusedOpsPlan_posthook);
    cudnn_hook_info.func_prehook[CUDNN_FUSED_OPS_EXECUTE] =
        reinterpret_cast<void *>(cudnnFusedOpsExecute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_FUSED_OPS_EXECUTE] =
        reinterpret_cast<void *>(cudnnFusedOpsExecute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_FUSED_OPS_EXECUTE] =
        reinterpret_cast<void *>(cudnnFusedOpsExecute_posthook);
    cudnn_hook_info.func_prehook[CUDNN_CNN_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnCnnTrainVersionCheck_prehook);
    cudnn_hook_info.func_proxy[CUDNN_CNN_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnCnnTrainVersionCheck_proxy);
    cudnn_hook_info.func_posthook[CUDNN_CNN_TRAIN_VERSION_CHECK] =
        reinterpret_cast<void *>(cudnnCnnTrainVersionCheck_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_CREATE_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnBackendCreateDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_CREATE_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnBackendCreateDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_CREATE_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnBackendCreateDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_DESTROY_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnBackendDestroyDescriptor_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_DESTROY_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnBackendDestroyDescriptor_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_DESTROY_DESCRIPTOR] =
        reinterpret_cast<void *>(cudnnBackendDestroyDescriptor_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_INITIALIZE] =
        reinterpret_cast<void *>(cudnnBackendInitialize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_INITIALIZE] =
        reinterpret_cast<void *>(cudnnBackendInitialize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_INITIALIZE] =
        reinterpret_cast<void *>(cudnnBackendInitialize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_FINALIZE] =
        reinterpret_cast<void *>(cudnnBackendFinalize_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_FINALIZE] =
        reinterpret_cast<void *>(cudnnBackendFinalize_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_FINALIZE] =
        reinterpret_cast<void *>(cudnnBackendFinalize_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_SET_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnBackendSetAttribute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_SET_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnBackendSetAttribute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_SET_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnBackendSetAttribute_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_GET_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnBackendGetAttribute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_GET_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnBackendGetAttribute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_GET_ATTRIBUTE] =
        reinterpret_cast<void *>(cudnnBackendGetAttribute_posthook);
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_EXECUTE] =
        reinterpret_cast<void *>(cudnnBackendExecute_prehook);
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_EXECUTE] =
        reinterpret_cast<void *>(cudnnBackendExecute_proxy);
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_EXECUTE] =
        reinterpret_cast<void *>(cudnnBackendExecute_posthook);
}

/* hook function start */
cudnnStatus_t cudnnCreate(cudnnHandle_t *handle)
{
    hook_log.debug("Enter function: "s + string(__func__));

    typedef decltype(&cudnnCreate) func_type;
    cudnnStatus_t result;
    void *actual_func;

    pthread_once(&cudnn_hook_init_done, cudnn_hook_init);

    if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_prehook[CUDNN_CREATE]) {
        actual_func = cudnn_hook_info.func_prehook[CUDNN_CREATE];
        ((func_type)actual_func)(handle);
    }

    if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_proxy[CUDNN_CREATE])
        actual_func = cudnn_hook_info.func_proxy[CUDNN_CREATE];
    else if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_CREATE])) {
        actual_func = actual_dlsym(libcudnn_handle, "cudnnCreate");
        cudnn_hook_info.func_actual[CUDNN_CREATE] = actual_func;
    }
    result = ((func_type)actual_func)(handle);

    if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_posthook[CUDNN_CREATE]) {
        actual_func = cudnn_hook_info.func_posthook[CUDNN_CREATE];
        ((func_type)actual_func)(handle);
    }

    hook_log.debug("Leave function: "s + string(__func__));

    return result;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle)
{
    hook_log.debug("Enter function: "s + string(__func__));

    typedef decltype(&cudnnDestroy) func_type;
    cudnnStatus_t result;
    void *actual_func;

    pthread_once(&cudnn_hook_init_done, cudnn_hook_init);

    if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_prehook[CUDNN_DESTROY]) {
        actual_func = cudnn_hook_info.func_prehook[CUDNN_DESTROY];
        ((func_type)actual_func)(handle);
    }

    if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_proxy[CUDNN_DESTROY])
        actual_func = cudnn_hook_info.func_proxy[CUDNN_DESTROY];
    else if(!(actual_func = cudnn_hook_info.func_actual[CUDNN_DESTROY])) {
        actual_func = actual_dlsym(libcudnn_handle, "cudnnDestroy");
        cudnn_hook_info.func_actual[CUDNN_DESTROY] = actual_func;
    }
    result = ((func_type)actual_func)(handle);

    if(cudnn_hook_info.hook_effect_enable && cudnn_hook_info.func_posthook[CUDNN_DESTROY]) {
        actual_func = cudnn_hook_info.func_posthook[CUDNN_DESTROY];
        ((func_type)actual_func)(handle);
    }

    hook_log.debug("Leave function: "s + string(__func__));

    return result;
}

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_QUERY_RUNTIME_ERROR,
    ,
    cudnnQueryRuntimeError,
    (cudnnHandle_t handle, cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag),
    handle, rstatus, mode, tag)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SET_STREAM,
    ,
    cudnnSetStream,
    (cudnnHandle_t handle, cudaStream_t streamId),
    handle, streamId)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_STREAM,
    ,
    cudnnGetStream,
    (cudnnHandle_t handle, cudaStream_t *streamId),
    handle, streamId)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_TRANSFORM_TENSOR,
    ,
    cudnnTransformTensor,
    (cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, alpha, xDesc, x,
    beta, yDesc, y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_TRANSFORM_TENSOR_EX,
    ,
    cudnnTransformTensorEx,
    (cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnTensorDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnTensorDescriptor_t destDesc, void *destData),
    handle, transDesc, alpha, srcDesc,
    srcData, beta, destDesc, destData)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_ADD_TENSOR,
    ,
    cudnnAddTensor,
    (cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C),
    handle, alpha, aDesc, A,
    beta, cDesc, C)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_OP_TENSOR,
    ,
    cudnnOpTensor,
    (cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *alpha2,
    const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc,
    void *C),
    handle, opTensorDesc, alpha1, aDesc,
    A, alpha2, bDesc, B,
    beta, cDesc, C)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_REDUCTION_INDICES_SIZE,
    ,
    cudnnGetReductionIndicesSize,
    (cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes),
    handle, reduceTensorDesc, aDesc, cDesc,
    sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_REDUCTION_WORKSPACE_SIZE,
    ,
    cudnnGetReductionWorkspaceSize,
    (cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes),
    handle, reduceTensorDesc, aDesc, cDesc,
    sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_REDUCE_TENSOR,
    ,
    cudnnReduceTensor,
    (cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes,
    void *workspace, size_t workspaceSizeInBytes,
    const void *alpha, const cudnnTensorDescriptor_t aDesc,
    const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C),
    handle, reduceTensorDesc, indices, indicesSizeInBytes,
    workspace, workspaceSizeInBytes, alpha, aDesc,
    A, beta, cDesc, C)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SET_TENSOR,
    ,
    cudnnSetTensor,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *valuePtr),
    handle, yDesc, y, valuePtr)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SCALE_TENSOR,
    ,
    cudnnScaleTensor,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
    void *y, const void *alpha),
    handle, yDesc, y, alpha)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_TRANSFORM_FILTER,
    ,
    cudnnTransformFilter,
    (cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnFilterDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnFilterDescriptor_t destDesc, void *destData),
    handle, transDesc, alpha, srcDesc,
    srcData, beta, destDesc, destData)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SOFTMAX_FORWARD,
    ,
    cudnnSoftmaxForward,
    (cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, algo, mode, alpha,
    xDesc, x, beta, yDesc,
    y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_POOLING_FORWARD,
    ,
    cudnnPoolingForward,
    (cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y),
    handle, poolingDesc, alpha, xDesc,
    x, beta, yDesc, y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_ACTIVATION_FORWARD,
    ,
    cudnnActivationForward,
    (cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y),
    handle, activationDesc, alpha, xDesc,
    x, beta, yDesc, y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_LRN_CROSS_CHANNEL_FORWARD,
    ,
    cudnnLRNCrossChannelForward,
    (cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, normDesc, lrnMode, alpha,
    xDesc, x, beta, yDesc,
    y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_DIVISIVE_NORMALIZATION_FORWARD,
    ,
    cudnnDivisiveNormalizationForward,
    (cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, void *temp,
    void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y),
    handle, normDesc, mode, alpha,
    xDesc, x, means, temp,
    temp2, beta, yDesc, y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE,
    ,
    cudnnBatchNormalizationForwardInference,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon),
    handle, mode, alpha, beta,
    xDesc, x, yDesc, y,
    bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
    estimatedVariance, epsilon)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_NORMALIZATION_FORWARD_INFERENCE,
    ,
    cudnnNormalizationForwardInference,
    (cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean, const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc,
    void *y, double epsilon,
    int groupCnt),
    handle, mode, normOps, algo,
    alpha, beta, xDesc, x,
    normScaleBiasDesc, normScale, normBias, normMeanVarDesc,
    estimatedMean, estimatedVariance, zDesc, z,
    activationDesc, yDesc, y, epsilon,
    groupCnt)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD,
    ,
    cudnnSpatialTfGridGeneratorForward,
    (cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid),
    handle, stDesc, theta, grid)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SPATIAL_TF_SAMPLER_FORWARD,
    ,
    cudnnSpatialTfSamplerForward,
    (cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *grid,
    const void *beta, cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, stDesc, alpha, xDesc,
    x, grid, beta, yDesc,
    y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_DROPOUT_GET_STATES_SIZE,
    ,
    cudnnDropoutGetStatesSize,
    (cudnnHandle_t handle, size_t *sizeInBytes),
    handle, sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_DROPOUT_FORWARD,
    ,
    cudnnDropoutForward,
    (cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc, const void *x,
    const cudnnTensorDescriptor_t ydesc, void *y,
    void *reserveSpace, size_t reserveSpaceSizeInBytes),
    handle, dropoutDesc, xdesc, x,
    ydesc, y, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_ALGORITHM_SPACE_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetAlgorithmSpaceSize,
    (cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes),
    handle, algoDesc, algoSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SAVE_ALGORITHM,
    CUDNN_DEPRECATED,
    cudnnSaveAlgorithm,
    (cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace, size_t algoSpaceSizeInBytes),
    handle, algoDesc, algoSpace, algoSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RESTORE_ALGORITHM,
    CUDNN_DEPRECATED,
    cudnnRestoreAlgorithm,
    (cudnnHandle_t handle, void *algoSpace,
    size_t algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t algoDesc),
    handle, algoSpace, algoSpaceSizeInBytes, algoDesc)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SOFTMAX_BACKWARD,
    ,
    cudnnSoftmaxBackward,
    (cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, algo, mode, alpha,
    yDesc, y, dyDesc, dy,
    beta, dxDesc, dx)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_POOLING_BACKWARD,
    ,
    cudnnPoolingBackward,
    (cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx),
    handle, poolingDesc, alpha, yDesc,
    y, dyDesc, dy, xDesc,
    x, beta, dxDesc, dx)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_ACTIVATION_BACKWARD,
    ,
    cudnnActivationBackward,
    (cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,
    const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx),
    handle, activationDesc, alpha, yDesc,
    y, dyDesc, dy, xDesc,
    x, beta, dxDesc, dx)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_LRN_CROSS_CHANNEL_BACKWARD,
    ,
    cudnnLRNCrossChannelBackward,
    (cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, normDesc, lrnMode, alpha,
    yDesc, y, dyDesc, dy,
    xDesc, x, beta, dxDesc,
    dx)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_DIVISIVE_NORMALIZATION_BACKWARD,
    ,
    cudnnDivisiveNormalizationBackward,
    (cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *means, const void *dy,
    void *temp, void *temp2,
    const void *beta, const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx, void *dMeans),
    handle, normDesc, mode, alpha,
    xDesc, x, means, dy,
    temp, temp2, beta, dXdMeansDesc,
    dx, dMeans)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE,
    ,
    cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes),
    handle, mode, bnOps, xDesc,
    zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc,
    sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE,
    ,
    cudnnGetBatchNormalizationBackwardExWorkspaceSize,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc, const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes),
    handle, mode, bnOps, xDesc,
    yDesc, dyDesc, dzDesc, dxDesc,
    dBnScaleBiasDesc, activationDesc, sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE,
    ,
    cudnnGetBatchNormalizationTrainingExReserveSpaceSize,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes),
    handle, mode, bnOps, activationDesc,
    xDesc, sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING,
    ,
    cudnnBatchNormalizationForwardTraining,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance),
    handle, mode, alpha, beta,
    xDesc, x, yDesc, y,
    bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor,
    resultRunningMean, resultRunningVariance, epsilon, resultSaveMean,
    resultSaveInvVariance)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX,
    ,
    cudnnBatchNormalizationForwardTrainingEx,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale, const void *bnBias,
    double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, mode, bnOps, alpha,
    beta, xDesc, xData, zDesc,
    zData, yDesc, yData, bnScaleBiasMeanVarDesc,
    bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
    resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance,
    activationDesc, workspace, workSpaceSizeInBytes, reserveSpace,
    reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_BACKWARD,
    ,
    cudnnBatchNormalizationBackward,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
    void *dBnScaleResult, void *dBnBiasResult,
    double epsilon, const void *savedMean,
    const void *savedInvVariance),
    handle, mode, alphaDataDiff, betaDataDiff,
    alphaParamDiff, betaParamDiff, xDesc, x,
    dyDesc, dy, dxDesc, dx,
    dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult,
    epsilon, savedMean, savedInvVariance)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_BACKWARD_EX,
    ,
    cudnnBatchNormalizationBackwardEx,
    (cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t yDesc,
    const void *yData, const cudnnTensorDescriptor_t dyDesc,
    const void *dyData, const cudnnTensorDescriptor_t dzDesc,
    void *dzData, const cudnnTensorDescriptor_t dxDesc,
    void *dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData, const void *bnBiasData,
    void *dBnScaleData, void *dBnBiasData,
    double epsilon, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes),
    handle, mode, bnOps, alphaDataDiff,
    betaDataDiff, alphaParamDiff, betaParamDiff, xDesc,
    xData, yDesc, yData, dyDesc,
    dyData, dzDesc, dzData, dxDesc,
    dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData,
    dBnScaleData, dBnBiasData, epsilon, savedMean,
    savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes,
    reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE,
    ,
    cudnnGetNormalizationForwardTrainingWorkspaceSize,
    (cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt),
    handle, mode, normOps, algo,
    xDesc, zDesc, yDesc, normScaleBiasDesc,
    activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE,
    ,
    cudnnGetNormalizationBackwardWorkspaceSize,
    (cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes, int groupCnt),
    handle, mode, normOps, algo,
    xDesc, yDesc, dyDesc, dzDesc,
    dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc,
    sizeInBytes, groupCnt)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE,
    ,
    cudnnGetNormalizationTrainingReserveSpaceSize,
    (cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes, int groupCnt),
    handle, mode, normOps, algo,
    activationDesc, xDesc, sizeInBytes, groupCnt)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_NORMALIZATION_FORWARD_TRAINING,
    ,
    cudnnNormalizationForwardTraining,
    (cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc, void *resultRunningMean,
    void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t zDesc,
    const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes, int groupCnt),
    handle, mode, normOps, algo,
    alpha, beta, xDesc, xData,
    normScaleBiasDesc, normScale, normBias, exponentialAverageFactor,
    normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon,
    resultSaveMean, resultSaveInvVariance, activationDesc, zDesc,
    zData, yDesc, yData, workspace,
    workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_NORMALIZATION_BACKWARD,
    ,
    cudnnNormalizationBackward,
    (cudnnHandle_t handle, cudnnNormMode_t mode,
    cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc, const void *normScaleData,
    const void *normBiasData, void *dNormScaleData,
    void *dNormBiasData, double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes,
    int groupCnt),
    handle, mode, normOps, algo,
    alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
    xDesc, xData, yDesc, yData,
    dyDesc, dyData, dzDesc, dzData,
    dxDesc, dxData, dNormScaleBiasDesc, normScaleData,
    normBiasData, dNormScaleData, dNormBiasData, epsilon,
    normMeanVarDesc, savedMean, savedInvVariance, activationDesc,
    workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes,
    groupCnt)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD,
    ,
    cudnnSpatialTfGridGeneratorBackward,
    (cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta),
    handle, stDesc, dgrid, dtheta)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SPATIAL_TF_SAMPLER_BACKWARD,
    ,
    cudnnSpatialTfSamplerBackward,
    (cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid,
    const void *betaDgrid, void *dgrid),
    handle, stDesc, alpha, xDesc,
    x, beta, dxDesc, dx,
    alphaDgrid, dyDesc, dy, grid,
    betaDgrid, dgrid)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_DROPOUT_BACKWARD,
    ,
    cudnnDropoutBackward,
    (cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc, const void *dy,
    const cudnnTensorDescriptor_t dxdesc, void *dx,
    void *reserveSpace, size_t reserveSpaceSizeInBytes),
    handle, dropoutDesc, dydesc, dy,
    dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SET_RNN_DESCRIPTOR_V6,
    CUDNN_DEPRECATED,
    cudnnSetRNNDescriptor_v6,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize, const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec),
    handle, rnnDesc, hiddenSize, numLayers,
    dropoutDesc, inputMode, direction, cellMode,
    algo, mathPrec)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_DESCRIPTOR_V6,
    CUDNN_DEPRECATED,
    cudnnGetRNNDescriptor_v6,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize, int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc, cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction, cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec),
    handle, rnnDesc, hiddenSize, numLayers,
    dropoutDesc, inputMode, direction, cellMode,
    algo, mathPrec)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_SET_CLIP,
    CUDNN_DEPRECATED,
    cudnnRNNSetClip,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt,
    double lclip, double rclip),
    handle, rnnDesc, clipMode, clipNanOpt,
    lclip, rclip)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_GET_CLIP,
    CUDNN_DEPRECATED,
    cudnnRNNGetClip,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt,
    double *lclip, double *rclip),
    handle, rnnDesc, clipMode, clipNanOpt,
    lclip, rclip)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SET_RNN_PROJECTION_LAYERS,
    CUDNN_DEPRECATED,
    cudnnSetRNNProjectionLayers,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize, const int outProjSize),
    handle, rnnDesc, recProjSize, outProjSize)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_PROJECTION_LAYERS,
    CUDNN_DEPRECATED,
    cudnnGetRNNProjectionLayers,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize, int *outProjSize),
    handle, rnnDesc, recProjSize, outProjSize)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_BUILD_RNN_DYNAMIC,
    ,
    cudnnBuildRNNDynamic,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int miniBatch),
    handle, rnnDesc, miniBatch)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_WORKSPACE_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetRNNWorkspaceSize,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_TRAINING_RESERVE_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetRNNTrainingReserveSize,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_TEMP_SPACE_SIZES,
    ,
    cudnnGetRNNTempSpaceSizes,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode, cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize, size_t *reserveSpaceSize),
    handle, rnnDesc, fMode, xDesc,
    workSpaceSize, reserveSpaceSize)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_PARAMS_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetRNNParamsSize,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
    cudnnDataType_t dataType),
    handle, rnnDesc, xDesc, sizeInBytes,
    dataType)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_WEIGHT_SPACE_SIZE,
    ,
    cudnnGetRNNWeightSpaceSize,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize),
    handle, rnnDesc, weightSpaceSize)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS,
    CUDNN_DEPRECATED,
    cudnnGetRNNLinLayerMatrixParams,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat),
    handle, rnnDesc, pseudoLayer, xDesc,
    wDesc, w, linLayerID, linLayerMatDesc,
    linLayerMat)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS,
    CUDNN_DEPRECATED,
    cudnnGetRNNLinLayerBiasParams,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias),
    handle, rnnDesc, pseudoLayer, xDesc,
    wDesc, w, linLayerID, linLayerBiasDesc,
    linLayerBias)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_WEIGHT_PARAMS,
    ,
    cudnnGetRNNWeightParams,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer, size_t weightSpaceSize,
    const void *weightSpace, int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc, void **mAddr,
    cudnnTensorDescriptor_t bDesc, void **bAddr),
    handle, rnnDesc, pseudoLayer, weightSpaceSize,
    weightSpace, linLayerID, mDesc, mAddr,
    bDesc, bAddr)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_FORWARD_INFERENCE,
    CUDNN_DEPRECATED,
    cudnnRNNForwardInference,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, workSpace, workSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_FORWARD_INFERENCE_EX,
    CUDNN_DEPRECATED,
    cudnnRNNForwardInferenceEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes),
    handle, rnnDesc, xDesc, x,
    hxDesc, hx, cxDesc, cx,
    wDesc, w, yDesc, y,
    hyDesc, hy, cyDesc, cy,
    kDesc, keys, cDesc, cAttn,
    iDesc, iAttn, qDesc, queries,
    workSpace, workSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_FORWARD,
    ,
    cudnnRNNForward,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnRNNDataDescriptor_t yDesc, void *y,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    void *hy, cudnnTensorDescriptor_t cDesc,
    const void *cx, void *cy,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace),
    handle, rnnDesc, fwdMode, devSeqLengths,
    xDesc, x, yDesc, y,
    hDesc, hx, hy, cDesc,
    cx, cy, weightSpaceSize, weightSpace,
    workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnSetRNNAlgorithmDescriptor,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc),
    handle, rnnDesc, algoDesc)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNForwardInferenceAlgorithmMaxCount,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNForwardInferenceAlgorithmEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
    perfResults, workspace, workSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS,
    ,
    cudnnGetMultiHeadAttnBuffers,
    (cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes),
    handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes,
    reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS,
    ,
    cudnnGetMultiHeadAttnWeights,
    (cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
    const void *weights, cudnnTensorDescriptor_t wDesc,
    void **wAddr),
    handle, attnDesc, wKind, weightSizeInBytes,
    weights, wDesc, wAddr)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_MULTI_HEAD_ATTN_FORWARD,
    ,
    cudnnMultiHeadAttnForward,
    (cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    int currIdx, const int loWinIdx[],
    const int hiWinIdx[], const int devSeqLengthsQO[],
    const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out,
    size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace),
    handle, attnDesc, currIdx, loWinIdx,
    hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc,
    queries, residuals, kDesc, keys,
    vDesc, values, oDesc, out,
    weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace,
    reserveSpaceSizeInBytes, reserveSpace)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_FORWARD_TRAINING,
    CUDNN_DEPRECATED,
    cudnnRNNForwardTraining,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, workSpace, workSpaceSizeInBytes, reserveSpace,
    reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_BACKWARD_DATA,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardData,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, yDesc,
    y, dyDesc, dy, dhyDesc,
    dhy, dcyDesc, dcy, wDesc,
    w, hxDesc, hx, cxDesc,
    cx, dxDesc, dx, dhxDesc,
    dhx, dcxDesc, dcx, workSpace,
    workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_BACKWARD_DATA_V8,
    ,
    cudnnRNNBackwardData_v8,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t yDesc,
    const void *y, const void *dy,
    cudnnRNNDataDescriptor_t xDesc, void *dx,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    const void *dhy, void *dhx,
    cudnnTensorDescriptor_t cDesc, const void *cx,
    const void *dcy, void *dcx,
    size_t weightSpaceSize, const void *weightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace),
    handle, rnnDesc, devSeqLengths, yDesc,
    y, dy, xDesc, dx,
    hDesc, hx, dhy, dhx,
    cDesc, cx, dcy, dcx,
    weightSpaceSize, weightSpace, workSpaceSize, workSpace,
    reserveSpaceSize, reserveSpace)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_BACKWARD_WEIGHTS,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardWeights,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const void *workSpace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, yDesc,
    y, workSpace, workSpaceSizeInBytes, dwDesc,
    dw, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_BACKWARD_WEIGHTS_V8,
    ,
    cudnnRNNBackwardWeights_v8,
    (cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    cudnnRNNDataDescriptor_t yDesc, const void *y,
    size_t weightSpaceSize, void *dweightSpace,
    size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace),
    handle, rnnDesc, addGrad, devSeqLengths,
    xDesc, x, hDesc, hx,
    yDesc, y, weightSpaceSize, dweightSpace,
    workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_FORWARD_TRAINING_EX,
    CUDNN_DEPRECATED,
    cudnnRNNForwardTrainingEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, xDesc, x,
    hxDesc, hx, cxDesc, cx,
    wDesc, w, yDesc, y,
    hyDesc, hy, cyDesc, cy,
    kDesc, keys, cDesc, cAttn,
    iDesc, iAttn, qDesc, queries,
    workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_BACKWARD_DATA_EX,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardDataEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, void *dkeys,
    void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, yDesc, y,
    dyDesc, dy, dcDesc, dcAttn,
    dhyDesc, dhy, dcyDesc, dcy,
    wDesc, w, hxDesc, hx,
    cxDesc, cx, dxDesc, dx,
    dhxDesc, dhx, dcxDesc, dcx,
    dkDesc, dkeys, workSpace, workSpaceSizeInBytes,
    reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_RNN_BACKWARD_WEIGHTS_EX,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardWeightsEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    void *workSpace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    void *reserveSpace, size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, xDesc, x,
    hxDesc, hx, yDesc, y,
    workSpace, workSpaceSizeInBytes, dwDesc, dw,
    reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNForwardTrainingAlgorithmMaxCount,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNForwardTrainingAlgorithmEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc,
    void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
    perfResults, workspace, workSpaceSizeInBytes, reserveSpace,
    reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNBackwardDataAlgorithmMaxCount,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNBackwardDataAlgorithmEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc,
    void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, yDesc,
    y, dyDesc, dy, dhyDesc,
    dhy, dcyDesc, dcy, wDesc,
    w, hxDesc, hx, cxDesc,
    cx, dxDesc, dx, dhxDesc,
    dhx, dcxDesc, dcx, findIntensity,
    requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
    workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNBackwardWeightsAlgorithmMaxCount,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNBackwardWeightsAlgorithmEx,
    (cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, yDesc,
    y, findIntensity, requestedAlgoCount, returnedAlgoCount,
    perfResults, workspace, workSpaceSizeInBytes, dwDesc,
    dw, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA,
    ,
    cudnnMultiHeadAttnBackwardData,
    (cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[], const int hiWinIdx[],
    const int devSeqLengthsDQDO[], const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys, const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes,
    void *workSpace, size_t reserveSpaceSizeInBytes,
    void *reserveSpace),
    handle, attnDesc, loWinIdx, hiWinIdx,
    devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout,
    dqDesc, dqueries, queries, dkDesc,
    dkeys, keys, dvDesc, dvalues,
    values, weightSizeInBytes, weights, workSpaceSizeInBytes,
    workSpace, reserveSpaceSizeInBytes, reserveSpace)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS,
    ,
    cudnnMultiHeadAttnBackwardWeights,
    (cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const cudnnSeqDataDescriptor_t kDesc,
    const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, size_t weightSizeInBytes,
    const void *weights, void *dweights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace),
    handle, attnDesc, addGrad, qDesc,
    queries, kDesc, keys, vDesc,
    values, doDesc, dout, weightSizeInBytes,
    weights, dweights, workSpaceSizeInBytes, workSpace,
    reserveSpaceSizeInBytes, reserveSpace)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_CTC_LOSS,
    ,
    cudnnCTCLoss,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int hostLabels[],
    const int hostLabelLengths[], const int hostInputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace,
    size_t workSpaceSizeInBytes),
    handle, probsDesc, probs, hostLabels,
    hostLabelLengths, hostInputLengths, costs, gradientsDesc,
    gradients, algo, ctcLossDesc, workspace,
    workSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_CTC_LOSS_V8,
    ,
    cudnnCTCLoss_v8,
    (cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const void *probs, const int labels[],
    const int labelLengths[], const int inputLengths[],
    void *costs, const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients, size_t workSpaceSizeInBytes,
    void *workspace),
    handle, algo, ctcLossDesc, probsDesc,
    probs, labels, labelLengths, inputLengths,
    costs, gradientsDesc, gradients, workSpaceSizeInBytes,
    workspace)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE,
    ,
    cudnnGetCTCLossWorkspaceSize,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths,
    cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes),
    handle, probsDesc, gradientsDesc, labels,
    labelLengths, inputLengths, algo, ctcLossDesc,
    sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8,
    ,
    cudnnGetCTCLossWorkspaceSize_v8,
    (cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, size_t *sizeInBytes),
    handle, algo, ctcLossDesc, probsDesc,
    gradientsDesc, sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT,
    ,
    cudnnGetConvolutionForwardAlgorithmMaxCount,
    (cudnnHandle_t handle, int *count),
    handle, count)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7,
    ,
    cudnnGetConvolutionForwardAlgorithm_v7,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults),
    handle, srcDesc, filterDesc, convDesc,
    destDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM,
    ,
    cudnnFindConvolutionForwardAlgorithm,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults),
    handle, xDesc, wDesc, convDesc,
    yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX,
    ,
    cudnnFindConvolutionForwardAlgorithmEx,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, xDesc, x, wDesc,
    w, convDesc, yDesc, y,
    requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
    workSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_IM_2_COL,
    ,
    cudnnIm2Col,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, void *colBuffer),
    handle, xDesc, x, wDesc,
    convDesc, colBuffer)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_REORDER_FILTER_AND_BIAS,
    ,
    cudnnReorderFilterAndBias,
    (cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias,
    const void *biasData, void *reorderedBiasData),
    handle, filterDesc, reorderType, filterData,
    reorderedFilterData, reorderBias, biasData, reorderedBiasData)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE,
    ,
    cudnnGetConvolutionForwardWorkspaceSize,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes),
    handle, xDesc, wDesc, convDesc,
    yDesc, algo, sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_CONVOLUTION_FORWARD,
    ,
    cudnnConvolutionForward,
    (cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, alpha, xDesc, x,
    wDesc, w, convDesc, algo,
    workSpace, workSpaceSizeInBytes, beta, yDesc,
    y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD,
    ,
    cudnnConvolutionBiasActivationForward,
    (cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *alpha2, const cudnnTensorDescriptor_t zDesc,
    const void *z, const cudnnTensorDescriptor_t biasDesc,
    const void *bias, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y),
    handle, alpha1, xDesc, x,
    wDesc, w, convDesc, algo,
    workSpace, workSpaceSizeInBytes, alpha2, zDesc,
    z, biasDesc, bias, activationDesc,
    yDesc, y)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT,
    ,
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount,
    (cudnnHandle_t handle, int *count),
    handle, count)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM,
    ,
    cudnnFindConvolutionBackwardDataAlgorithm,
    (cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults),
    handle, wDesc, dyDesc, convDesc,
    dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX,
    ,
    cudnnFindConvolutionBackwardDataAlgorithmEx,
    (cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, wDesc, w, dyDesc,
    dy, convDesc, dxDesc, dx,
    requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
    workSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7,
    ,
    cudnnGetConvolutionBackwardDataAlgorithm_v7,
    (cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults),
    handle, filterDesc, diffDesc, convDesc,
    gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE,
    ,
    cudnnGetConvolutionBackwardDataWorkspaceSize,
    (cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes),
    handle, wDesc, dyDesc, convDesc,
    dxDesc, algo, sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_CONVOLUTION_BACKWARD_DATA,
    ,
    cudnnConvolutionBackwardData,
    (cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, alpha, wDesc, w,
    dyDesc, dy, convDesc, algo,
    workSpace, workSpaceSizeInBytes, beta, dxDesc,
    dx)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS,
    ,
    cudnnGetFoldedConvBackwardDataDescriptors,
    (const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc, cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc, cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc, cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc, cudnnTensorTransformDescriptor_t gradUnfoldTransDesc),
    handle, filterDesc, diffDesc, convDesc,
    gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc,
    foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc,
    gradFoldTransDesc, gradUnfoldTransDesc)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT,
    ,
    cudnnGetConvolutionBackwardFilterAlgorithmMaxCount,
    (cudnnHandle_t handle, int *count),
    handle, count)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM,
    ,
    cudnnFindConvolutionBackwardFilterAlgorithm,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults),
    handle, xDesc, dyDesc, convDesc,
    dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX,
    ,
    cudnnFindConvolutionBackwardFilterAlgorithmEx,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *y, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, xDesc, x, dyDesc,
    y, convDesc, dwDesc, dw,
    requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
    workSpaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7,
    ,
    cudnnGetConvolutionBackwardFilterAlgorithm_v7,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults),
    handle, srcDesc, diffDesc, convDesc,
    gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE,
    ,
    cudnnGetConvolutionBackwardFilterWorkspaceSize,
    (cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes),
    handle, xDesc, dyDesc, convDesc,
    gradDesc, algo, sizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_CONVOLUTION_BACKWARD_FILTER,
    ,
    cudnnConvolutionBackwardFilter,
    (cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta, const cudnnFilterDescriptor_t dwDesc,
    void *dw),
    handle, alpha, xDesc, x,
    dyDesc, dy, convDesc, algo,
    workSpace, workSpaceSizeInBytes, beta, dwDesc,
    dw)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_CONVOLUTION_BACKWARD_BIAS,
    ,
    cudnnConvolutionBackwardBias,
    (cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dbDesc,
    void *db),
    handle, alpha, dyDesc, dy,
    beta, dbDesc, db)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_MAKE_FUSED_OPS_PLAN,
    ,
    cudnnMakeFusedOpsPlan,
    (cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack, size_t *workspaceSizeInBytes),
    handle, plan, constPack, workspaceSizeInBytes)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_FUSED_OPS_EXECUTE,
    ,
    cudnnFusedOpsExecute,
    (cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack),
    handle, plan, varPack)

CUDNN_HANDLE_HOOK_GEN(
    CUDNN_BACKEND_EXECUTE,
    ,
    cudnnBackendExecute,
    (cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack),
    handle, executionPlan, variantPack)

CUDNN_HOOK_GEN(
    CUDNN_GET_PROPERTY,
    ,
    cudnnGetProperty,
    (libraryPropertyType type, int *value),
    type, value)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_TENSOR_DESCRIPTOR,
    ,
    cudnnCreateTensorDescriptor,
    (cudnnTensorDescriptor_t *tensorDesc),
    tensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_4D_DESCRIPTOR,
    ,
    cudnnSetTensor4dDescriptor,
    (cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int n,
    int c, int h,
    int w),
    tensorDesc, format, dataType, n,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX,
    ,
    cudnnSetTensor4dDescriptorEx,
    (cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int n, int c,
    int h, int w,
    int nStride, int cStride,
    int hStride, int wStride),
    tensorDesc, dataType, n, c,
    h, w, nStride, cStride,
    hStride, wStride)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_4D_DESCRIPTOR,
    ,
    cudnnGetTensor4dDescriptor,
    (const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType,
    int *n, int *c,
    int *h, int *w,
    int *nStride, int *cStride,
    int *hStride, int *wStride),
    tensorDesc, dataType, n, c,
    h, w, nStride, cStride,
    hStride, wStride)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_ND_DESCRIPTOR,
    ,
    cudnnSetTensorNdDescriptor,
    (cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const int strideA[]),
    tensorDesc, dataType, nbDims, dimA,
    strideA)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX,
    ,
    cudnnSetTensorNdDescriptorEx,
    (cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims,
    const int dimA[]),
    tensorDesc, format, dataType, nbDims,
    dimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_ND_DESCRIPTOR,
    ,
    cudnnGetTensorNdDescriptor,
    (const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, int *nbDims,
    int dimA[], int strideA[]),
    tensorDesc, nbDimsRequested, dataType, nbDims,
    dimA, strideA)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_SIZE_IN_BYTES,
    ,
    cudnnGetTensorSizeInBytes,
    (const cudnnTensorDescriptor_t tensorDesc, size_t *size),
    tensorDesc, size)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_TENSOR_DESCRIPTOR,
    ,
    cudnnDestroyTensorDescriptor,
    (cudnnTensorDescriptor_t tensorDesc),
    tensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_INIT_TRANSFORM_DEST,
    ,
    cudnnInitTransformDest,
    (const cudnnTensorTransformDescriptor_t transformDesc, const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes),
    transformDesc, srcDesc, destDesc, destSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR,
    ,
    cudnnCreateTensorTransformDescriptor,
    (cudnnTensorTransformDescriptor_t *transformDesc),
    transformDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR,
    ,
    cudnnSetTensorTransformDescriptor,
    (cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t padBeforeA[],
    const int32_t padAfterA[], const uint32_t foldA[],
    const cudnnFoldingDirection_t direction),
    transformDesc, nbDims, destFormat, padBeforeA,
    padAfterA, foldA, direction)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR,
    ,
    cudnnGetTensorTransformDescriptor,
    (cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t padBeforeA[],
    int32_t padAfterA[], uint32_t foldA[],
    cudnnFoldingDirection_t *direction),
    transformDesc, nbDimsRequested, destFormat, padBeforeA,
    padAfterA, foldA, direction)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR,
    ,
    cudnnDestroyTensorTransformDescriptor,
    (cudnnTensorTransformDescriptor_t transformDesc),
    transformDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_OP_TENSOR_DESCRIPTOR,
    ,
    cudnnCreateOpTensorDescriptor,
    (cudnnOpTensorDescriptor_t *opTensorDesc),
    opTensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_OP_TENSOR_DESCRIPTOR,
    ,
    cudnnSetOpTensorDescriptor,
    (cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt),
    opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)

CUDNN_HOOK_GEN(
    CUDNN_GET_OP_TENSOR_DESCRIPTOR,
    ,
    cudnnGetOpTensorDescriptor,
    (const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt),
    opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR,
    ,
    cudnnDestroyOpTensorDescriptor,
    (cudnnOpTensorDescriptor_t opTensorDesc),
    opTensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR,
    ,
    cudnnCreateReduceTensorDescriptor,
    (cudnnReduceTensorDescriptor_t *reduceTensorDesc),
    reduceTensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR,
    ,
    cudnnSetReduceTensorDescriptor,
    (cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType),
    reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
    reduceTensorIndices, reduceTensorIndicesType)

CUDNN_HOOK_GEN(
    CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR,
    ,
    cudnnGetReduceTensorDescriptor,
    (const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType),
    reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
    reduceTensorIndices, reduceTensorIndicesType)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR,
    ,
    cudnnDestroyReduceTensorDescriptor,
    (cudnnReduceTensorDescriptor_t reduceTensorDesc),
    reduceTensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_FILTER_DESCRIPTOR,
    ,
    cudnnCreateFilterDescriptor,
    (cudnnFilterDescriptor_t *filterDesc),
    filterDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_FILTER_4D_DESCRIPTOR,
    ,
    cudnnSetFilter4dDescriptor,
    (cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int k,
    int c, int h,
    int w),
    filterDesc, dataType, format, k,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_GET_FILTER_4D_DESCRIPTOR,
    ,
    cudnnGetFilter4dDescriptor,
    (const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *k,
    int *c, int *h,
    int *w),
    filterDesc, dataType, format, k,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_SET_FILTER_ND_DESCRIPTOR,
    ,
    cudnnSetFilterNdDescriptor,
    (cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int nbDims,
    const int filterDimA[]),
    filterDesc, dataType, format, nbDims,
    filterDimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_FILTER_ND_DESCRIPTOR,
    ,
    cudnnGetFilterNdDescriptor,
    (const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, cudnnTensorFormat_t *format,
    int *nbDims, int filterDimA[]),
    filterDesc, nbDimsRequested, dataType, format,
    nbDims, filterDimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_FILTER_SIZE_IN_BYTES,
    ,
    cudnnGetFilterSizeInBytes,
    (const cudnnFilterDescriptor_t filterDesc, size_t *size),
    filterDesc, size)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_FILTER_DESCRIPTOR,
    ,
    cudnnDestroyFilterDescriptor,
    (cudnnFilterDescriptor_t filterDesc),
    filterDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_POOLING_DESCRIPTOR,
    ,
    cudnnCreatePoolingDescriptor,
    (cudnnPoolingDescriptor_t *poolingDesc),
    poolingDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_POOLING_2D_DESCRIPTOR,
    ,
    cudnnSetPooling2dDescriptor,
    (cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
    int windowWidth, int verticalPadding,
    int horizontalPadding, int verticalStride,
    int horizontalStride),
    poolingDesc, mode, maxpoolingNanOpt, windowHeight,
    windowWidth, verticalPadding, horizontalPadding, verticalStride,
    horizontalStride)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_2D_DESCRIPTOR,
    ,
    cudnnGetPooling2dDescriptor,
    (const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding,
    int *horizontalPadding, int *verticalStride,
    int *horizontalStride),
    poolingDesc, mode, maxpoolingNanOpt, windowHeight,
    windowWidth, verticalPadding, horizontalPadding, verticalStride,
    horizontalStride)

CUDNN_HOOK_GEN(
    CUDNN_SET_POOLING_ND_DESCRIPTOR,
    ,
    cudnnSetPoolingNdDescriptor,
    (cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[],
    const int strideA[]),
    poolingDesc, mode, maxpoolingNanOpt, nbDims,
    windowDimA, paddingA, strideA)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_ND_DESCRIPTOR,
    ,
    cudnnGetPoolingNdDescriptor,
    (const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[],
    int paddingA[], int strideA[]),
    poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt,
    nbDims, windowDimA, paddingA, strideA)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetPoolingNdForwardOutputDim,
    (const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims, int outputTensorDimA[]),
    poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetPooling2dForwardOutputDim,
    (const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int *n, int *c,
    int *h, int *w),
    poolingDesc, inputTensorDesc, n, c,
    h, w)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_POOLING_DESCRIPTOR,
    ,
    cudnnDestroyPoolingDescriptor,
    (cudnnPoolingDescriptor_t poolingDesc),
    poolingDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_ACTIVATION_DESCRIPTOR,
    ,
    cudnnCreateActivationDescriptor,
    (cudnnActivationDescriptor_t *activationDesc),
    activationDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_ACTIVATION_DESCRIPTOR,
    ,
    cudnnSetActivationDescriptor,
    (cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt, double coef),
    activationDesc, mode, reluNanOpt, coef)

CUDNN_HOOK_GEN(
    CUDNN_GET_ACTIVATION_DESCRIPTOR,
    ,
    cudnnGetActivationDescriptor,
    (const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt, double *coef),
    activationDesc, mode, reluNanOpt, coef)

CUDNN_HOOK_GEN(
    CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA,
    ,
    cudnnSetActivationDescriptorSwishBeta,
    (cudnnActivationDescriptor_t activationDesc, double swish_beta),
    activationDesc, swish_beta)

CUDNN_HOOK_GEN(
    CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA,
    ,
    cudnnGetActivationDescriptorSwishBeta,
    (cudnnActivationDescriptor_t activationDesc, double *swish_beta),
    activationDesc, swish_beta)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_ACTIVATION_DESCRIPTOR,
    ,
    cudnnDestroyActivationDescriptor,
    (cudnnActivationDescriptor_t activationDesc),
    activationDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_LRN_DESCRIPTOR,
    ,
    cudnnCreateLRNDescriptor,
    (cudnnLRNDescriptor_t *normDesc),
    normDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_LRN_DESCRIPTOR,
    ,
    cudnnSetLRNDescriptor,
    (cudnnLRNDescriptor_t normDesc, unsigned lrnN,
    double lrnAlpha, double lrnBeta,
    double lrnK),
    normDesc, lrnN, lrnAlpha, lrnBeta,
    lrnK)

CUDNN_HOOK_GEN(
    CUDNN_GET_LRN_DESCRIPTOR,
    ,
    cudnnGetLRNDescriptor,
    (cudnnLRNDescriptor_t normDesc, unsigned *lrnN,
    double *lrnAlpha, double *lrnBeta,
    double *lrnK),
    normDesc, lrnN, lrnAlpha, lrnBeta,
    lrnK)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_LRN_DESCRIPTOR,
    ,
    cudnnDestroyLRNDescriptor,
    (cudnnLRNDescriptor_t lrnDesc),
    lrnDesc)

CUDNN_HOOK_GEN(
    CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR,
    ,
    cudnnDeriveBNTensorDescriptor,
    (cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode),
    derivedBnDesc, xDesc, mode)

CUDNN_HOOK_GEN(
    CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR,
    ,
    cudnnDeriveNormTensorDescriptor,
    (cudnnTensorDescriptor_t derivedNormScaleBiasDesc, cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc, cudnnNormMode_t mode,
    int groupCnt),
    derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode,
    groupCnt)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR,
    ,
    cudnnCreateSpatialTransformerDescriptor,
    (cudnnSpatialTransformerDescriptor_t *stDesc),
    stDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR,
    ,
    cudnnSetSpatialTransformerNdDescriptor,
    (cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims,
    const int dimA[]),
    stDesc, samplerType, dataType, nbDims,
    dimA)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR,
    ,
    cudnnDestroySpatialTransformerDescriptor,
    (cudnnSpatialTransformerDescriptor_t stDesc),
    stDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_DROPOUT_DESCRIPTOR,
    ,
    cudnnCreateDropoutDescriptor,
    (cudnnDropoutDescriptor_t *dropoutDesc),
    dropoutDesc)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_DROPOUT_DESCRIPTOR,
    ,
    cudnnDestroyDropoutDescriptor,
    (cudnnDropoutDescriptor_t dropoutDesc),
    dropoutDesc)

CUDNN_HOOK_GEN(
    CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE,
    ,
    cudnnDropoutGetReserveSpaceSize,
    (cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes),
    xdesc, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_SET_DROPOUT_DESCRIPTOR,
    ,
    cudnnSetDropoutDescriptor,
    (cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed),
    dropoutDesc, handle, dropout, states,
    stateSizeInBytes, seed)

CUDNN_HOOK_GEN(
    CUDNN_RESTORE_DROPOUT_DESCRIPTOR,
    ,
    cudnnRestoreDropoutDescriptor,
    (cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed),
    dropoutDesc, handle, dropout, states,
    stateSizeInBytes, seed)

CUDNN_HOOK_GEN(
    CUDNN_GET_DROPOUT_DESCRIPTOR,
    ,
    cudnnGetDropoutDescriptor,
    (cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
    float *dropout, void **states,
    unsigned long long *seed),
    dropoutDesc, handle, dropout, states,
    seed)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnCreateAlgorithmDescriptor,
    (cudnnAlgorithmDescriptor_t *algoDesc),
    algoDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnSetAlgorithmDescriptor,
    (cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm),
    algoDesc, algorithm)

CUDNN_HOOK_GEN(
    CUDNN_GET_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnGetAlgorithmDescriptor,
    (const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm),
    algoDesc, algorithm)

CUDNN_HOOK_GEN(
    CUDNN_COPY_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnCopyAlgorithmDescriptor,
    (const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest),
    src, dest)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnDestroyAlgorithmDescriptor,
    (cudnnAlgorithmDescriptor_t algoDesc),
    algoDesc)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_ALGORITHM_PERFORMANCE,
    CUDNN_DEPRECATED,
    cudnnCreateAlgorithmPerformance,
    (cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate),
    algoPerf, numberToCreate)

CUDNN_HOOK_GEN(
    CUDNN_SET_ALGORITHM_PERFORMANCE,
    CUDNN_DEPRECATED,
    cudnnSetAlgorithmPerformance,
    (cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status, float time,
    size_t memory),
    algoPerf, algoDesc, status, time,
    memory)

CUDNN_HOOK_GEN(
    CUDNN_GET_ALGORITHM_PERFORMANCE,
    CUDNN_DEPRECATED,
    cudnnGetAlgorithmPerformance,
    (const cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status, float *time,
    size_t *memory),
    algoPerf, algoDesc, status, time,
    memory)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_ALGORITHM_PERFORMANCE,
    CUDNN_DEPRECATED,
    cudnnDestroyAlgorithmPerformance,
    (cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy),
    algoPerf, numberToDestroy)

CUDNN_HOOK_GEN(
    CUDNN_SET_CALLBACK,
    ,
    cudnnSetCallback,
    (unsigned mask, void *udata,
    cudnnCallback_t fptr),
    mask, udata, fptr)

CUDNN_HOOK_GEN(
    CUDNN_GET_CALLBACK,
    ,
    cudnnGetCallback,
    (unsigned *mask, void **udata,
    cudnnCallback_t *fptr),
    mask, udata, fptr)

CUDNN_HOOK_GEN(
    CUDNN_OPS_INFER_VERSION_CHECK,
    ,
    cudnnOpsInferVersionCheck,
    (),
)

CUDNN_HOOK_GEN(
    CUDNN_OPS_TRAIN_VERSION_CHECK,
    ,
    cudnnOpsTrainVersionCheck,
    (),
)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_RNN_DESCRIPTOR,
    ,
    cudnnCreateRNNDescriptor,
    (cudnnRNNDescriptor_t *rnnDesc),
    rnnDesc)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_RNN_DESCRIPTOR,
    ,
    cudnnDestroyRNNDescriptor,
    (cudnnRNNDescriptor_t rnnDesc),
    rnnDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_DESCRIPTOR_V8,
    ,
    cudnnSetRNNDescriptor_v8,
    (cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType, cudnnDataType_t mathPrec,
    cudnnMathType_t mathType, int32_t inputSize,
    int32_t hiddenSize, int32_t projSize,
    int32_t numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags),
    rnnDesc, algo, cellMode, biasMode,
    dirMode, inputMode, dataType, mathPrec,
    mathType, inputSize, hiddenSize, projSize,
    numLayers, dropoutDesc, auxFlags)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_DESCRIPTOR_V8,
    ,
    cudnnGetRNNDescriptor_v8,
    (cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode, cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode, cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType, cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType, int32_t *inputSize,
    int32_t *hiddenSize, int32_t *projSize,
    int32_t *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags),
    rnnDesc, algo, cellMode, biasMode,
    dirMode, inputMode, dataType, mathPrec,
    mathType, inputSize, hiddenSize, projSize,
    numLayers, dropoutDesc, auxFlags)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_MATRIX_MATH_TYPE,
    CUDNN_DEPRECATED,
    cudnnSetRNNMatrixMathType,
    (cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType),
    rnnDesc, mType)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_MATRIX_MATH_TYPE,
    CUDNN_DEPRECATED,
    cudnnGetRNNMatrixMathType,
    (cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType),
    rnnDesc, mType)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_BIAS_MODE,
    CUDNN_DEPRECATED,
    cudnnSetRNNBiasMode,
    (cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode),
    rnnDesc, biasMode)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_BIAS_MODE,
    CUDNN_DEPRECATED,
    cudnnGetRNNBiasMode,
    (cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode),
    rnnDesc, biasMode)

CUDNN_HOOK_GEN(
    CUDNN_RNN_SET_CLIP_V8,
    ,
    cudnnRNNSetClip_v8,
    (cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt, double lclip,
    double rclip),
    rnnDesc, clipMode, clipNanOpt, lclip,
    rclip)

CUDNN_HOOK_GEN(
    CUDNN_RNN_GET_CLIP_V8,
    ,
    cudnnRNNGetClip_v8,
    (cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt, double *lclip,
    double *rclip),
    rnnDesc, clipMode, clipNanOpt, lclip,
    rclip)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_PERSISTENT_RNN_PLAN,
    CUDNN_DEPRECATED,
    cudnnCreatePersistentRNNPlan,
    (cudnnRNNDescriptor_t rnnDesc, const int minibatch,
    const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan),
    rnnDesc, minibatch, dataType, plan)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_PERSISTENT_RNN_PLAN,
    CUDNN_DEPRECATED,
    cudnnDestroyPersistentRNNPlan,
    (cudnnPersistentRNNPlan_t plan),
    plan)

CUDNN_HOOK_GEN(
    CUDNN_SET_PERSISTENT_RNN_PLAN,
    CUDNN_DEPRECATED,
    cudnnSetPersistentRNNPlan,
    (cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan),
    rnnDesc, plan)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_PADDING_MODE,
    CUDNN_DEPRECATED,
    cudnnSetRNNPaddingMode,
    (cudnnRNNDescriptor_t rnnDesc, unsigned paddingMode),
    rnnDesc, paddingMode)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_PADDING_MODE,
    CUDNN_DEPRECATED,
    cudnnGetRNNPaddingMode,
    (cudnnRNNDescriptor_t rnnDesc, unsigned *paddingMode),
    rnnDesc, paddingMode)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_RNN_DATA_DESCRIPTOR,
    ,
    cudnnCreateRNNDataDescriptor,
    (cudnnRNNDataDescriptor_t *rnnDataDesc),
    rnnDataDesc)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_RNN_DATA_DESCRIPTOR,
    ,
    cudnnDestroyRNNDataDescriptor,
    (cudnnRNNDataDescriptor_t rnnDataDesc),
    rnnDataDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_DATA_DESCRIPTOR,
    ,
    cudnnSetRNNDataDescriptor,
    (cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout, int maxSeqLength,
    int batchSize, int vectorSize,
    const int seqLengthArray[], void *paddingFill),
    rnnDataDesc, dataType, layout, maxSeqLength,
    batchSize, vectorSize, seqLengthArray, paddingFill)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_DATA_DESCRIPTOR,
    ,
    cudnnGetRNNDataDescriptor,
    (cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout, int *maxSeqLength,
    int *batchSize, int *vectorSize,
    int arrayLengthRequested, int seqLengthArray[],
    void *paddingFill),
    rnnDataDesc, dataType, layout, maxSeqLength,
    batchSize, vectorSize, arrayLengthRequested, seqLengthArray,
    paddingFill)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_SEQ_DATA_DESCRIPTOR,
    ,
    cudnnCreateSeqDataDescriptor,
    (cudnnSeqDataDescriptor_t *seqDataDesc),
    seqDataDesc)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR,
    ,
    cudnnDestroySeqDataDescriptor,
    (cudnnSeqDataDescriptor_t seqDataDesc),
    seqDataDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_SEQ_DATA_DESCRIPTOR,
    ,
    cudnnSetSeqDataDescriptor,
    (cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType,
    int nbDims, const int dimA[],
    const cudnnSeqDataAxis_t axes[], size_t seqLengthArraySize,
    const int seqLengthArray[], void *paddingFill),
    seqDataDesc, dataType, nbDims, dimA,
    axes, seqLengthArraySize, seqLengthArray, paddingFill)

CUDNN_HOOK_GEN(
    CUDNN_GET_SEQ_DATA_DESCRIPTOR,
    ,
    cudnnGetSeqDataDescriptor,
    (const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType,
    int *nbDims, int nbDimsRequested,
    int dimA[], cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize, size_t seqLengthSizeRequested,
    int seqLengthArray[], void *paddingFill),
    seqDataDesc, dataType, nbDims, nbDimsRequested,
    dimA, axes, seqLengthArraySize, seqLengthSizeRequested,
    seqLengthArray, paddingFill)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_ATTN_DESCRIPTOR,
    ,
    cudnnCreateAttnDescriptor,
    (cudnnAttnDescriptor_t *attnDesc),
    attnDesc)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_ATTN_DESCRIPTOR,
    ,
    cudnnDestroyAttnDescriptor,
    (cudnnAttnDescriptor_t attnDesc),
    attnDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_ATTN_DESCRIPTOR,
    ,
    cudnnSetAttnDescriptor,
    (cudnnAttnDescriptor_t attnDesc, unsigned attnMode,
    int nHeads, double smScaler,
    cudnnDataType_t dataType, cudnnDataType_t computePrec,
    cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize,
    int kSize, int vSize,
    int qProjSize, int kProjSize,
    int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength,
    int maxBatchSize, int maxBeamSize),
    attnDesc, attnMode, nHeads, smScaler,
    dataType, computePrec, mathType, attnDropoutDesc,
    postDropoutDesc, qSize, kSize, vSize,
    qProjSize, kProjSize, vProjSize, oProjSize,
    qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)

CUDNN_HOOK_GEN(
    CUDNN_GET_ATTN_DESCRIPTOR,
    ,
    cudnnGetAttnDescriptor,
    (cudnnAttnDescriptor_t attnDesc, unsigned *attnMode,
    int *nHeads, double *smScaler,
    cudnnDataType_t *dataType, cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize,
    int *kSize, int *vSize,
    int *qProjSize, int *kProjSize,
    int *vProjSize, int *oProjSize,
    int *qoMaxSeqLength, int *kvMaxSeqLength,
    int *maxBatchSize, int *maxBeamSize),
    attnDesc, attnMode, nHeads, smScaler,
    dataType, computePrec, mathType, attnDropoutDesc,
    postDropoutDesc, qSize, kSize, vSize,
    qProjSize, kProjSize, vProjSize, oProjSize,
    qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)

CUDNN_HOOK_GEN(
    CUDNN_ADV_INFER_VERSION_CHECK,
    ,
    cudnnAdvInferVersionCheck,
    (),
)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_CTC_LOSS_DESCRIPTOR,
    ,
    cudnnCreateCTCLossDescriptor,
    (cudnnCTCLossDescriptor_t *ctcLossDesc),
    ctcLossDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_CTC_LOSS_DESCRIPTOR,
    ,
    cudnnSetCTCLossDescriptor,
    (cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType),
    ctcLossDesc, compType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX,
    ,
    cudnnSetCTCLossDescriptorEx,
    (cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode),
    ctcLossDesc, compType, normMode, gradMode)

CUDNN_HOOK_GEN(
    CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8,
    ,
    cudnnSetCTCLossDescriptor_v8,
    (cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode,
    int maxLabelLength),
    ctcLossDesc, compType, normMode, gradMode,
    maxLabelLength)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_DESCRIPTOR,
    ,
    cudnnGetCTCLossDescriptor,
    (cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType),
    ctcLossDesc, compType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX,
    ,
    cudnnGetCTCLossDescriptorEx,
    (cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode),
    ctcLossDesc, compType, normMode, gradMode)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8,
    ,
    cudnnGetCTCLossDescriptor_v8,
    (cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode,
    int *maxLabelLength),
    ctcLossDesc, compType, normMode, gradMode,
    maxLabelLength)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR,
    ,
    cudnnDestroyCTCLossDescriptor,
    (cudnnCTCLossDescriptor_t ctcLossDesc),
    ctcLossDesc)

CUDNN_HOOK_GEN(
    CUDNN_ADV_TRAIN_VERSION_CHECK,
    ,
    cudnnAdvTrainVersionCheck,
    (),
)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_CONVOLUTION_DESCRIPTOR,
    ,
    cudnnCreateConvolutionDescriptor,
    (cudnnConvolutionDescriptor_t *convDesc),
    convDesc)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR,
    ,
    cudnnDestroyConvolutionDescriptor,
    (cudnnConvolutionDescriptor_t convDesc),
    convDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_MATH_TYPE,
    ,
    cudnnSetConvolutionMathType,
    (cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType),
    convDesc, mathType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_MATH_TYPE,
    ,
    cudnnGetConvolutionMathType,
    (cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType),
    convDesc, mathType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_GROUP_COUNT,
    ,
    cudnnSetConvolutionGroupCount,
    (cudnnConvolutionDescriptor_t convDesc, int groupCount),
    convDesc, groupCount)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_GROUP_COUNT,
    ,
    cudnnGetConvolutionGroupCount,
    (cudnnConvolutionDescriptor_t convDesc, int *groupCount),
    convDesc, groupCount)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_REORDER_TYPE,
    ,
    cudnnSetConvolutionReorderType,
    (cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType),
    convDesc, reorderType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_REORDER_TYPE,
    ,
    cudnnGetConvolutionReorderType,
    (cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType),
    convDesc, reorderType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR,
    ,
    cudnnSetConvolution2dDescriptor,
    (cudnnConvolutionDescriptor_t convDesc, int pad_h,
    int pad_w, int u,
    int v, int dilation_h,
    int dilation_w, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType),
    convDesc, pad_h, pad_w, u,
    v, dilation_h, dilation_w, mode,
    computeType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR,
    ,
    cudnnGetConvolution2dDescriptor,
    (const cudnnConvolutionDescriptor_t convDesc, int *pad_h,
    int *pad_w, int *u,
    int *v, int *dilation_h,
    int *dilation_w, cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType),
    convDesc, pad_h, pad_w, u,
    v, dilation_h, dilation_w, mode,
    computeType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR,
    ,
    cudnnSetConvolutionNdDescriptor,
    (cudnnConvolutionDescriptor_t convDesc, int arrayLength,
    const int padA[], const int filterStrideA[],
    const int dilationA[], cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType),
    convDesc, arrayLength, padA, filterStrideA,
    dilationA, mode, computeType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR,
    ,
    cudnnGetConvolutionNdDescriptor,
    (const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[],
    int strideA[], int dilationA[],
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType),
    convDesc, arrayLengthRequested, arrayLength, padA,
    strideA, dilationA, mode, computeType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetConvolution2dForwardOutputDim,
    (const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n,
    int *c, int *h,
    int *w),
    convDesc, inputTensorDesc, filterDesc, n,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetConvolutionNdForwardOutputDim,
    (const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[]),
    convDesc, inputTensorDesc, filterDesc, nbDims,
    tensorOuputDimA)

CUDNN_HOOK_GEN(
    CUDNN_CNN_INFER_VERSION_CHECK,
    ,
    cudnnCnnInferVersionCheck,
    (),
)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK,
    ,
    cudnnCreateFusedOpsConstParamPack,
    (cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops),
    constPack, ops)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK,
    ,
    cudnnDestroyFusedOpsConstParamPack,
    (cudnnFusedOpsConstParamPack_t constPack),
    constPack)

CUDNN_HOOK_GEN(
    CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE,
    ,
    cudnnSetFusedOpsConstParamPackAttribute,
    (cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param),
    constPack, paramLabel, param)

CUDNN_HOOK_GEN(
    CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE,
    ,
    cudnnGetFusedOpsConstParamPackAttribute,
    (const cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param, int *isNULL),
    constPack, paramLabel, param, isNULL)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK,
    ,
    cudnnCreateFusedOpsVariantParamPack,
    (cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops),
    varPack, ops)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK,
    ,
    cudnnDestroyFusedOpsVariantParamPack,
    (cudnnFusedOpsVariantParamPack_t varPack),
    varPack)

CUDNN_HOOK_GEN(
    CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE,
    ,
    cudnnSetFusedOpsVariantParamPackAttribute,
    (cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr),
    varPack, paramLabel, ptr)

CUDNN_HOOK_GEN(
    CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE,
    ,
    cudnnGetFusedOpsVariantParamPackAttribute,
    (const cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr),
    varPack, paramLabel, ptr)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_FUSED_OPS_PLAN,
    ,
    cudnnCreateFusedOpsPlan,
    (cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops),
    plan, ops)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_FUSED_OPS_PLAN,
    ,
    cudnnDestroyFusedOpsPlan,
    (cudnnFusedOpsPlan_t plan),
    plan)

CUDNN_HOOK_GEN(
    CUDNN_CNN_TRAIN_VERSION_CHECK,
    ,
    cudnnCnnTrainVersionCheck,
    (),
)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_CREATE_DESCRIPTOR,
    ,
    cudnnBackendCreateDescriptor,
    (cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor),
    descriptorType, descriptor)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_DESTROY_DESCRIPTOR,
    ,
    cudnnBackendDestroyDescriptor,
    (cudnnBackendDescriptor_t descriptor),
    descriptor)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_INITIALIZE,
    ,
    cudnnBackendInitialize,
    (cudnnBackendDescriptor_t descriptor),
    descriptor)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_FINALIZE,
    ,
    cudnnBackendFinalize,
    (cudnnBackendDescriptor_t descriptor),
    descriptor)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_SET_ATTRIBUTE,
    ,
    cudnnBackendSetAttribute,
    (cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t elementCount,
    const void *arrayOfElements),
    descriptor, attributeName, attributeType, elementCount,
    arrayOfElements)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_GET_ATTRIBUTE,
    ,
    cudnnBackendGetAttribute,
    (cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount,
    int64_t *elementCount, void *arrayOfElements),
    descriptor, attributeName, attributeType, requestedElementCount,
    elementCount, arrayOfElements)
/* hook function end */
