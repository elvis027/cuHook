#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#include <cudnn.h>
#include <cuda_runtime.h>

#include "hook.h"
#include "cudnn_hook.h"
#include "debug.h"

#ifdef _CUDNN_HOOK_ENABLE

static struct cudnnHookInfo cudnn_hook_info;
static pthread_once_t cudnn_hook_init_done = PTHREAD_ONCE_INIT;

/* prehook, proxy, posthook functions start */
cudnnStatus_t cudnnCreate_prehook(
    cudnnHandle_t *handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreate_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreate_proxy(
    cudnnHandle_t *handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreate_posthook(
    cudnnHandle_t *handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreate_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy_prehook(
    cudnnHandle_t handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroy_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy_proxy(
    cudnnHandle_t handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy_posthook(
    cudnnHandle_t handle
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroy_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnQueryRuntimeError_prehook(
    cudnnHandle_t handle,
    cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode,
    cudnnRuntimeTag_t *tag
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnQueryRuntimeError_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnQueryRuntimeError_proxy(
    cudnnHandle_t handle,
    cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode,
    cudnnRuntimeTag_t *tag
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnQueryRuntimeError_posthook(
    cudnnHandle_t handle,
    cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode,
    cudnnRuntimeTag_t *tag
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnQueryRuntimeError_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream_prehook(
    cudnnHandle_t handle,
    cudaStream_t streamId
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetStream_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream_proxy(
    cudnnHandle_t handle,
    cudaStream_t streamId
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream_posthook(
    cudnnHandle_t handle,
    cudaStream_t streamId
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetStream_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream_prehook(
    cudnnHandle_t handle,
    cudaStream_t *streamId
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetStream_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream_proxy(
    cudnnHandle_t handle,
    cudaStream_t *streamId
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream_posthook(
    cudnnHandle_t handle,
    cudaStream_t *streamId
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetStream_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensor_prehook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnTransformTensor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensor_proxy(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensor_posthook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnTransformTensor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensorEx_prehook(
    cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnTensorDescriptor_t destDesc,
    void *destData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnTransformTensorEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensorEx_proxy(
    cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnTensorDescriptor_t destDesc,
    void *destData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensorEx_posthook(
    cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnTensorDescriptor_t destDesc,
    void *destData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnTransformTensorEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAddTensor_prehook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnAddTensor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAddTensor_proxy(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAddTensor_posthook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnAddTensor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpTensor_prehook(
    cudnnHandle_t handle,
    const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *alpha2,
    const cudnnTensorDescriptor_t bDesc,
    const void *B,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnOpTensor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpTensor_proxy(
    cudnnHandle_t handle,
    const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *alpha2,
    const cudnnTensorDescriptor_t bDesc,
    const void *B,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpTensor_posthook(
    cudnnHandle_t handle,
    const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *alpha2,
    const cudnnTensorDescriptor_t bDesc,
    const void *B,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnOpTensor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionIndicesSize_prehook(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetReductionIndicesSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionIndicesSize_proxy(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionIndicesSize_posthook(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetReductionIndicesSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize_prehook(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetReductionWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize_proxy(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize_posthook(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetReductionWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor_prehook(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices,
    size_t indicesSizeInBytes,
    void *workspace,
    size_t workspaceSizeInBytes,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnReduceTensor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor_proxy(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices,
    size_t indicesSizeInBytes,
    void *workspace,
    size_t workspaceSizeInBytes,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor_posthook(
    cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices,
    size_t indicesSizeInBytes,
    void *workspace,
    size_t workspaceSizeInBytes,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnReduceTensor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *valuePtr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *valuePtr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *valuePtr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnScaleTensor_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *alpha
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnScaleTensor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnScaleTensor_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *alpha
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnScaleTensor_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *alpha
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnScaleTensor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformFilter_prehook(
    cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnFilterDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnFilterDescriptor_t destDesc,
    void *destData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnTransformFilter_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformFilter_proxy(
    cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnFilterDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnFilterDescriptor_t destDesc,
    void *destData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformFilter_posthook(
    cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnFilterDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnFilterDescriptor_t destDesc,
    void *destData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnTransformFilter_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxForward_prehook(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSoftmaxForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxForward_proxy(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxForward_posthook(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSoftmaxForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward_prehook(
    cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnPoolingForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward_proxy(
    cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward_posthook(
    cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnPoolingForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationForward_prehook(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnActivationForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationForward_proxy(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationForward_posthook(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnActivationForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelForward_prehook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnLRNCrossChannelForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelForward_proxy(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelForward_posthook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnLRNCrossChannelForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationForward_prehook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDivisiveNormalizationForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationForward_proxy(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationForward_posthook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDivisiveNormalizationForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    const void *estimatedMean,
    const void *estimatedVariance,
    double epsilon
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationForwardInference_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    const void *estimatedMean,
    const void *estimatedVariance,
    double epsilon
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    const void *estimatedMean,
    const void *estimatedVariance,
    double epsilon
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationForwardInference_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardInference_prehook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean,
    const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    double epsilon,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnNormalizationForwardInference_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardInference_proxy(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean,
    const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    double epsilon,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardInference_posthook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean,
    const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    double epsilon,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnNormalizationForwardInference_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward_prehook(
    cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta,
    void *grid
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfGridGeneratorForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward_proxy(
    cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta,
    void *grid
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward_posthook(
    cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta,
    void *grid
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfGridGeneratorForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerForward_prehook(
    cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *grid,
    const void *beta,
    cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfSamplerForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerForward_proxy(
    cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *grid,
    const void *beta,
    cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerForward_posthook(
    cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *grid,
    const void *beta,
    cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfSamplerForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetStatesSize_prehook(
    cudnnHandle_t handle,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutGetStatesSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetStatesSize_proxy(
    cudnnHandle_t handle,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetStatesSize_posthook(
    cudnnHandle_t handle,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutGetStatesSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutForward_prehook(
    cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc,
    const void *x,
    const cudnnTensorDescriptor_t ydesc,
    void *y,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutForward_proxy(
    cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc,
    const void *x,
    const cudnnTensorDescriptor_t ydesc,
    void *y,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutForward_posthook(
    cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc,
    const void *x,
    const cudnnTensorDescriptor_t ydesc,
    void *y,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize_prehook(
    cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAlgorithmSpaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize_proxy(
    cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize_posthook(
    cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAlgorithmSpaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSaveAlgorithm_prehook(
    cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace,
    size_t algoSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSaveAlgorithm_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSaveAlgorithm_proxy(
    cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace,
    size_t algoSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSaveAlgorithm_posthook(
    cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace,
    size_t algoSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSaveAlgorithm_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreAlgorithm_prehook(
    cudnnHandle_t handle,
    void *algoSpace,
    size_t algoSpaceSizeInBytes,
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRestoreAlgorithm_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreAlgorithm_proxy(
    cudnnHandle_t handle,
    void *algoSpace,
    size_t algoSpaceSizeInBytes,
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreAlgorithm_posthook(
    cudnnHandle_t handle,
    void *algoSpace,
    size_t algoSpaceSizeInBytes,
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRestoreAlgorithm_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxBackward_prehook(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSoftmaxBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxBackward_proxy(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxBackward_posthook(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSoftmaxBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward_prehook(
    cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnPoolingBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward_proxy(
    cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward_posthook(
    cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnPoolingBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward_prehook(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnActivationBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward_proxy(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward_posthook(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnActivationBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelBackward_prehook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnLRNCrossChannelBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelBackward_proxy(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnLRNCrossChannelBackward_posthook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnLRNCrossChannelBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward_prehook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    const void *dy,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx,
    void *dMeans
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDivisiveNormalizationBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward_proxy(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    const void *dy,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx,
    void *dMeans
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward_posthook(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    const void *dy,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx,
    void *dMeans
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDivisiveNormalizationBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetBatchNormalizationBackwardExWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetBatchNormalizationBackwardExWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetBatchNormalizationTrainingExReserveSpaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetBatchNormalizationTrainingExReserveSpaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationForwardTraining_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationForwardTraining_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationForwardTrainingEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationForwardTrainingEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackward_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale,
    void *dBnScaleResult,
    void *dBnBiasResult,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackward_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale,
    void *dBnScaleResult,
    void *dBnBiasResult,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackward_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale,
    void *dBnScaleResult,
    void *dBnBiasResult,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx_prehook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData,
    const void *bnBiasData,
    void *dBnScaleData,
    void *dBnBiasData,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationBackwardEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx_proxy(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData,
    const void *bnBiasData,
    void *dBnScaleData,
    void *dBnBiasData,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx_posthook(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData,
    const void *bnBiasData,
    void *dBnScaleData,
    void *dBnBiasData,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBatchNormalizationBackwardEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize_prehook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetNormalizationForwardTrainingWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize_proxy(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize_posthook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetNormalizationForwardTrainingWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize_prehook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetNormalizationBackwardWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize_proxy(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize_posthook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetNormalizationBackwardWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize_prehook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetNormalizationTrainingReserveSpaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize_proxy(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize_posthook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetNormalizationTrainingReserveSpaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardTraining_prehook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnNormalizationForwardTraining_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardTraining_proxy(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardTraining_posthook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnNormalizationForwardTraining_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationBackward_prehook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const void *normScaleData,
    const void *normBiasData,
    void *dNormScaleData,
    void *dNormBiasData,
    double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnNormalizationBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationBackward_proxy(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const void *normScaleData,
    const void *normBiasData,
    void *dNormScaleData,
    void *dNormBiasData,
    double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationBackward_posthook(
    cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const void *normScaleData,
    const void *normBiasData,
    void *dNormScaleData,
    void *dNormBiasData,
    double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnNormalizationBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward_prehook(
    cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid,
    void *dtheta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfGridGeneratorBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward_proxy(
    cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid,
    void *dtheta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward_posthook(
    cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid,
    void *dtheta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfGridGeneratorBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward_prehook(
    cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const void *alphaDgrid,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *grid,
    const void *betaDgrid,
    void *dgrid
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfSamplerBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward_proxy(
    cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const void *alphaDgrid,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *grid,
    const void *betaDgrid,
    void *dgrid
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward_posthook(
    cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const void *alphaDgrid,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *grid,
    const void *betaDgrid,
    void *dgrid
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSpatialTfSamplerBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutBackward_prehook(
    cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxdesc,
    void *dx,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutBackward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutBackward_proxy(
    cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxdesc,
    void *dx,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutBackward_posthook(
    cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxdesc,
    void *dx,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutBackward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize,
    const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction,
    cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo,
    cudnnDataType_t mathPrec
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNDescriptor_v6_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize,
    const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction,
    cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo,
    cudnnDataType_t mathPrec
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize,
    const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction,
    cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo,
    cudnnDataType_t mathPrec
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNDescriptor_v6_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize,
    int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction,
    cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo,
    cudnnDataType_t *mathPrec
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNDescriptor_v6_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize,
    int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction,
    cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo,
    cudnnDataType_t *mathPrec
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize,
    int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction,
    cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo,
    cudnnDataType_t *mathPrec
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNDescriptor_v6_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNSetClip_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNSetClip_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNGetClip_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNGetClip_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNProjectionLayers_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize,
    const int outProjSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNProjectionLayers_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNProjectionLayers_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize,
    const int outProjSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNProjectionLayers_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize,
    const int outProjSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNProjectionLayers_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNProjectionLayers_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize,
    int *outProjSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNProjectionLayers_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNProjectionLayers_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize,
    int *outProjSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNProjectionLayers_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize,
    int *outProjSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNProjectionLayers_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBuildRNNDynamic_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int miniBatch
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBuildRNNDynamic_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBuildRNNDynamic_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int miniBatch
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBuildRNNDynamic_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int miniBatch
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBuildRNNDynamic_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNTrainingReserveSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNTrainingReserveSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode,
    cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize,
    size_t *reserveSpaceSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNTempSpaceSizes_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode,
    cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize,
    size_t *reserveSpaceSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode,
    cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize,
    size_t *reserveSpaceSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNTempSpaceSizes_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNParamsSize_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    cudnnDataType_t dataType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNParamsSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNParamsSize_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    cudnnDataType_t dataType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNParamsSize_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    cudnnDataType_t dataType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNParamsSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNWeightSpaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNWeightSpaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNLinLayerMatrixParams_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNLinLayerMatrixParams_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNLinLayerBiasParams_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNLinLayerBiasParams_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightParams_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer,
    size_t weightSpaceSize,
    const void *weightSpace,
    int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc,
    void **mAddr,
    cudnnTensorDescriptor_t bDesc,
    void **bAddr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNWeightParams_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightParams_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer,
    size_t weightSpaceSize,
    const void *weightSpace,
    int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc,
    void **mAddr,
    cudnnTensorDescriptor_t bDesc,
    void **bAddr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightParams_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer,
    size_t weightSpaceSize,
    const void *weightSpace,
    int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc,
    void **mAddr,
    cudnnTensorDescriptor_t bDesc,
    void **bAddr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNWeightParams_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardInference_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardInference_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInferenceEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardInferenceEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInferenceEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInferenceEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardInferenceEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForward_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnRNNDataDescriptor_t yDesc,
    void *y,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    void *hy,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    void *cy,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForward_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnRNNDataDescriptor_t yDesc,
    void *y,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    void *hy,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    void *cy,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForward_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnRNNDataDescriptor_t yDesc,
    void *y,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    void *hy,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    void *cy,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNAlgorithmDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNAlgorithmDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNForwardInferenceAlgorithmMaxCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNForwardInferenceAlgorithmMaxCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNForwardInferenceAlgorithmEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNForwardInferenceAlgorithmEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers_prehook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes,
    size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetMultiHeadAttnBuffers_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers_proxy(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes,
    size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers_posthook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes,
    size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetMultiHeadAttnBuffers_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights_prehook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind,
    size_t weightSizeInBytes,
    const void *weights,
    cudnnTensorDescriptor_t wDesc,
    void **wAddr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetMultiHeadAttnWeights_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights_proxy(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind,
    size_t weightSizeInBytes,
    const void *weights,
    cudnnTensorDescriptor_t wDesc,
    void **wAddr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights_posthook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind,
    size_t weightSizeInBytes,
    const void *weights,
    cudnnTensorDescriptor_t wDesc,
    void **wAddr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetMultiHeadAttnWeights_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnForward_prehook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    int currIdx,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsQO[],
    const int devSeqLengthsKV[],
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t oDesc,
    void *out,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMultiHeadAttnForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnForward_proxy(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    int currIdx,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsQO[],
    const int devSeqLengthsKV[],
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t oDesc,
    void *out,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnForward_posthook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    int currIdx,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsQO[],
    const int devSeqLengthsKV[],
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t oDesc,
    void *out,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMultiHeadAttnForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTraining_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardTraining_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTraining_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTraining_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardTraining_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardData_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardData_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_v8_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const void *dy,
    cudnnRNNDataDescriptor_t xDesc,
    void *dx,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    const void *dhy,
    void *dhx,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    const void *dcy,
    void *dcx,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardData_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_v8_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const void *dy,
    cudnnRNNDataDescriptor_t xDesc,
    void *dx,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    const void *dhy,
    void *dhx,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    const void *dcy,
    void *dcx,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardData_v8_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const void *dy,
    cudnnRNNDataDescriptor_t xDesc,
    void *dx,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    const void *dhy,
    void *dhx,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    const void *dcy,
    void *dcx,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardData_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardWeights_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardWeights_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_v8_prehook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    size_t weightSpaceSize,
    void *dweightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardWeights_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_v8_proxy(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    size_t weightSpaceSize,
    void *dweightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights_v8_posthook(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    size_t weightSpaceSize,
    void *dweightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardWeights_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTrainingEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardTrainingEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTrainingEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTrainingEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNForwardTrainingEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardDataEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const cudnnRNNDataDescriptor_t dyDesc,
    const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc,
    const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc,
    void *dkeys,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardDataEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardDataEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const cudnnRNNDataDescriptor_t dyDesc,
    const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc,
    const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc,
    void *dkeys,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardDataEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const cudnnRNNDataDescriptor_t dyDesc,
    const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc,
    const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc,
    void *dkeys,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardDataEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardWeightsEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNBackwardWeightsEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNForwardTrainingAlgorithmMaxCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNForwardTrainingAlgorithmMaxCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNForwardTrainingAlgorithmEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNForwardTrainingAlgorithmEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNBackwardDataAlgorithmMaxCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNBackwardDataAlgorithmMaxCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNBackwardDataAlgorithmEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNBackwardDataAlgorithmEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNBackwardWeightsAlgorithmMaxCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNBackwardWeightsAlgorithmMaxCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx_prehook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNBackwardWeightsAlgorithmEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx_proxy(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx_posthook(
    cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindRNNBackwardWeightsAlgorithmEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData_prehook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsDQDO[],
    const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc,
    void *dqueries,
    const void *queries,
    const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys,
    const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc,
    void *dvalues,
    const void *values,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMultiHeadAttnBackwardData_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData_proxy(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsDQDO[],
    const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc,
    void *dqueries,
    const void *queries,
    const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys,
    const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc,
    void *dvalues,
    const void *values,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData_posthook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsDQDO[],
    const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc,
    void *dqueries,
    const void *queries,
    const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys,
    const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc,
    void *dvalues,
    const void *values,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMultiHeadAttnBackwardData_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights_prehook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    size_t weightSizeInBytes,
    const void *weights,
    void *dweights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMultiHeadAttnBackwardWeights_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights_proxy(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    size_t weightSizeInBytes,
    const void *weights,
    void *dweights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights_posthook(
    cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    size_t weightSizeInBytes,
    const void *weights,
    void *dweights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMultiHeadAttnBackwardWeights_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int hostLabels[],
    const int hostLabelLengths[],
    const int hostInputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCTCLoss_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int hostLabels[],
    const int hostLabelLengths[],
    const int hostInputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int hostLabels[],
    const int hostLabelLengths[],
    const int hostInputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCTCLoss_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_v8_prehook(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int labels[],
    const int labelLengths[],
    const int inputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    size_t workSpaceSizeInBytes,
    void *workspace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCTCLoss_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_v8_proxy(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int labels[],
    const int labelLengths[],
    const int inputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    size_t workSpaceSizeInBytes,
    void *workspace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss_v8_posthook(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int labels[],
    const int labelLengths[],
    const int inputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    size_t workSpaceSizeInBytes,
    void *workspace
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCTCLoss_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    const int *labels,
    const int *labelLengths,
    const int *inputLengths,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    const int *labels,
    const int *labelLengths,
    const int *inputLengths,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    const int *labels,
    const int *labelLengths,
    const int *inputLengths,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8_prehook(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossWorkspaceSize_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8_proxy(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8_posthook(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossWorkspaceSize_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount_prehook(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionForwardAlgorithmMaxCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount_proxy(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount_posthook(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionForwardAlgorithmMaxCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionForwardAlgorithm_v7_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionForwardAlgorithm_v7_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionForwardAlgorithm_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionForwardAlgorithm_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionForwardAlgorithmEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionForwardAlgorithmEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnIm2Col_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    void *colBuffer
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnIm2Col_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnIm2Col_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    void *colBuffer
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnIm2Col_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    void *colBuffer
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnIm2Col_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReorderFilterAndBias_prehook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType,
    const void *filterData,
    void *reorderedFilterData,
    int reorderBias,
    const void *biasData,
    void *reorderedBiasData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnReorderFilterAndBias_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReorderFilterAndBias_proxy(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType,
    const void *filterData,
    void *reorderedFilterData,
    int reorderBias,
    const void *biasData,
    void *reorderedBiasData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReorderFilterAndBias_posthook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType,
    const void *filterData,
    void *reorderedFilterData,
    int reorderBias,
    const void *biasData,
    void *reorderedBiasData
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnReorderFilterAndBias_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionForwardWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionForwardWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward_prehook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward_proxy(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward_posthook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward_prehook(
    cudnnHandle_t handle,
    const void *alpha1,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *alpha2,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    const cudnnTensorDescriptor_t biasDesc,
    const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBiasActivationForward_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward_proxy(
    cudnnHandle_t handle,
    const void *alpha1,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *alpha2,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    const cudnnTensorDescriptor_t biasDesc,
    const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward_posthook(
    cudnnHandle_t handle,
    const void *alpha1,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *alpha2,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    const cudnnTensorDescriptor_t biasDesc,
    const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBiasActivationForward_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount_prehook(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardDataAlgorithmMaxCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount_proxy(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount_posthook(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardDataAlgorithmMaxCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm_prehook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardDataAlgorithm_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm_proxy(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm_posthook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardDataAlgorithm_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx_prehook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardDataAlgorithmEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx_proxy(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx_posthook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardDataAlgorithmEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7_prehook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardDataAlgorithm_v7_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7_proxy(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7_posthook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardDataAlgorithm_v7_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize_prehook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardDataWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize_proxy(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize_posthook(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardDataWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData_prehook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBackwardData_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData_proxy(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData_posthook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBackwardData_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors_prehook(
    const cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFoldedConvBackwardDataDescriptors_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors_proxy(
    const cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors_posthook(
    const cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFoldedConvBackwardDataDescriptors_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_prehook(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_proxy(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_posthook(
    cudnnHandle_t handle,
    int *count
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardFilterAlgorithm_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardFilterAlgorithm_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardFilterAlgorithmEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFindConvolutionBackwardFilterAlgorithmEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardFilterAlgorithm_v7_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardFilterAlgorithm_v7_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize_prehook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardFilterWorkspaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize_proxy(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize_posthook(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionBackwardFilterWorkspaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_prehook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBackwardFilter_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_proxy(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_posthook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBackwardFilter_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardBias_prehook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dbDesc,
    void *db
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBackwardBias_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardBias_proxy(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dbDesc,
    void *db
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardBias_posthook(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dbDesc,
    void *db
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnConvolutionBackwardBias_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMakeFusedOpsPlan_prehook(
    cudnnHandle_t handle,
    cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack,
    size_t *workspaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMakeFusedOpsPlan_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMakeFusedOpsPlan_proxy(
    cudnnHandle_t handle,
    cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack,
    size_t *workspaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMakeFusedOpsPlan_posthook(
    cudnnHandle_t handle,
    cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack,
    size_t *workspaceSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnMakeFusedOpsPlan_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFusedOpsExecute_prehook(
    cudnnHandle_t handle,
    const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFusedOpsExecute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFusedOpsExecute_proxy(
    cudnnHandle_t handle,
    const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFusedOpsExecute_posthook(
    cudnnHandle_t handle,
    const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnFusedOpsExecute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendExecute_prehook(
    cudnnHandle_t handle,
    cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendExecute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendExecute_proxy(
    cudnnHandle_t handle,
    cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendExecute_posthook(
    cudnnHandle_t handle,
    cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendExecute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetProperty_prehook(
    libraryPropertyType type,
    int *value
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetProperty_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetProperty_proxy(
    libraryPropertyType type,
    int *value
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetProperty_posthook(
    libraryPropertyType type,
    int *value
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetProperty_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorDescriptor_prehook(
    cudnnTensorDescriptor_t *tensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorDescriptor_proxy(
    cudnnTensorDescriptor_t *tensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorDescriptor_posthook(
    cudnnTensorDescriptor_t *tensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor_prehook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensor4dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor_proxy(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor_posthook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensor4dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx_prehook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w,
    int nStride,
    int cStride,
    int hStride,
    int wStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensor4dDescriptorEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx_proxy(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w,
    int nStride,
    int cStride,
    int hStride,
    int wStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx_posthook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w,
    int nStride,
    int cStride,
    int hStride,
    int wStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensor4dDescriptorEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor_prehook(
    const cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t *dataType,
    int *n,
    int *c,
    int *h,
    int *w,
    int *nStride,
    int *cStride,
    int *hStride,
    int *wStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensor4dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor_proxy(
    const cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t *dataType,
    int *n,
    int *c,
    int *h,
    int *w,
    int *nStride,
    int *cStride,
    int *hStride,
    int *wStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor_posthook(
    const cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t *dataType,
    int *n,
    int *c,
    int *h,
    int *w,
    int *nStride,
    int *cStride,
    int *hStride,
    int *wStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensor4dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor_prehook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensorNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor_proxy(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor_posthook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensorNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx_prehook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensorNdDescriptorEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx_proxy(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx_posthook(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensorNdDescriptorEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor_prehook(
    const cudnnTensorDescriptor_t tensorDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    int *nbDims,
    int dimA[],
    int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensorNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor_proxy(
    const cudnnTensorDescriptor_t tensorDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    int *nbDims,
    int dimA[],
    int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor_posthook(
    const cudnnTensorDescriptor_t tensorDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    int *nbDims,
    int dimA[],
    int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensorNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorSizeInBytes_prehook(
    const cudnnTensorDescriptor_t tensorDesc,
    size_t *size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensorSizeInBytes_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorSizeInBytes_proxy(
    const cudnnTensorDescriptor_t tensorDesc,
    size_t *size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorSizeInBytes_posthook(
    const cudnnTensorDescriptor_t tensorDesc,
    size_t *size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensorSizeInBytes_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor_prehook(
    cudnnTensorDescriptor_t tensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor_proxy(
    cudnnTensorDescriptor_t tensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor_posthook(
    cudnnTensorDescriptor_t tensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnInitTransformDest_prehook(
    const cudnnTensorTransformDescriptor_t transformDesc,
    const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc,
    size_t *destSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnInitTransformDest_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnInitTransformDest_proxy(
    const cudnnTensorTransformDescriptor_t transformDesc,
    const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc,
    size_t *destSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnInitTransformDest_posthook(
    const cudnnTensorTransformDescriptor_t transformDesc,
    const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc,
    size_t *destSizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnInitTransformDest_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t *transformDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateTensorTransformDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t *transformDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t *transformDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateTensorTransformDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t transformDesc,
    const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat,
    const int32_t padBeforeA[],
    const int32_t padAfterA[],
    const uint32_t foldA[],
    const cudnnFoldingDirection_t direction
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensorTransformDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t transformDesc,
    const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat,
    const int32_t padBeforeA[],
    const int32_t padAfterA[],
    const uint32_t foldA[],
    const cudnnFoldingDirection_t direction
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t transformDesc,
    const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat,
    const int32_t padBeforeA[],
    const int32_t padAfterA[],
    const uint32_t foldA[],
    const cudnnFoldingDirection_t direction
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetTensorTransformDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t transformDesc,
    uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat,
    int32_t padBeforeA[],
    int32_t padAfterA[],
    uint32_t foldA[],
    cudnnFoldingDirection_t *direction
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensorTransformDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t transformDesc,
    uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat,
    int32_t padBeforeA[],
    int32_t padAfterA[],
    uint32_t foldA[],
    cudnnFoldingDirection_t *direction
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t transformDesc,
    uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat,
    int32_t padBeforeA[],
    int32_t padAfterA[],
    uint32_t foldA[],
    cudnnFoldingDirection_t *direction
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetTensorTransformDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor_prehook(
    cudnnTensorTransformDescriptor_t transformDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyTensorTransformDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor_proxy(
    cudnnTensorTransformDescriptor_t transformDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor_posthook(
    cudnnTensorTransformDescriptor_t transformDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyTensorTransformDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateOpTensorDescriptor_prehook(
    cudnnOpTensorDescriptor_t *opTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateOpTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateOpTensorDescriptor_proxy(
    cudnnOpTensorDescriptor_t *opTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateOpTensorDescriptor_posthook(
    cudnnOpTensorDescriptor_t *opTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateOpTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor_prehook(
    cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType,
    cudnnNanPropagation_t opTensorNanOpt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetOpTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor_proxy(
    cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType,
    cudnnNanPropagation_t opTensorNanOpt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor_posthook(
    cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType,
    cudnnNanPropagation_t opTensorNanOpt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetOpTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetOpTensorDescriptor_prehook(
    const cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType,
    cudnnNanPropagation_t *opTensorNanOpt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetOpTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetOpTensorDescriptor_proxy(
    const cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType,
    cudnnNanPropagation_t *opTensorNanOpt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetOpTensorDescriptor_posthook(
    const cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType,
    cudnnNanPropagation_t *opTensorNanOpt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetOpTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor_prehook(
    cudnnOpTensorDescriptor_t opTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyOpTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor_proxy(
    cudnnOpTensorDescriptor_t opTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor_posthook(
    cudnnOpTensorDescriptor_t opTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyOpTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor_prehook(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateReduceTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor_proxy(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor_posthook(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateReduceTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor_prehook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType,
    cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices,
    cudnnIndicesType_t reduceTensorIndicesType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetReduceTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor_proxy(
    cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType,
    cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices,
    cudnnIndicesType_t reduceTensorIndicesType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor_posthook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType,
    cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices,
    cudnnIndicesType_t reduceTensorIndicesType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetReduceTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor_prehook(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetReduceTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor_proxy(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor_posthook(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetReduceTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor_prehook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyReduceTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor_proxy(
    cudnnReduceTensorDescriptor_t reduceTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor_posthook(
    cudnnReduceTensorDescriptor_t reduceTensorDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyReduceTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor_prehook(
    cudnnFilterDescriptor_t *filterDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFilterDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor_proxy(
    cudnnFilterDescriptor_t *filterDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor_posthook(
    cudnnFilterDescriptor_t *filterDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFilterDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor_prehook(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int k,
    int c,
    int h,
    int w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFilter4dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor_proxy(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int k,
    int c,
    int h,
    int w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor_posthook(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int k,
    int c,
    int h,
    int w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFilter4dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilter4dDescriptor_prehook(
    const cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *k,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFilter4dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilter4dDescriptor_proxy(
    const cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *k,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilter4dDescriptor_posthook(
    const cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *k,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFilter4dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor_prehook(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int nbDims,
    const int filterDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFilterNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor_proxy(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int nbDims,
    const int filterDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor_posthook(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int nbDims,
    const int filterDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFilterNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterNdDescriptor_prehook(
    const cudnnFilterDescriptor_t filterDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *nbDims,
    int filterDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFilterNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterNdDescriptor_proxy(
    const cudnnFilterDescriptor_t filterDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *nbDims,
    int filterDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterNdDescriptor_posthook(
    const cudnnFilterDescriptor_t filterDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *nbDims,
    int filterDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFilterNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterSizeInBytes_prehook(
    const cudnnFilterDescriptor_t filterDesc,
    size_t *size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFilterSizeInBytes_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterSizeInBytes_proxy(
    const cudnnFilterDescriptor_t filterDesc,
    size_t *size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterSizeInBytes_posthook(
    const cudnnFilterDescriptor_t filterDesc,
    size_t *size
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFilterSizeInBytes_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor_prehook(
    cudnnFilterDescriptor_t filterDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFilterDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor_proxy(
    cudnnFilterDescriptor_t filterDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor_posthook(
    cudnnFilterDescriptor_t filterDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFilterDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePoolingDescriptor_prehook(
    cudnnPoolingDescriptor_t *poolingDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreatePoolingDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePoolingDescriptor_proxy(
    cudnnPoolingDescriptor_t *poolingDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePoolingDescriptor_posthook(
    cudnnPoolingDescriptor_t *poolingDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreatePoolingDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor_prehook(
    cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt,
    int windowHeight,
    int windowWidth,
    int verticalPadding,
    int horizontalPadding,
    int verticalStride,
    int horizontalStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetPooling2dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor_proxy(
    cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt,
    int windowHeight,
    int windowWidth,
    int verticalPadding,
    int horizontalPadding,
    int verticalStride,
    int horizontalStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor_posthook(
    cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt,
    int windowHeight,
    int windowWidth,
    int verticalPadding,
    int horizontalPadding,
    int verticalStride,
    int horizontalStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetPooling2dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dDescriptor_prehook(
    const cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *windowHeight,
    int *windowWidth,
    int *verticalPadding,
    int *horizontalPadding,
    int *verticalStride,
    int *horizontalStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPooling2dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dDescriptor_proxy(
    const cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *windowHeight,
    int *windowWidth,
    int *verticalPadding,
    int *horizontalPadding,
    int *verticalStride,
    int *horizontalStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dDescriptor_posthook(
    const cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *windowHeight,
    int *windowWidth,
    int *verticalPadding,
    int *horizontalPadding,
    int *verticalStride,
    int *horizontalStride
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPooling2dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_prehook(
    cudnnPoolingDescriptor_t poolingDesc,
    const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt,
    int nbDims,
    const int windowDimA[],
    const int paddingA[],
    const int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetPoolingNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_proxy(
    cudnnPoolingDescriptor_t poolingDesc,
    const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt,
    int nbDims,
    const int windowDimA[],
    const int paddingA[],
    const int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_posthook(
    cudnnPoolingDescriptor_t poolingDesc,
    const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt,
    int nbDims,
    const int windowDimA[],
    const int paddingA[],
    const int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetPoolingNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor_prehook(
    const cudnnPoolingDescriptor_t poolingDesc,
    int nbDimsRequested,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims,
    int windowDimA[],
    int paddingA[],
    int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPoolingNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor_proxy(
    const cudnnPoolingDescriptor_t poolingDesc,
    int nbDimsRequested,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims,
    int windowDimA[],
    int paddingA[],
    int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor_posthook(
    const cudnnPoolingDescriptor_t poolingDesc,
    int nbDimsRequested,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims,
    int windowDimA[],
    int paddingA[],
    int strideA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPoolingNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim_prehook(
    const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims,
    int outputTensorDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPoolingNdForwardOutputDim_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim_proxy(
    const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims,
    int outputTensorDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim_posthook(
    const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims,
    int outputTensorDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPoolingNdForwardOutputDim_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim_prehook(
    const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int *n,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPooling2dForwardOutputDim_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim_proxy(
    const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int *n,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim_posthook(
    const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int *n,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetPooling2dForwardOutputDim_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor_prehook(
    cudnnPoolingDescriptor_t poolingDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyPoolingDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor_proxy(
    cudnnPoolingDescriptor_t poolingDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor_posthook(
    cudnnPoolingDescriptor_t poolingDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyPoolingDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateActivationDescriptor_prehook(
    cudnnActivationDescriptor_t *activationDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateActivationDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateActivationDescriptor_proxy(
    cudnnActivationDescriptor_t *activationDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateActivationDescriptor_posthook(
    cudnnActivationDescriptor_t *activationDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateActivationDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor_prehook(
    cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt,
    double coef
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetActivationDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor_proxy(
    cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt,
    double coef
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor_posthook(
    cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt,
    double coef
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetActivationDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptor_prehook(
    const cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt,
    double *coef
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetActivationDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptor_proxy(
    const cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt,
    double *coef
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptor_posthook(
    const cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt,
    double *coef
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetActivationDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta_prehook(
    cudnnActivationDescriptor_t activationDesc,
    double swish_beta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetActivationDescriptorSwishBeta_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta_proxy(
    cudnnActivationDescriptor_t activationDesc,
    double swish_beta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta_posthook(
    cudnnActivationDescriptor_t activationDesc,
    double swish_beta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetActivationDescriptorSwishBeta_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta_prehook(
    cudnnActivationDescriptor_t activationDesc,
    double *swish_beta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetActivationDescriptorSwishBeta_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta_proxy(
    cudnnActivationDescriptor_t activationDesc,
    double *swish_beta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta_posthook(
    cudnnActivationDescriptor_t activationDesc,
    double *swish_beta
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetActivationDescriptorSwishBeta_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor_prehook(
    cudnnActivationDescriptor_t activationDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyActivationDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor_proxy(
    cudnnActivationDescriptor_t activationDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor_posthook(
    cudnnActivationDescriptor_t activationDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyActivationDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateLRNDescriptor_prehook(
    cudnnLRNDescriptor_t *normDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateLRNDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateLRNDescriptor_proxy(
    cudnnLRNDescriptor_t *normDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateLRNDescriptor_posthook(
    cudnnLRNDescriptor_t *normDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateLRNDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetLRNDescriptor_prehook(
    cudnnLRNDescriptor_t normDesc,
    unsigned lrnN,
    double lrnAlpha,
    double lrnBeta,
    double lrnK
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetLRNDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetLRNDescriptor_proxy(
    cudnnLRNDescriptor_t normDesc,
    unsigned lrnN,
    double lrnAlpha,
    double lrnBeta,
    double lrnK
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetLRNDescriptor_posthook(
    cudnnLRNDescriptor_t normDesc,
    unsigned lrnN,
    double lrnAlpha,
    double lrnBeta,
    double lrnK
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetLRNDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetLRNDescriptor_prehook(
    cudnnLRNDescriptor_t normDesc,
    unsigned *lrnN,
    double *lrnAlpha,
    double *lrnBeta,
    double *lrnK
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetLRNDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetLRNDescriptor_proxy(
    cudnnLRNDescriptor_t normDesc,
    unsigned *lrnN,
    double *lrnAlpha,
    double *lrnBeta,
    double *lrnK
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetLRNDescriptor_posthook(
    cudnnLRNDescriptor_t normDesc,
    unsigned *lrnN,
    double *lrnAlpha,
    double *lrnBeta,
    double *lrnK
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetLRNDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyLRNDescriptor_prehook(
    cudnnLRNDescriptor_t lrnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyLRNDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyLRNDescriptor_proxy(
    cudnnLRNDescriptor_t lrnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyLRNDescriptor_posthook(
    cudnnLRNDescriptor_t lrnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyLRNDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor_prehook(
    cudnnTensorDescriptor_t derivedBnDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDeriveBNTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor_proxy(
    cudnnTensorDescriptor_t derivedBnDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor_posthook(
    cudnnTensorDescriptor_t derivedBnDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDeriveBNTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor_prehook(
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc,
    cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnNormMode_t mode,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDeriveNormTensorDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor_proxy(
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc,
    cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnNormMode_t mode,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor_posthook(
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc,
    cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnNormMode_t mode,
    int groupCnt
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDeriveNormTensorDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor_prehook(
    cudnnSpatialTransformerDescriptor_t *stDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateSpatialTransformerDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor_proxy(
    cudnnSpatialTransformerDescriptor_t *stDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor_posthook(
    cudnnSpatialTransformerDescriptor_t *stDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateSpatialTransformerDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor_prehook(
    cudnnSpatialTransformerDescriptor_t stDesc,
    cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType,
    const int nbDims,
    const int dimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetSpatialTransformerNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor_proxy(
    cudnnSpatialTransformerDescriptor_t stDesc,
    cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType,
    const int nbDims,
    const int dimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor_posthook(
    cudnnSpatialTransformerDescriptor_t stDesc,
    cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType,
    const int nbDims,
    const int dimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetSpatialTransformerNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor_prehook(
    cudnnSpatialTransformerDescriptor_t stDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroySpatialTransformerDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor_proxy(
    cudnnSpatialTransformerDescriptor_t stDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor_posthook(
    cudnnSpatialTransformerDescriptor_t stDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroySpatialTransformerDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t *dropoutDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateDropoutDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t *dropoutDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t *dropoutDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateDropoutDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyDropoutDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyDropoutDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize_prehook(
    cudnnTensorDescriptor_t xdesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutGetReserveSpaceSize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize_proxy(
    cudnnTensorDescriptor_t xdesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize_posthook(
    cudnnTensorDescriptor_t xdesc,
    size_t *sizeInBytes
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDropoutGetReserveSpaceSize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetDropoutDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetDropoutDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRestoreDropoutDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRestoreDropoutDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetDropoutDescriptor_prehook(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float *dropout,
    void **states,
    unsigned long long *seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetDropoutDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetDropoutDescriptor_proxy(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float *dropout,
    void **states,
    unsigned long long *seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetDropoutDescriptor_posthook(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float *dropout,
    void **states,
    unsigned long long *seed
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetDropoutDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor_prehook(
    cudnnAlgorithmDescriptor_t *algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateAlgorithmDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor_proxy(
    cudnnAlgorithmDescriptor_t *algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor_posthook(
    cudnnAlgorithmDescriptor_t *algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateAlgorithmDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor_prehook(
    cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t algorithm
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetAlgorithmDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor_proxy(
    cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t algorithm
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor_posthook(
    cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t algorithm
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetAlgorithmDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmDescriptor_prehook(
    const cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t *algorithm
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAlgorithmDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmDescriptor_proxy(
    const cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t *algorithm
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmDescriptor_posthook(
    const cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t *algorithm
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAlgorithmDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor_prehook(
    const cudnnAlgorithmDescriptor_t src,
    cudnnAlgorithmDescriptor_t dest
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCopyAlgorithmDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor_proxy(
    const cudnnAlgorithmDescriptor_t src,
    cudnnAlgorithmDescriptor_t dest
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor_posthook(
    const cudnnAlgorithmDescriptor_t src,
    cudnnAlgorithmDescriptor_t dest
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCopyAlgorithmDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor_prehook(
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyAlgorithmDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor_proxy(
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor_posthook(
    cudnnAlgorithmDescriptor_t algoDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyAlgorithmDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmPerformance_prehook(
    cudnnAlgorithmPerformance_t *algoPerf,
    int numberToCreate
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateAlgorithmPerformance_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmPerformance_proxy(
    cudnnAlgorithmPerformance_t *algoPerf,
    int numberToCreate
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAlgorithmPerformance_posthook(
    cudnnAlgorithmPerformance_t *algoPerf,
    int numberToCreate
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateAlgorithmPerformance_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmPerformance_prehook(
    cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status,
    float time,
    size_t memory
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetAlgorithmPerformance_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmPerformance_proxy(
    cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status,
    float time,
    size_t memory
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAlgorithmPerformance_posthook(
    cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status,
    float time,
    size_t memory
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetAlgorithmPerformance_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmPerformance_prehook(
    const cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status,
    float *time,
    size_t *memory
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAlgorithmPerformance_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmPerformance_proxy(
    const cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status,
    float *time,
    size_t *memory
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAlgorithmPerformance_posthook(
    const cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status,
    float *time,
    size_t *memory
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAlgorithmPerformance_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance_prehook(
    cudnnAlgorithmPerformance_t *algoPerf,
    int numberToDestroy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyAlgorithmPerformance_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance_proxy(
    cudnnAlgorithmPerformance_t *algoPerf,
    int numberToDestroy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance_posthook(
    cudnnAlgorithmPerformance_t *algoPerf,
    int numberToDestroy
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyAlgorithmPerformance_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCallback_prehook(
    unsigned mask,
    void *udata,
    cudnnCallback_t fptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCallback_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCallback_proxy(
    unsigned mask,
    void *udata,
    cudnnCallback_t fptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCallback_posthook(
    unsigned mask,
    void *udata,
    cudnnCallback_t fptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCallback_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCallback_prehook(
    unsigned *mask,
    void **udata,
    cudnnCallback_t *fptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCallback_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCallback_proxy(
    unsigned *mask,
    void **udata,
    cudnnCallback_t *fptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCallback_posthook(
    unsigned *mask,
    void **udata,
    cudnnCallback_t *fptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCallback_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsInferVersionCheck_prehook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnOpsInferVersionCheck_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsInferVersionCheck_proxy(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsInferVersionCheck_posthook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnOpsInferVersionCheck_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsTrainVersionCheck_prehook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnOpsTrainVersionCheck_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsTrainVersionCheck_proxy(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpsTrainVersionCheck_posthook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnOpsTrainVersionCheck_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDescriptor_prehook(
    cudnnRNNDescriptor_t *rnnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateRNNDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDescriptor_proxy(
    cudnnRNNDescriptor_t *rnnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDescriptor_posthook(
    cudnnRNNDescriptor_t *rnnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateRNNDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDescriptor_prehook(
    cudnnRNNDescriptor_t rnnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyRNNDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDescriptor_proxy(
    cudnnRNNDescriptor_t rnnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDescriptor_posthook(
    cudnnRNNDescriptor_t rnnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyRNNDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode,
    cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode,
    cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType,
    cudnnDataType_t mathPrec,
    cudnnMathType_t mathType,
    int32_t inputSize,
    int32_t hiddenSize,
    int32_t projSize,
    int32_t numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNDescriptor_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode,
    cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode,
    cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType,
    cudnnDataType_t mathPrec,
    cudnnMathType_t mathType,
    int32_t inputSize,
    int32_t hiddenSize,
    int32_t projSize,
    int32_t numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode,
    cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode,
    cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType,
    cudnnDataType_t mathPrec,
    cudnnMathType_t mathType,
    int32_t inputSize,
    int32_t hiddenSize,
    int32_t projSize,
    int32_t numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNDescriptor_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode,
    cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode,
    cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType,
    cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType,
    int32_t *inputSize,
    int32_t *hiddenSize,
    int32_t *projSize,
    int32_t *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNDescriptor_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode,
    cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode,
    cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType,
    cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType,
    int32_t *inputSize,
    int32_t *hiddenSize,
    int32_t *projSize,
    int32_t *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode,
    cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode,
    cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType,
    cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType,
    int32_t *inputSize,
    int32_t *hiddenSize,
    int32_t *projSize,
    int32_t *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNDescriptor_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNMatrixMathType_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t mType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNMatrixMathType_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNMatrixMathType_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t mType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNMatrixMathType_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t mType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNMatrixMathType_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNMatrixMathType_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t *mType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNMatrixMathType_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNMatrixMathType_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t *mType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNMatrixMathType_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t *mType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNMatrixMathType_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNBiasMode_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t biasMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNBiasMode_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNBiasMode_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t biasMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNBiasMode_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t biasMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNBiasMode_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBiasMode_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t *biasMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNBiasMode_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBiasMode_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t *biasMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNBiasMode_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t *biasMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNBiasMode_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNSetClip_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNSetClip_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNSetClip_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_v8_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNGetClip_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_v8_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNGetClip_v8_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnRNNGetClip_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    const int minibatch,
    const cudnnDataType_t dataType,
    cudnnPersistentRNNPlan_t *plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreatePersistentRNNPlan_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    const int minibatch,
    const cudnnDataType_t dataType,
    cudnnPersistentRNNPlan_t *plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    const int minibatch,
    const cudnnDataType_t dataType,
    cudnnPersistentRNNPlan_t *plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreatePersistentRNNPlan_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan_prehook(
    cudnnPersistentRNNPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyPersistentRNNPlan_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan_proxy(
    cudnnPersistentRNNPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan_posthook(
    cudnnPersistentRNNPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyPersistentRNNPlan_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPersistentRNNPlan_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnPersistentRNNPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetPersistentRNNPlan_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPersistentRNNPlan_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnPersistentRNNPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPersistentRNNPlan_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    cudnnPersistentRNNPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetPersistentRNNPlan_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNPaddingMode_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    unsigned paddingMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNPaddingMode_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNPaddingMode_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    unsigned paddingMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNPaddingMode_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    unsigned paddingMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNPaddingMode_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNPaddingMode_prehook(
    cudnnRNNDescriptor_t rnnDesc,
    unsigned *paddingMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNPaddingMode_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNPaddingMode_proxy(
    cudnnRNNDescriptor_t rnnDesc,
    unsigned *paddingMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNPaddingMode_posthook(
    cudnnRNNDescriptor_t rnnDesc,
    unsigned *paddingMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNPaddingMode_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t *rnnDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateRNNDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t *rnnDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t *rnnDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateRNNDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t rnnDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyRNNDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t rnnDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t rnnDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyRNNDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout,
    int maxSeqLength,
    int batchSize,
    int vectorSize,
    const int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout,
    int maxSeqLength,
    int batchSize,
    int vectorSize,
    const int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout,
    int maxSeqLength,
    int batchSize,
    int vectorSize,
    const int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetRNNDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDataDescriptor_prehook(
    cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout,
    int *maxSeqLength,
    int *batchSize,
    int *vectorSize,
    int arrayLengthRequested,
    int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDataDescriptor_proxy(
    cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout,
    int *maxSeqLength,
    int *batchSize,
    int *vectorSize,
    int arrayLengthRequested,
    int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDataDescriptor_posthook(
    cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout,
    int *maxSeqLength,
    int *batchSize,
    int *vectorSize,
    int arrayLengthRequested,
    int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetRNNDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSeqDataDescriptor_prehook(
    cudnnSeqDataDescriptor_t *seqDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateSeqDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSeqDataDescriptor_proxy(
    cudnnSeqDataDescriptor_t *seqDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateSeqDataDescriptor_posthook(
    cudnnSeqDataDescriptor_t *seqDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateSeqDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor_prehook(
    cudnnSeqDataDescriptor_t seqDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroySeqDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor_proxy(
    cudnnSeqDataDescriptor_t seqDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor_posthook(
    cudnnSeqDataDescriptor_t seqDataDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroySeqDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSeqDataDescriptor_prehook(
    cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const cudnnSeqDataAxis_t axes[],
    size_t seqLengthArraySize,
    const int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetSeqDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSeqDataDescriptor_proxy(
    cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const cudnnSeqDataAxis_t axes[],
    size_t seqLengthArraySize,
    const int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSeqDataDescriptor_posthook(
    cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const cudnnSeqDataAxis_t axes[],
    size_t seqLengthArraySize,
    const int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetSeqDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetSeqDataDescriptor_prehook(
    const cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t *dataType,
    int *nbDims,
    int nbDimsRequested,
    int dimA[],
    cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize,
    size_t seqLengthSizeRequested,
    int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetSeqDataDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetSeqDataDescriptor_proxy(
    const cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t *dataType,
    int *nbDims,
    int nbDimsRequested,
    int dimA[],
    cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize,
    size_t seqLengthSizeRequested,
    int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetSeqDataDescriptor_posthook(
    const cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t *dataType,
    int *nbDims,
    int nbDimsRequested,
    int dimA[],
    cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize,
    size_t seqLengthSizeRequested,
    int seqLengthArray[],
    void *paddingFill
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetSeqDataDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAttnDescriptor_prehook(
    cudnnAttnDescriptor_t *attnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateAttnDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAttnDescriptor_proxy(
    cudnnAttnDescriptor_t *attnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateAttnDescriptor_posthook(
    cudnnAttnDescriptor_t *attnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateAttnDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAttnDescriptor_prehook(
    cudnnAttnDescriptor_t attnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyAttnDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAttnDescriptor_proxy(
    cudnnAttnDescriptor_t attnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyAttnDescriptor_posthook(
    cudnnAttnDescriptor_t attnDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyAttnDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAttnDescriptor_prehook(
    cudnnAttnDescriptor_t attnDesc,
    unsigned attnMode,
    int nHeads,
    double smScaler,
    cudnnDataType_t dataType,
    cudnnDataType_t computePrec,
    cudnnMathType_t mathType,
    cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc,
    int qSize,
    int kSize,
    int vSize,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int oProjSize,
    int qoMaxSeqLength,
    int kvMaxSeqLength,
    int maxBatchSize,
    int maxBeamSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetAttnDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAttnDescriptor_proxy(
    cudnnAttnDescriptor_t attnDesc,
    unsigned attnMode,
    int nHeads,
    double smScaler,
    cudnnDataType_t dataType,
    cudnnDataType_t computePrec,
    cudnnMathType_t mathType,
    cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc,
    int qSize,
    int kSize,
    int vSize,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int oProjSize,
    int qoMaxSeqLength,
    int kvMaxSeqLength,
    int maxBatchSize,
    int maxBeamSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetAttnDescriptor_posthook(
    cudnnAttnDescriptor_t attnDesc,
    unsigned attnMode,
    int nHeads,
    double smScaler,
    cudnnDataType_t dataType,
    cudnnDataType_t computePrec,
    cudnnMathType_t mathType,
    cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc,
    int qSize,
    int kSize,
    int vSize,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int oProjSize,
    int qoMaxSeqLength,
    int kvMaxSeqLength,
    int maxBatchSize,
    int maxBeamSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetAttnDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAttnDescriptor_prehook(
    cudnnAttnDescriptor_t attnDesc,
    unsigned *attnMode,
    int *nHeads,
    double *smScaler,
    cudnnDataType_t *dataType,
    cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType,
    cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc,
    int *qSize,
    int *kSize,
    int *vSize,
    int *qProjSize,
    int *kProjSize,
    int *vProjSize,
    int *oProjSize,
    int *qoMaxSeqLength,
    int *kvMaxSeqLength,
    int *maxBatchSize,
    int *maxBeamSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAttnDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAttnDescriptor_proxy(
    cudnnAttnDescriptor_t attnDesc,
    unsigned *attnMode,
    int *nHeads,
    double *smScaler,
    cudnnDataType_t *dataType,
    cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType,
    cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc,
    int *qSize,
    int *kSize,
    int *vSize,
    int *qProjSize,
    int *kProjSize,
    int *vProjSize,
    int *oProjSize,
    int *qoMaxSeqLength,
    int *kvMaxSeqLength,
    int *maxBatchSize,
    int *maxBeamSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetAttnDescriptor_posthook(
    cudnnAttnDescriptor_t attnDesc,
    unsigned *attnMode,
    int *nHeads,
    double *smScaler,
    cudnnDataType_t *dataType,
    cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType,
    cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc,
    int *qSize,
    int *kSize,
    int *vSize,
    int *qProjSize,
    int *kProjSize,
    int *vProjSize,
    int *oProjSize,
    int *qoMaxSeqLength,
    int *kvMaxSeqLength,
    int *maxBatchSize,
    int *maxBeamSize
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetAttnDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvInferVersionCheck_prehook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnAdvInferVersionCheck_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvInferVersionCheck_proxy(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvInferVersionCheck_posthook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnAdvInferVersionCheck_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t *ctcLossDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateCTCLossDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t *ctcLossDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t *ctcLossDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateCTCLossDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCTCLossDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCTCLossDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCTCLossDescriptorEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCTCLossDescriptorEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode,
    int maxLabelLength
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCTCLossDescriptor_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode,
    int maxLabelLength
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode,
    int maxLabelLength
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetCTCLossDescriptor_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossDescriptorEx_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossDescriptorEx_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode,
    int *maxLabelLength
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossDescriptor_v8_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode,
    int *maxLabelLength
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode,
    int *maxLabelLength
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetCTCLossDescriptor_v8_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor_prehook(
    cudnnCTCLossDescriptor_t ctcLossDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyCTCLossDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor_proxy(
    cudnnCTCLossDescriptor_t ctcLossDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor_posthook(
    cudnnCTCLossDescriptor_t ctcLossDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyCTCLossDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvTrainVersionCheck_prehook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnAdvTrainVersionCheck_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvTrainVersionCheck_proxy(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAdvTrainVersionCheck_posthook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnAdvTrainVersionCheck_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor_prehook(
    cudnnConvolutionDescriptor_t *convDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateConvolutionDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor_proxy(
    cudnnConvolutionDescriptor_t *convDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor_posthook(
    cudnnConvolutionDescriptor_t *convDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateConvolutionDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor_prehook(
    cudnnConvolutionDescriptor_t convDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyConvolutionDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor_proxy(
    cudnnConvolutionDescriptor_t convDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor_posthook(
    cudnnConvolutionDescriptor_t convDesc
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyConvolutionDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t mathType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionMathType_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t mathType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t mathType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionMathType_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t *mathType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionMathType_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t *mathType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t *mathType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionMathType_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    int groupCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionGroupCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    int groupCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    int groupCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionGroupCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    int *groupCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionGroupCount_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    int *groupCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    int *groupCount
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionGroupCount_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionReorderType_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t reorderType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionReorderType_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionReorderType_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t reorderType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionReorderType_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t reorderType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionReorderType_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionReorderType_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t *reorderType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionReorderType_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionReorderType_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t *reorderType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionReorderType_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t *reorderType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionReorderType_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    int pad_h,
    int pad_w,
    int u,
    int v,
    int dilation_h,
    int dilation_w,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolution2dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    int pad_h,
    int pad_w,
    int u,
    int v,
    int dilation_h,
    int dilation_w,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    int pad_h,
    int pad_w,
    int u,
    int v,
    int dilation_h,
    int dilation_w,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolution2dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor_prehook(
    const cudnnConvolutionDescriptor_t convDesc,
    int *pad_h,
    int *pad_w,
    int *u,
    int *v,
    int *dilation_h,
    int *dilation_w,
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolution2dDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor_proxy(
    const cudnnConvolutionDescriptor_t convDesc,
    int *pad_h,
    int *pad_w,
    int *u,
    int *v,
    int *dilation_h,
    int *dilation_w,
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor_posthook(
    const cudnnConvolutionDescriptor_t convDesc,
    int *pad_h,
    int *pad_w,
    int *u,
    int *v,
    int *dilation_h,
    int *dilation_w,
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolution2dDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_prehook(
    cudnnConvolutionDescriptor_t convDesc,
    int arrayLength,
    const int padA[],
    const int filterStrideA[],
    const int dilationA[],
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_proxy(
    cudnnConvolutionDescriptor_t convDesc,
    int arrayLength,
    const int padA[],
    const int filterStrideA[],
    const int dilationA[],
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_posthook(
    cudnnConvolutionDescriptor_t convDesc,
    int arrayLength,
    const int padA[],
    const int filterStrideA[],
    const int dilationA[],
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetConvolutionNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_prehook(
    const cudnnConvolutionDescriptor_t convDesc,
    int arrayLengthRequested,
    int *arrayLength,
    int padA[],
    int strideA[],
    int dilationA[],
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionNdDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_proxy(
    const cudnnConvolutionDescriptor_t convDesc,
    int arrayLengthRequested,
    int *arrayLength,
    int padA[],
    int strideA[],
    int dilationA[],
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_posthook(
    const cudnnConvolutionDescriptor_t convDesc,
    int arrayLengthRequested,
    int *arrayLength,
    int padA[],
    int strideA[],
    int dilationA[],
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionNdDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim_prehook(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int *n,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolution2dForwardOutputDim_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim_proxy(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int *n,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim_posthook(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int *n,
    int *c,
    int *h,
    int *w
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolution2dForwardOutputDim_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim_prehook(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int nbDims,
    int tensorOuputDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionNdForwardOutputDim_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim_proxy(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int nbDims,
    int tensorOuputDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim_posthook(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int nbDims,
    int tensorOuputDimA[]
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetConvolutionNdForwardOutputDim_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnInferVersionCheck_prehook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCnnInferVersionCheck_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnInferVersionCheck_proxy(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnInferVersionCheck_posthook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCnnInferVersionCheck_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack_prehook(
    cudnnFusedOpsConstParamPack_t *constPack,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFusedOpsConstParamPack_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack_proxy(
    cudnnFusedOpsConstParamPack_t *constPack,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack_posthook(
    cudnnFusedOpsConstParamPack_t *constPack,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFusedOpsConstParamPack_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack_prehook(
    cudnnFusedOpsConstParamPack_t constPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFusedOpsConstParamPack_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack_proxy(
    cudnnFusedOpsConstParamPack_t constPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack_posthook(
    cudnnFusedOpsConstParamPack_t constPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFusedOpsConstParamPack_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute_prehook(
    cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFusedOpsConstParamPackAttribute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute_proxy(
    cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute_posthook(
    cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFusedOpsConstParamPackAttribute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute_prehook(
    const cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param,
    int *isNULL
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFusedOpsConstParamPackAttribute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute_proxy(
    const cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param,
    int *isNULL
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute_posthook(
    const cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param,
    int *isNULL
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFusedOpsConstParamPackAttribute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack_prehook(
    cudnnFusedOpsVariantParamPack_t *varPack,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFusedOpsVariantParamPack_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack_proxy(
    cudnnFusedOpsVariantParamPack_t *varPack,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack_posthook(
    cudnnFusedOpsVariantParamPack_t *varPack,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFusedOpsVariantParamPack_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack_prehook(
    cudnnFusedOpsVariantParamPack_t varPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFusedOpsVariantParamPack_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack_proxy(
    cudnnFusedOpsVariantParamPack_t varPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack_posthook(
    cudnnFusedOpsVariantParamPack_t varPack
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFusedOpsVariantParamPack_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute_prehook(
    cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFusedOpsVariantParamPackAttribute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute_proxy(
    cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute_posthook(
    cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnSetFusedOpsVariantParamPackAttribute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute_prehook(
    const cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFusedOpsVariantParamPackAttribute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute_proxy(
    const cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute_posthook(
    const cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnGetFusedOpsVariantParamPackAttribute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsPlan_prehook(
    cudnnFusedOpsPlan_t *plan,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFusedOpsPlan_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsPlan_proxy(
    cudnnFusedOpsPlan_t *plan,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFusedOpsPlan_posthook(
    cudnnFusedOpsPlan_t *plan,
    cudnnFusedOps_t ops
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCreateFusedOpsPlan_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan_prehook(
    cudnnFusedOpsPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFusedOpsPlan_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan_proxy(
    cudnnFusedOpsPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan_posthook(
    cudnnFusedOpsPlan_t plan
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnDestroyFusedOpsPlan_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnTrainVersionCheck_prehook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCnnTrainVersionCheck_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnTrainVersionCheck_proxy(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCnnTrainVersionCheck_posthook(

) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnCnnTrainVersionCheck_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendCreateDescriptor_prehook(
    cudnnBackendDescriptorType_t descriptorType,
    cudnnBackendDescriptor_t *descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendCreateDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendCreateDescriptor_proxy(
    cudnnBackendDescriptorType_t descriptorType,
    cudnnBackendDescriptor_t *descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendCreateDescriptor_posthook(
    cudnnBackendDescriptorType_t descriptorType,
    cudnnBackendDescriptor_t *descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendCreateDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor_prehook(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendDestroyDescriptor_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor_proxy(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor_posthook(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendDestroyDescriptor_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendInitialize_prehook(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendInitialize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendInitialize_proxy(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendInitialize_posthook(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendInitialize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendFinalize_prehook(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendFinalize_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendFinalize_proxy(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendFinalize_posthook(
    cudnnBackendDescriptor_t descriptor
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendFinalize_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendSetAttribute_prehook(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void *arrayOfElements
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendSetAttribute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendSetAttribute_proxy(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void *arrayOfElements
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendSetAttribute_posthook(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void *arrayOfElements
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendSetAttribute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute_prehook(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t *elementCount,
    void *arrayOfElements
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendGetAttribute_pre\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute_proxy(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t *elementCount,
    void *arrayOfElements
) {
    DEBUG("[%s] Enter func\n", __func__);
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute_posthook(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t *elementCount,
    void *arrayOfElements
) {
    DEBUG("[%s] Enter func\n", __func__);
    DUMP_TRACE("cudnnBackendGetAttribute_pos\n");
    DEBUG("[%s] Leave func\n", __func__);
    return CUDNN_STATUS_SUCCESS;
}
/* prehook, proxy, posthook functions end */

static void cudnn_hook_init()
{
    cudnn_hook_info.func_prehook[CUDNN_CREATE] =
        (void *)cudnnCreate_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE] =
        (void *)cudnnCreate_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE] =
        (void *)cudnnCreate_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY] =
        (void *)cudnnDestroy_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY] =
        (void *)cudnnDestroy_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY] =
        (void *)cudnnDestroy_posthook;
    cudnn_hook_info.func_prehook[CUDNN_QUERY_RUNTIME_ERROR] =
        (void *)cudnnQueryRuntimeError_prehook;
    cudnn_hook_info.func_proxy[CUDNN_QUERY_RUNTIME_ERROR] =
        (void *)cudnnQueryRuntimeError_proxy;
    cudnn_hook_info.func_posthook[CUDNN_QUERY_RUNTIME_ERROR] =
        (void *)cudnnQueryRuntimeError_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_PROPERTY] =
        (void *)cudnnGetProperty_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_PROPERTY] =
        (void *)cudnnGetProperty_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_PROPERTY] =
        (void *)cudnnGetProperty_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_STREAM] =
        (void *)cudnnSetStream_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_STREAM] =
        (void *)cudnnSetStream_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_STREAM] =
        (void *)cudnnSetStream_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_STREAM] =
        (void *)cudnnGetStream_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_STREAM] =
        (void *)cudnnGetStream_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_STREAM] =
        (void *)cudnnGetStream_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_4D_DESCRIPTOR] =
        (void *)cudnnSetTensor4dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_4D_DESCRIPTOR] =
        (void *)cudnnSetTensor4dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_4D_DESCRIPTOR] =
        (void *)cudnnSetTensor4dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX] =
        (void *)cudnnSetTensor4dDescriptorEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX] =
        (void *)cudnnSetTensor4dDescriptorEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX] =
        (void *)cudnnSetTensor4dDescriptorEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_4D_DESCRIPTOR] =
        (void *)cudnnGetTensor4dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_4D_DESCRIPTOR] =
        (void *)cudnnGetTensor4dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_4D_DESCRIPTOR] =
        (void *)cudnnGetTensor4dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_ND_DESCRIPTOR] =
        (void *)cudnnSetTensorNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_ND_DESCRIPTOR] =
        (void *)cudnnSetTensorNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_ND_DESCRIPTOR] =
        (void *)cudnnSetTensorNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX] =
        (void *)cudnnSetTensorNdDescriptorEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX] =
        (void *)cudnnSetTensorNdDescriptorEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX] =
        (void *)cudnnSetTensorNdDescriptorEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_ND_DESCRIPTOR] =
        (void *)cudnnGetTensorNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_ND_DESCRIPTOR] =
        (void *)cudnnGetTensorNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_ND_DESCRIPTOR] =
        (void *)cudnnGetTensorNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_SIZE_IN_BYTES] =
        (void *)cudnnGetTensorSizeInBytes_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_SIZE_IN_BYTES] =
        (void *)cudnnGetTensorSizeInBytes_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_SIZE_IN_BYTES] =
        (void *)cudnnGetTensorSizeInBytes_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_INIT_TRANSFORM_DEST] =
        (void *)cudnnInitTransformDest_prehook;
    cudnn_hook_info.func_proxy[CUDNN_INIT_TRANSFORM_DEST] =
        (void *)cudnnInitTransformDest_proxy;
    cudnn_hook_info.func_posthook[CUDNN_INIT_TRANSFORM_DEST] =
        (void *)cudnnInitTransformDest_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnCreateTensorTransformDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnCreateTensorTransformDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnCreateTensorTransformDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnSetTensorTransformDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnSetTensorTransformDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnSetTensorTransformDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnGetTensorTransformDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnGetTensorTransformDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnGetTensorTransformDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnDestroyTensorTransformDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnDestroyTensorTransformDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_TENSOR_TRANSFORM_DESCRIPTOR] =
        (void *)cudnnDestroyTensorTransformDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_TRANSFORM_TENSOR] =
        (void *)cudnnTransformTensor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_TRANSFORM_TENSOR] =
        (void *)cudnnTransformTensor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_TRANSFORM_TENSOR] =
        (void *)cudnnTransformTensor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_TRANSFORM_TENSOR_EX] =
        (void *)cudnnTransformTensorEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_TRANSFORM_TENSOR_EX] =
        (void *)cudnnTransformTensorEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_TRANSFORM_TENSOR_EX] =
        (void *)cudnnTransformTensorEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_ADD_TENSOR] =
        (void *)cudnnAddTensor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_ADD_TENSOR] =
        (void *)cudnnAddTensor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_ADD_TENSOR] =
        (void *)cudnnAddTensor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateOpTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateOpTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateOpTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnSetOpTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnSetOpTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnSetOpTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnGetOpTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnGetOpTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnGetOpTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyOpTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyOpTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyOpTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_OP_TENSOR] =
        (void *)cudnnOpTensor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_OP_TENSOR] =
        (void *)cudnnOpTensor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_OP_TENSOR] =
        (void *)cudnnOpTensor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateReduceTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateReduceTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnCreateReduceTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnSetReduceTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnSetReduceTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnSetReduceTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnGetReduceTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnGetReduceTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnGetReduceTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyReduceTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyReduceTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR] =
        (void *)cudnnDestroyReduceTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_REDUCTION_INDICES_SIZE] =
        (void *)cudnnGetReductionIndicesSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_REDUCTION_INDICES_SIZE] =
        (void *)cudnnGetReductionIndicesSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_REDUCTION_INDICES_SIZE] =
        (void *)cudnnGetReductionIndicesSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_REDUCTION_WORKSPACE_SIZE] =
        (void *)cudnnGetReductionWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_REDUCTION_WORKSPACE_SIZE] =
        (void *)cudnnGetReductionWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_REDUCTION_WORKSPACE_SIZE] =
        (void *)cudnnGetReductionWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_REDUCE_TENSOR] =
        (void *)cudnnReduceTensor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_REDUCE_TENSOR] =
        (void *)cudnnReduceTensor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_REDUCE_TENSOR] =
        (void *)cudnnReduceTensor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_TENSOR] =
        (void *)cudnnSetTensor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_TENSOR] =
        (void *)cudnnSetTensor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_TENSOR] =
        (void *)cudnnSetTensor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SCALE_TENSOR] =
        (void *)cudnnScaleTensor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SCALE_TENSOR] =
        (void *)cudnnScaleTensor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SCALE_TENSOR] =
        (void *)cudnnScaleTensor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FILTER_DESCRIPTOR] =
        (void *)cudnnCreateFilterDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FILTER_DESCRIPTOR] =
        (void *)cudnnCreateFilterDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FILTER_DESCRIPTOR] =
        (void *)cudnnCreateFilterDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_FILTER_4D_DESCRIPTOR] =
        (void *)cudnnSetFilter4dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_FILTER_4D_DESCRIPTOR] =
        (void *)cudnnSetFilter4dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_FILTER_4D_DESCRIPTOR] =
        (void *)cudnnSetFilter4dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_FILTER_4D_DESCRIPTOR] =
        (void *)cudnnGetFilter4dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_FILTER_4D_DESCRIPTOR] =
        (void *)cudnnGetFilter4dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_FILTER_4D_DESCRIPTOR] =
        (void *)cudnnGetFilter4dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_FILTER_ND_DESCRIPTOR] =
        (void *)cudnnSetFilterNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_FILTER_ND_DESCRIPTOR] =
        (void *)cudnnSetFilterNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_FILTER_ND_DESCRIPTOR] =
        (void *)cudnnSetFilterNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_FILTER_ND_DESCRIPTOR] =
        (void *)cudnnGetFilterNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_FILTER_ND_DESCRIPTOR] =
        (void *)cudnnGetFilterNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_FILTER_ND_DESCRIPTOR] =
        (void *)cudnnGetFilterNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_FILTER_SIZE_IN_BYTES] =
        (void *)cudnnGetFilterSizeInBytes_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_FILTER_SIZE_IN_BYTES] =
        (void *)cudnnGetFilterSizeInBytes_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_FILTER_SIZE_IN_BYTES] =
        (void *)cudnnGetFilterSizeInBytes_posthook;
    cudnn_hook_info.func_prehook[CUDNN_TRANSFORM_FILTER] =
        (void *)cudnnTransformFilter_prehook;
    cudnn_hook_info.func_proxy[CUDNN_TRANSFORM_FILTER] =
        (void *)cudnnTransformFilter_proxy;
    cudnn_hook_info.func_posthook[CUDNN_TRANSFORM_FILTER] =
        (void *)cudnnTransformFilter_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FILTER_DESCRIPTOR] =
        (void *)cudnnDestroyFilterDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FILTER_DESCRIPTOR] =
        (void *)cudnnDestroyFilterDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FILTER_DESCRIPTOR] =
        (void *)cudnnDestroyFilterDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SOFTMAX_FORWARD] =
        (void *)cudnnSoftmaxForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SOFTMAX_FORWARD] =
        (void *)cudnnSoftmaxForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SOFTMAX_FORWARD] =
        (void *)cudnnSoftmaxForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_POOLING_DESCRIPTOR] =
        (void *)cudnnCreatePoolingDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_POOLING_DESCRIPTOR] =
        (void *)cudnnCreatePoolingDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_POOLING_DESCRIPTOR] =
        (void *)cudnnCreatePoolingDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_POOLING_2D_DESCRIPTOR] =
        (void *)cudnnSetPooling2dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_POOLING_2D_DESCRIPTOR] =
        (void *)cudnnSetPooling2dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_POOLING_2D_DESCRIPTOR] =
        (void *)cudnnSetPooling2dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_2D_DESCRIPTOR] =
        (void *)cudnnGetPooling2dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_2D_DESCRIPTOR] =
        (void *)cudnnGetPooling2dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_2D_DESCRIPTOR] =
        (void *)cudnnGetPooling2dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_POOLING_ND_DESCRIPTOR] =
        (void *)cudnnSetPoolingNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_POOLING_ND_DESCRIPTOR] =
        (void *)cudnnSetPoolingNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_POOLING_ND_DESCRIPTOR] =
        (void *)cudnnSetPoolingNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_ND_DESCRIPTOR] =
        (void *)cudnnGetPoolingNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_ND_DESCRIPTOR] =
        (void *)cudnnGetPoolingNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_ND_DESCRIPTOR] =
        (void *)cudnnGetPoolingNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetPoolingNdForwardOutputDim_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetPoolingNdForwardOutputDim_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetPoolingNdForwardOutputDim_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetPooling2dForwardOutputDim_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetPooling2dForwardOutputDim_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetPooling2dForwardOutputDim_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_POOLING_DESCRIPTOR] =
        (void *)cudnnDestroyPoolingDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_POOLING_DESCRIPTOR] =
        (void *)cudnnDestroyPoolingDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_POOLING_DESCRIPTOR] =
        (void *)cudnnDestroyPoolingDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_POOLING_FORWARD] =
        (void *)cudnnPoolingForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_POOLING_FORWARD] =
        (void *)cudnnPoolingForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_POOLING_FORWARD] =
        (void *)cudnnPoolingForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnCreateActivationDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnCreateActivationDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnCreateActivationDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnSetActivationDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnSetActivationDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnSetActivationDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnGetActivationDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnGetActivationDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnGetActivationDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        (void *)cudnnSetActivationDescriptorSwishBeta_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        (void *)cudnnSetActivationDescriptorSwishBeta_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        (void *)cudnnSetActivationDescriptorSwishBeta_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        (void *)cudnnGetActivationDescriptorSwishBeta_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        (void *)cudnnGetActivationDescriptorSwishBeta_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA] =
        (void *)cudnnGetActivationDescriptorSwishBeta_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnDestroyActivationDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnDestroyActivationDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ACTIVATION_DESCRIPTOR] =
        (void *)cudnnDestroyActivationDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_ACTIVATION_FORWARD] =
        (void *)cudnnActivationForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_ACTIVATION_FORWARD] =
        (void *)cudnnActivationForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_ACTIVATION_FORWARD] =
        (void *)cudnnActivationForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_LRN_DESCRIPTOR] =
        (void *)cudnnCreateLRNDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_LRN_DESCRIPTOR] =
        (void *)cudnnCreateLRNDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_LRN_DESCRIPTOR] =
        (void *)cudnnCreateLRNDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_LRN_DESCRIPTOR] =
        (void *)cudnnSetLRNDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_LRN_DESCRIPTOR] =
        (void *)cudnnSetLRNDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_LRN_DESCRIPTOR] =
        (void *)cudnnSetLRNDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_LRN_DESCRIPTOR] =
        (void *)cudnnGetLRNDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_LRN_DESCRIPTOR] =
        (void *)cudnnGetLRNDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_LRN_DESCRIPTOR] =
        (void *)cudnnGetLRNDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_LRN_DESCRIPTOR] =
        (void *)cudnnDestroyLRNDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_LRN_DESCRIPTOR] =
        (void *)cudnnDestroyLRNDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_LRN_DESCRIPTOR] =
        (void *)cudnnDestroyLRNDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_LRN_CROSS_CHANNEL_FORWARD] =
        (void *)cudnnLRNCrossChannelForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_LRN_CROSS_CHANNEL_FORWARD] =
        (void *)cudnnLRNCrossChannelForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_LRN_CROSS_CHANNEL_FORWARD] =
        (void *)cudnnLRNCrossChannelForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DIVISIVE_NORMALIZATION_FORWARD] =
        (void *)cudnnDivisiveNormalizationForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DIVISIVE_NORMALIZATION_FORWARD] =
        (void *)cudnnDivisiveNormalizationForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DIVISIVE_NORMALIZATION_FORWARD] =
        (void *)cudnnDivisiveNormalizationForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR] =
        (void *)cudnnDeriveBNTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR] =
        (void *)cudnnDeriveBNTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR] =
        (void *)cudnnDeriveBNTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE] =
        (void *)cudnnBatchNormalizationForwardInference_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE] =
        (void *)cudnnBatchNormalizationForwardInference_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE] =
        (void *)cudnnBatchNormalizationForwardInference_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR] =
        (void *)cudnnDeriveNormTensorDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR] =
        (void *)cudnnDeriveNormTensorDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR] =
        (void *)cudnnDeriveNormTensorDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_NORMALIZATION_FORWARD_INFERENCE] =
        (void *)cudnnNormalizationForwardInference_prehook;
    cudnn_hook_info.func_proxy[CUDNN_NORMALIZATION_FORWARD_INFERENCE] =
        (void *)cudnnNormalizationForwardInference_proxy;
    cudnn_hook_info.func_posthook[CUDNN_NORMALIZATION_FORWARD_INFERENCE] =
        (void *)cudnnNormalizationForwardInference_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        (void *)cudnnCreateSpatialTransformerDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        (void *)cudnnCreateSpatialTransformerDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        (void *)cudnnCreateSpatialTransformerDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR] =
        (void *)cudnnSetSpatialTransformerNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR] =
        (void *)cudnnSetSpatialTransformerNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_SPATIAL_TRANSFORMER_ND_DESCRIPTOR] =
        (void *)cudnnSetSpatialTransformerNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        (void *)cudnnDestroySpatialTransformerDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        (void *)cudnnDestroySpatialTransformerDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_SPATIAL_TRANSFORMER_DESCRIPTOR] =
        (void *)cudnnDestroySpatialTransformerDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD] =
        (void *)cudnnSpatialTfGridGeneratorForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD] =
        (void *)cudnnSpatialTfGridGeneratorForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD] =
        (void *)cudnnSpatialTfGridGeneratorForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_SAMPLER_FORWARD] =
        (void *)cudnnSpatialTfSamplerForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_SAMPLER_FORWARD] =
        (void *)cudnnSpatialTfSamplerForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_SAMPLER_FORWARD] =
        (void *)cudnnSpatialTfSamplerForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_DROPOUT_DESCRIPTOR] =
        (void *)cudnnCreateDropoutDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_DROPOUT_DESCRIPTOR] =
        (void *)cudnnCreateDropoutDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_DROPOUT_DESCRIPTOR] =
        (void *)cudnnCreateDropoutDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_DROPOUT_DESCRIPTOR] =
        (void *)cudnnDestroyDropoutDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_DROPOUT_DESCRIPTOR] =
        (void *)cudnnDestroyDropoutDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_DROPOUT_DESCRIPTOR] =
        (void *)cudnnDestroyDropoutDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_GET_STATES_SIZE] =
        (void *)cudnnDropoutGetStatesSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_GET_STATES_SIZE] =
        (void *)cudnnDropoutGetStatesSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_GET_STATES_SIZE] =
        (void *)cudnnDropoutGetStatesSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE] =
        (void *)cudnnDropoutGetReserveSpaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE] =
        (void *)cudnnDropoutGetReserveSpaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE] =
        (void *)cudnnDropoutGetReserveSpaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_DROPOUT_DESCRIPTOR] =
        (void *)cudnnSetDropoutDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_DROPOUT_DESCRIPTOR] =
        (void *)cudnnSetDropoutDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_DROPOUT_DESCRIPTOR] =
        (void *)cudnnSetDropoutDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RESTORE_DROPOUT_DESCRIPTOR] =
        (void *)cudnnRestoreDropoutDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RESTORE_DROPOUT_DESCRIPTOR] =
        (void *)cudnnRestoreDropoutDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RESTORE_DROPOUT_DESCRIPTOR] =
        (void *)cudnnRestoreDropoutDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_DROPOUT_DESCRIPTOR] =
        (void *)cudnnGetDropoutDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_DROPOUT_DESCRIPTOR] =
        (void *)cudnnGetDropoutDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_DROPOUT_DESCRIPTOR] =
        (void *)cudnnGetDropoutDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_FORWARD] =
        (void *)cudnnDropoutForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_FORWARD] =
        (void *)cudnnDropoutForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_FORWARD] =
        (void *)cudnnDropoutForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnCreateAlgorithmDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnCreateAlgorithmDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnCreateAlgorithmDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnSetAlgorithmDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnSetAlgorithmDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnSetAlgorithmDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnGetAlgorithmDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnGetAlgorithmDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnGetAlgorithmDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_COPY_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnCopyAlgorithmDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_COPY_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnCopyAlgorithmDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_COPY_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnCopyAlgorithmDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnDestroyAlgorithmDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnDestroyAlgorithmDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnDestroyAlgorithmDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ALGORITHM_PERFORMANCE] =
        (void *)cudnnCreateAlgorithmPerformance_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ALGORITHM_PERFORMANCE] =
        (void *)cudnnCreateAlgorithmPerformance_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ALGORITHM_PERFORMANCE] =
        (void *)cudnnCreateAlgorithmPerformance_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_ALGORITHM_PERFORMANCE] =
        (void *)cudnnSetAlgorithmPerformance_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_ALGORITHM_PERFORMANCE] =
        (void *)cudnnSetAlgorithmPerformance_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_ALGORITHM_PERFORMANCE] =
        (void *)cudnnSetAlgorithmPerformance_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_ALGORITHM_PERFORMANCE] =
        (void *)cudnnGetAlgorithmPerformance_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_ALGORITHM_PERFORMANCE] =
        (void *)cudnnGetAlgorithmPerformance_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_ALGORITHM_PERFORMANCE] =
        (void *)cudnnGetAlgorithmPerformance_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ALGORITHM_PERFORMANCE] =
        (void *)cudnnDestroyAlgorithmPerformance_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ALGORITHM_PERFORMANCE] =
        (void *)cudnnDestroyAlgorithmPerformance_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ALGORITHM_PERFORMANCE] =
        (void *)cudnnDestroyAlgorithmPerformance_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_ALGORITHM_SPACE_SIZE] =
        (void *)cudnnGetAlgorithmSpaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_ALGORITHM_SPACE_SIZE] =
        (void *)cudnnGetAlgorithmSpaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_ALGORITHM_SPACE_SIZE] =
        (void *)cudnnGetAlgorithmSpaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SAVE_ALGORITHM] =
        (void *)cudnnSaveAlgorithm_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SAVE_ALGORITHM] =
        (void *)cudnnSaveAlgorithm_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SAVE_ALGORITHM] =
        (void *)cudnnSaveAlgorithm_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RESTORE_ALGORITHM] =
        (void *)cudnnRestoreAlgorithm_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RESTORE_ALGORITHM] =
        (void *)cudnnRestoreAlgorithm_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RESTORE_ALGORITHM] =
        (void *)cudnnRestoreAlgorithm_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CALLBACK] =
        (void *)cudnnSetCallback_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CALLBACK] =
        (void *)cudnnSetCallback_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CALLBACK] =
        (void *)cudnnSetCallback_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CALLBACK] =
        (void *)cudnnGetCallback_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CALLBACK] =
        (void *)cudnnGetCallback_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CALLBACK] =
        (void *)cudnnGetCallback_posthook;
    cudnn_hook_info.func_prehook[CUDNN_OPS_INFER_VERSION_CHECK] =
        (void *)cudnnOpsInferVersionCheck_prehook;
    cudnn_hook_info.func_proxy[CUDNN_OPS_INFER_VERSION_CHECK] =
        (void *)cudnnOpsInferVersionCheck_proxy;
    cudnn_hook_info.func_posthook[CUDNN_OPS_INFER_VERSION_CHECK] =
        (void *)cudnnOpsInferVersionCheck_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SOFTMAX_BACKWARD] =
        (void *)cudnnSoftmaxBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SOFTMAX_BACKWARD] =
        (void *)cudnnSoftmaxBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SOFTMAX_BACKWARD] =
        (void *)cudnnSoftmaxBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_POOLING_BACKWARD] =
        (void *)cudnnPoolingBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_POOLING_BACKWARD] =
        (void *)cudnnPoolingBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_POOLING_BACKWARD] =
        (void *)cudnnPoolingBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_ACTIVATION_BACKWARD] =
        (void *)cudnnActivationBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_ACTIVATION_BACKWARD] =
        (void *)cudnnActivationBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_ACTIVATION_BACKWARD] =
        (void *)cudnnActivationBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_LRN_CROSS_CHANNEL_BACKWARD] =
        (void *)cudnnLRNCrossChannelBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_LRN_CROSS_CHANNEL_BACKWARD] =
        (void *)cudnnLRNCrossChannelBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_LRN_CROSS_CHANNEL_BACKWARD] =
        (void *)cudnnLRNCrossChannelBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD] =
        (void *)cudnnDivisiveNormalizationBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD] =
        (void *)cudnnDivisiveNormalizationBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DIVISIVE_NORMALIZATION_BACKWARD] =
        (void *)cudnnDivisiveNormalizationBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationBackwardExWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationBackwardExWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationBackwardExWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationTrainingExReserveSpaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationTrainingExReserveSpaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE] =
        (void *)cudnnGetBatchNormalizationTrainingExReserveSpaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING] =
        (void *)cudnnBatchNormalizationForwardTraining_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING] =
        (void *)cudnnBatchNormalizationForwardTraining_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING] =
        (void *)cudnnBatchNormalizationForwardTraining_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX] =
        (void *)cudnnBatchNormalizationForwardTrainingEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX] =
        (void *)cudnnBatchNormalizationForwardTrainingEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX] =
        (void *)cudnnBatchNormalizationForwardTrainingEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_BACKWARD] =
        (void *)cudnnBatchNormalizationBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_BACKWARD] =
        (void *)cudnnBatchNormalizationBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_BACKWARD] =
        (void *)cudnnBatchNormalizationBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX] =
        (void *)cudnnBatchNormalizationBackwardEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX] =
        (void *)cudnnBatchNormalizationBackwardEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BATCH_NORMALIZATION_BACKWARD_EX] =
        (void *)cudnnBatchNormalizationBackwardEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE] =
        (void *)cudnnGetNormalizationForwardTrainingWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE] =
        (void *)cudnnGetNormalizationForwardTrainingWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE] =
        (void *)cudnnGetNormalizationForwardTrainingWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE] =
        (void *)cudnnGetNormalizationBackwardWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE] =
        (void *)cudnnGetNormalizationBackwardWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE] =
        (void *)cudnnGetNormalizationBackwardWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE] =
        (void *)cudnnGetNormalizationTrainingReserveSpaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE] =
        (void *)cudnnGetNormalizationTrainingReserveSpaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE] =
        (void *)cudnnGetNormalizationTrainingReserveSpaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_NORMALIZATION_FORWARD_TRAINING] =
        (void *)cudnnNormalizationForwardTraining_prehook;
    cudnn_hook_info.func_proxy[CUDNN_NORMALIZATION_FORWARD_TRAINING] =
        (void *)cudnnNormalizationForwardTraining_proxy;
    cudnn_hook_info.func_posthook[CUDNN_NORMALIZATION_FORWARD_TRAINING] =
        (void *)cudnnNormalizationForwardTraining_posthook;
    cudnn_hook_info.func_prehook[CUDNN_NORMALIZATION_BACKWARD] =
        (void *)cudnnNormalizationBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_NORMALIZATION_BACKWARD] =
        (void *)cudnnNormalizationBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_NORMALIZATION_BACKWARD] =
        (void *)cudnnNormalizationBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD] =
        (void *)cudnnSpatialTfGridGeneratorBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD] =
        (void *)cudnnSpatialTfGridGeneratorBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD] =
        (void *)cudnnSpatialTfGridGeneratorBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD] =
        (void *)cudnnSpatialTfSamplerBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD] =
        (void *)cudnnSpatialTfSamplerBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SPATIAL_TF_SAMPLER_BACKWARD] =
        (void *)cudnnSpatialTfSamplerBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DROPOUT_BACKWARD] =
        (void *)cudnnDropoutBackward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DROPOUT_BACKWARD] =
        (void *)cudnnDropoutBackward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DROPOUT_BACKWARD] =
        (void *)cudnnDropoutBackward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_OPS_TRAIN_VERSION_CHECK] =
        (void *)cudnnOpsTrainVersionCheck_prehook;
    cudnn_hook_info.func_proxy[CUDNN_OPS_TRAIN_VERSION_CHECK] =
        (void *)cudnnOpsTrainVersionCheck_proxy;
    cudnn_hook_info.func_posthook[CUDNN_OPS_TRAIN_VERSION_CHECK] =
        (void *)cudnnOpsTrainVersionCheck_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_RNN_DESCRIPTOR] =
        (void *)cudnnCreateRNNDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_RNN_DESCRIPTOR] =
        (void *)cudnnCreateRNNDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_RNN_DESCRIPTOR] =
        (void *)cudnnCreateRNNDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_RNN_DESCRIPTOR] =
        (void *)cudnnDestroyRNNDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_RNN_DESCRIPTOR] =
        (void *)cudnnDestroyRNNDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_RNN_DESCRIPTOR] =
        (void *)cudnnDestroyRNNDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_DESCRIPTOR_V8] =
        (void *)cudnnSetRNNDescriptor_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_DESCRIPTOR_V8] =
        (void *)cudnnSetRNNDescriptor_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_DESCRIPTOR_V8] =
        (void *)cudnnSetRNNDescriptor_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_DESCRIPTOR_V8] =
        (void *)cudnnGetRNNDescriptor_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_DESCRIPTOR_V8] =
        (void *)cudnnGetRNNDescriptor_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_DESCRIPTOR_V8] =
        (void *)cudnnGetRNNDescriptor_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_DESCRIPTOR_V6] =
        (void *)cudnnSetRNNDescriptor_v6_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_DESCRIPTOR_V6] =
        (void *)cudnnSetRNNDescriptor_v6_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_DESCRIPTOR_V6] =
        (void *)cudnnSetRNNDescriptor_v6_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_DESCRIPTOR_V6] =
        (void *)cudnnGetRNNDescriptor_v6_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_DESCRIPTOR_V6] =
        (void *)cudnnGetRNNDescriptor_v6_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_DESCRIPTOR_V6] =
        (void *)cudnnGetRNNDescriptor_v6_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_MATRIX_MATH_TYPE] =
        (void *)cudnnSetRNNMatrixMathType_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_MATRIX_MATH_TYPE] =
        (void *)cudnnSetRNNMatrixMathType_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_MATRIX_MATH_TYPE] =
        (void *)cudnnSetRNNMatrixMathType_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_MATRIX_MATH_TYPE] =
        (void *)cudnnGetRNNMatrixMathType_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_MATRIX_MATH_TYPE] =
        (void *)cudnnGetRNNMatrixMathType_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_MATRIX_MATH_TYPE] =
        (void *)cudnnGetRNNMatrixMathType_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_BIAS_MODE] =
        (void *)cudnnSetRNNBiasMode_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_BIAS_MODE] =
        (void *)cudnnSetRNNBiasMode_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_BIAS_MODE] =
        (void *)cudnnSetRNNBiasMode_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_BIAS_MODE] =
        (void *)cudnnGetRNNBiasMode_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_BIAS_MODE] =
        (void *)cudnnGetRNNBiasMode_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_BIAS_MODE] =
        (void *)cudnnGetRNNBiasMode_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_SET_CLIP_V8] =
        (void *)cudnnRNNSetClip_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_SET_CLIP_V8] =
        (void *)cudnnRNNSetClip_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_SET_CLIP_V8] =
        (void *)cudnnRNNSetClip_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_GET_CLIP_V8] =
        (void *)cudnnRNNGetClip_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_GET_CLIP_V8] =
        (void *)cudnnRNNGetClip_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_GET_CLIP_V8] =
        (void *)cudnnRNNGetClip_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_SET_CLIP] =
        (void *)cudnnRNNSetClip_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_SET_CLIP] =
        (void *)cudnnRNNSetClip_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_SET_CLIP] =
        (void *)cudnnRNNSetClip_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_GET_CLIP] =
        (void *)cudnnRNNGetClip_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_GET_CLIP] =
        (void *)cudnnRNNGetClip_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_GET_CLIP] =
        (void *)cudnnRNNGetClip_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_PROJECTION_LAYERS] =
        (void *)cudnnSetRNNProjectionLayers_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_PROJECTION_LAYERS] =
        (void *)cudnnSetRNNProjectionLayers_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_PROJECTION_LAYERS] =
        (void *)cudnnSetRNNProjectionLayers_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_PROJECTION_LAYERS] =
        (void *)cudnnGetRNNProjectionLayers_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_PROJECTION_LAYERS] =
        (void *)cudnnGetRNNProjectionLayers_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_PROJECTION_LAYERS] =
        (void *)cudnnGetRNNProjectionLayers_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_PERSISTENT_RNN_PLAN] =
        (void *)cudnnCreatePersistentRNNPlan_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_PERSISTENT_RNN_PLAN] =
        (void *)cudnnCreatePersistentRNNPlan_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_PERSISTENT_RNN_PLAN] =
        (void *)cudnnCreatePersistentRNNPlan_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_PERSISTENT_RNN_PLAN] =
        (void *)cudnnDestroyPersistentRNNPlan_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_PERSISTENT_RNN_PLAN] =
        (void *)cudnnDestroyPersistentRNNPlan_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_PERSISTENT_RNN_PLAN] =
        (void *)cudnnDestroyPersistentRNNPlan_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_PERSISTENT_RNN_PLAN] =
        (void *)cudnnSetPersistentRNNPlan_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_PERSISTENT_RNN_PLAN] =
        (void *)cudnnSetPersistentRNNPlan_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_PERSISTENT_RNN_PLAN] =
        (void *)cudnnSetPersistentRNNPlan_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BUILD_RNN_DYNAMIC] =
        (void *)cudnnBuildRNNDynamic_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BUILD_RNN_DYNAMIC] =
        (void *)cudnnBuildRNNDynamic_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BUILD_RNN_DYNAMIC] =
        (void *)cudnnBuildRNNDynamic_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_WORKSPACE_SIZE] =
        (void *)cudnnGetRNNWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_WORKSPACE_SIZE] =
        (void *)cudnnGetRNNWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_WORKSPACE_SIZE] =
        (void *)cudnnGetRNNWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE] =
        (void *)cudnnGetRNNTrainingReserveSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE] =
        (void *)cudnnGetRNNTrainingReserveSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_TRAINING_RESERVE_SIZE] =
        (void *)cudnnGetRNNTrainingReserveSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_TEMP_SPACE_SIZES] =
        (void *)cudnnGetRNNTempSpaceSizes_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_TEMP_SPACE_SIZES] =
        (void *)cudnnGetRNNTempSpaceSizes_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_TEMP_SPACE_SIZES] =
        (void *)cudnnGetRNNTempSpaceSizes_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_PARAMS_SIZE] =
        (void *)cudnnGetRNNParamsSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_PARAMS_SIZE] =
        (void *)cudnnGetRNNParamsSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_PARAMS_SIZE] =
        (void *)cudnnGetRNNParamsSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE] =
        (void *)cudnnGetRNNWeightSpaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE] =
        (void *)cudnnGetRNNWeightSpaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_WEIGHT_SPACE_SIZE] =
        (void *)cudnnGetRNNWeightSpaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS] =
        (void *)cudnnGetRNNLinLayerMatrixParams_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS] =
        (void *)cudnnGetRNNLinLayerMatrixParams_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS] =
        (void *)cudnnGetRNNLinLayerMatrixParams_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS] =
        (void *)cudnnGetRNNLinLayerBiasParams_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS] =
        (void *)cudnnGetRNNLinLayerBiasParams_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS] =
        (void *)cudnnGetRNNLinLayerBiasParams_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_WEIGHT_PARAMS] =
        (void *)cudnnGetRNNWeightParams_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_WEIGHT_PARAMS] =
        (void *)cudnnGetRNNWeightParams_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_WEIGHT_PARAMS] =
        (void *)cudnnGetRNNWeightParams_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_INFERENCE] =
        (void *)cudnnRNNForwardInference_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_INFERENCE] =
        (void *)cudnnRNNForwardInference_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_INFERENCE] =
        (void *)cudnnRNNForwardInference_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_PADDING_MODE] =
        (void *)cudnnSetRNNPaddingMode_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_PADDING_MODE] =
        (void *)cudnnSetRNNPaddingMode_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_PADDING_MODE] =
        (void *)cudnnSetRNNPaddingMode_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_PADDING_MODE] =
        (void *)cudnnGetRNNPaddingMode_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_PADDING_MODE] =
        (void *)cudnnGetRNNPaddingMode_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_PADDING_MODE] =
        (void *)cudnnGetRNNPaddingMode_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnCreateRNNDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnCreateRNNDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnCreateRNNDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnDestroyRNNDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnDestroyRNNDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnDestroyRNNDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnSetRNNDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnSetRNNDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnSetRNNDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnGetRNNDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnGetRNNDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_DATA_DESCRIPTOR] =
        (void *)cudnnGetRNNDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_INFERENCE_EX] =
        (void *)cudnnRNNForwardInferenceEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_INFERENCE_EX] =
        (void *)cudnnRNNForwardInferenceEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_INFERENCE_EX] =
        (void *)cudnnRNNForwardInferenceEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD] =
        (void *)cudnnRNNForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD] =
        (void *)cudnnRNNForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD] =
        (void *)cudnnRNNForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnSetRNNAlgorithmDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnSetRNNAlgorithmDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR] =
        (void *)cudnnSetRNNAlgorithmDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNForwardInferenceAlgorithmMaxCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNForwardInferenceAlgorithmMaxCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNForwardInferenceAlgorithmMaxCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX] =
        (void *)cudnnFindRNNForwardInferenceAlgorithmEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX] =
        (void *)cudnnFindRNNForwardInferenceAlgorithmEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX] =
        (void *)cudnnFindRNNForwardInferenceAlgorithmEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnCreateSeqDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnCreateSeqDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnCreateSeqDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnDestroySeqDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnDestroySeqDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnDestroySeqDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnSetSeqDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnSetSeqDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnSetSeqDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnGetSeqDataDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnGetSeqDataDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_SEQ_DATA_DESCRIPTOR] =
        (void *)cudnnGetSeqDataDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_ATTN_DESCRIPTOR] =
        (void *)cudnnCreateAttnDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_ATTN_DESCRIPTOR] =
        (void *)cudnnCreateAttnDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_ATTN_DESCRIPTOR] =
        (void *)cudnnCreateAttnDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_ATTN_DESCRIPTOR] =
        (void *)cudnnDestroyAttnDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_ATTN_DESCRIPTOR] =
        (void *)cudnnDestroyAttnDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_ATTN_DESCRIPTOR] =
        (void *)cudnnDestroyAttnDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_ATTN_DESCRIPTOR] =
        (void *)cudnnSetAttnDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_ATTN_DESCRIPTOR] =
        (void *)cudnnSetAttnDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_ATTN_DESCRIPTOR] =
        (void *)cudnnSetAttnDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_ATTN_DESCRIPTOR] =
        (void *)cudnnGetAttnDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_ATTN_DESCRIPTOR] =
        (void *)cudnnGetAttnDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_ATTN_DESCRIPTOR] =
        (void *)cudnnGetAttnDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS] =
        (void *)cudnnGetMultiHeadAttnBuffers_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS] =
        (void *)cudnnGetMultiHeadAttnBuffers_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS] =
        (void *)cudnnGetMultiHeadAttnBuffers_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS] =
        (void *)cudnnGetMultiHeadAttnWeights_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS] =
        (void *)cudnnGetMultiHeadAttnWeights_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS] =
        (void *)cudnnGetMultiHeadAttnWeights_posthook;
    cudnn_hook_info.func_prehook[CUDNN_MULTI_HEAD_ATTN_FORWARD] =
        (void *)cudnnMultiHeadAttnForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_MULTI_HEAD_ATTN_FORWARD] =
        (void *)cudnnMultiHeadAttnForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_MULTI_HEAD_ATTN_FORWARD] =
        (void *)cudnnMultiHeadAttnForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_ADV_INFER_VERSION_CHECK] =
        (void *)cudnnAdvInferVersionCheck_prehook;
    cudnn_hook_info.func_proxy[CUDNN_ADV_INFER_VERSION_CHECK] =
        (void *)cudnnAdvInferVersionCheck_proxy;
    cudnn_hook_info.func_posthook[CUDNN_ADV_INFER_VERSION_CHECK] =
        (void *)cudnnAdvInferVersionCheck_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_TRAINING] =
        (void *)cudnnRNNForwardTraining_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_TRAINING] =
        (void *)cudnnRNNForwardTraining_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_TRAINING] =
        (void *)cudnnRNNForwardTraining_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_DATA] =
        (void *)cudnnRNNBackwardData_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_DATA] =
        (void *)cudnnRNNBackwardData_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_DATA] =
        (void *)cudnnRNNBackwardData_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_DATA_V8] =
        (void *)cudnnRNNBackwardData_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_DATA_V8] =
        (void *)cudnnRNNBackwardData_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_DATA_V8] =
        (void *)cudnnRNNBackwardData_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_WEIGHTS] =
        (void *)cudnnRNNBackwardWeights_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_WEIGHTS] =
        (void *)cudnnRNNBackwardWeights_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_WEIGHTS] =
        (void *)cudnnRNNBackwardWeights_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_WEIGHTS_V8] =
        (void *)cudnnRNNBackwardWeights_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_WEIGHTS_V8] =
        (void *)cudnnRNNBackwardWeights_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_WEIGHTS_V8] =
        (void *)cudnnRNNBackwardWeights_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_FORWARD_TRAINING_EX] =
        (void *)cudnnRNNForwardTrainingEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_FORWARD_TRAINING_EX] =
        (void *)cudnnRNNForwardTrainingEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_FORWARD_TRAINING_EX] =
        (void *)cudnnRNNForwardTrainingEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_DATA_EX] =
        (void *)cudnnRNNBackwardDataEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_DATA_EX] =
        (void *)cudnnRNNBackwardDataEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_DATA_EX] =
        (void *)cudnnRNNBackwardDataEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_RNN_BACKWARD_WEIGHTS_EX] =
        (void *)cudnnRNNBackwardWeightsEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_RNN_BACKWARD_WEIGHTS_EX] =
        (void *)cudnnRNNBackwardWeightsEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_RNN_BACKWARD_WEIGHTS_EX] =
        (void *)cudnnRNNBackwardWeightsEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNForwardTrainingAlgorithmMaxCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNForwardTrainingAlgorithmMaxCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNForwardTrainingAlgorithmMaxCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX] =
        (void *)cudnnFindRNNForwardTrainingAlgorithmEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX] =
        (void *)cudnnFindRNNForwardTrainingAlgorithmEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX] =
        (void *)cudnnFindRNNForwardTrainingAlgorithmEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNBackwardDataAlgorithmMaxCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNBackwardDataAlgorithmMaxCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNBackwardDataAlgorithmMaxCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX] =
        (void *)cudnnFindRNNBackwardDataAlgorithmEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX] =
        (void *)cudnnFindRNNBackwardDataAlgorithmEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX] =
        (void *)cudnnFindRNNBackwardDataAlgorithmEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNBackwardWeightsAlgorithmMaxCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNBackwardWeightsAlgorithmMaxCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetRNNBackwardWeightsAlgorithmMaxCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX] =
        (void *)cudnnFindRNNBackwardWeightsAlgorithmEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX] =
        (void *)cudnnFindRNNBackwardWeightsAlgorithmEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX] =
        (void *)cudnnFindRNNBackwardWeightsAlgorithmEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA] =
        (void *)cudnnMultiHeadAttnBackwardData_prehook;
    cudnn_hook_info.func_proxy[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA] =
        (void *)cudnnMultiHeadAttnBackwardData_proxy;
    cudnn_hook_info.func_posthook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA] =
        (void *)cudnnMultiHeadAttnBackwardData_posthook;
    cudnn_hook_info.func_prehook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS] =
        (void *)cudnnMultiHeadAttnBackwardWeights_prehook;
    cudnn_hook_info.func_proxy[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS] =
        (void *)cudnnMultiHeadAttnBackwardWeights_proxy;
    cudnn_hook_info.func_posthook[CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS] =
        (void *)cudnnMultiHeadAttnBackwardWeights_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnCreateCTCLossDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnCreateCTCLossDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnCreateCTCLossDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnSetCTCLossDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnSetCTCLossDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnSetCTCLossDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX] =
        (void *)cudnnSetCTCLossDescriptorEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX] =
        (void *)cudnnSetCTCLossDescriptorEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX] =
        (void *)cudnnSetCTCLossDescriptorEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8] =
        (void *)cudnnSetCTCLossDescriptor_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8] =
        (void *)cudnnSetCTCLossDescriptor_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8] =
        (void *)cudnnSetCTCLossDescriptor_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnGetCTCLossDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnGetCTCLossDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnGetCTCLossDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX] =
        (void *)cudnnGetCTCLossDescriptorEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX] =
        (void *)cudnnGetCTCLossDescriptorEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX] =
        (void *)cudnnGetCTCLossDescriptorEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8] =
        (void *)cudnnGetCTCLossDescriptor_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8] =
        (void *)cudnnGetCTCLossDescriptor_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8] =
        (void *)cudnnGetCTCLossDescriptor_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnDestroyCTCLossDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnDestroyCTCLossDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_CTC_LOSS_DESCRIPTOR] =
        (void *)cudnnDestroyCTCLossDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CTC_LOSS] =
        (void *)cudnnCTCLoss_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CTC_LOSS] =
        (void *)cudnnCTCLoss_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CTC_LOSS] =
        (void *)cudnnCTCLoss_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CTC_LOSS_V8] =
        (void *)cudnnCTCLoss_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CTC_LOSS_V8] =
        (void *)cudnnCTCLoss_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CTC_LOSS_V8] =
        (void *)cudnnCTCLoss_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE] =
        (void *)cudnnGetCTCLossWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE] =
        (void *)cudnnGetCTCLossWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE] =
        (void *)cudnnGetCTCLossWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8] =
        (void *)cudnnGetCTCLossWorkspaceSize_v8_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8] =
        (void *)cudnnGetCTCLossWorkspaceSize_v8_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8] =
        (void *)cudnnGetCTCLossWorkspaceSize_v8_posthook;
    cudnn_hook_info.func_prehook[CUDNN_ADV_TRAIN_VERSION_CHECK] =
        (void *)cudnnAdvTrainVersionCheck_prehook;
    cudnn_hook_info.func_proxy[CUDNN_ADV_TRAIN_VERSION_CHECK] =
        (void *)cudnnAdvTrainVersionCheck_proxy;
    cudnn_hook_info.func_posthook[CUDNN_ADV_TRAIN_VERSION_CHECK] =
        (void *)cudnnAdvTrainVersionCheck_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR] =
        (void *)cudnnCreateConvolutionDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR] =
        (void *)cudnnCreateConvolutionDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_CONVOLUTION_DESCRIPTOR] =
        (void *)cudnnCreateConvolutionDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR] =
        (void *)cudnnDestroyConvolutionDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR] =
        (void *)cudnnDestroyConvolutionDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR] =
        (void *)cudnnDestroyConvolutionDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_MATH_TYPE] =
        (void *)cudnnSetConvolutionMathType_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_MATH_TYPE] =
        (void *)cudnnSetConvolutionMathType_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_MATH_TYPE] =
        (void *)cudnnSetConvolutionMathType_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_MATH_TYPE] =
        (void *)cudnnGetConvolutionMathType_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_MATH_TYPE] =
        (void *)cudnnGetConvolutionMathType_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_MATH_TYPE] =
        (void *)cudnnGetConvolutionMathType_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_GROUP_COUNT] =
        (void *)cudnnSetConvolutionGroupCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_GROUP_COUNT] =
        (void *)cudnnSetConvolutionGroupCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_GROUP_COUNT] =
        (void *)cudnnSetConvolutionGroupCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_GROUP_COUNT] =
        (void *)cudnnGetConvolutionGroupCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_GROUP_COUNT] =
        (void *)cudnnGetConvolutionGroupCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_GROUP_COUNT] =
        (void *)cudnnGetConvolutionGroupCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_REORDER_TYPE] =
        (void *)cudnnSetConvolutionReorderType_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_REORDER_TYPE] =
        (void *)cudnnSetConvolutionReorderType_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_REORDER_TYPE] =
        (void *)cudnnSetConvolutionReorderType_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_REORDER_TYPE] =
        (void *)cudnnGetConvolutionReorderType_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_REORDER_TYPE] =
        (void *)cudnnGetConvolutionReorderType_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_REORDER_TYPE] =
        (void *)cudnnGetConvolutionReorderType_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR] =
        (void *)cudnnSetConvolution2dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR] =
        (void *)cudnnSetConvolution2dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR] =
        (void *)cudnnSetConvolution2dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR] =
        (void *)cudnnGetConvolution2dDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR] =
        (void *)cudnnGetConvolution2dDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR] =
        (void *)cudnnGetConvolution2dDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR] =
        (void *)cudnnSetConvolutionNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR] =
        (void *)cudnnSetConvolutionNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR] =
        (void *)cudnnSetConvolutionNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR] =
        (void *)cudnnGetConvolutionNdDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR] =
        (void *)cudnnGetConvolutionNdDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR] =
        (void *)cudnnGetConvolutionNdDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetConvolution2dForwardOutputDim_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetConvolution2dForwardOutputDim_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetConvolution2dForwardOutputDim_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetConvolutionNdForwardOutputDim_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetConvolutionNdForwardOutputDim_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM] =
        (void *)cudnnGetConvolutionNdForwardOutputDim_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionForwardAlgorithmMaxCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionForwardAlgorithmMaxCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionForwardAlgorithmMaxCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionForwardAlgorithm_v7_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionForwardAlgorithm_v7_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionForwardAlgorithm_v7_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM] =
        (void *)cudnnFindConvolutionForwardAlgorithm_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM] =
        (void *)cudnnFindConvolutionForwardAlgorithm_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM] =
        (void *)cudnnFindConvolutionForwardAlgorithm_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionForwardAlgorithmEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionForwardAlgorithmEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionForwardAlgorithmEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_IM_2_COL] =
        (void *)cudnnIm2Col_prehook;
    cudnn_hook_info.func_proxy[CUDNN_IM_2_COL] =
        (void *)cudnnIm2Col_proxy;
    cudnn_hook_info.func_posthook[CUDNN_IM_2_COL] =
        (void *)cudnnIm2Col_posthook;
    cudnn_hook_info.func_prehook[CUDNN_REORDER_FILTER_AND_BIAS] =
        (void *)cudnnReorderFilterAndBias_prehook;
    cudnn_hook_info.func_proxy[CUDNN_REORDER_FILTER_AND_BIAS] =
        (void *)cudnnReorderFilterAndBias_proxy;
    cudnn_hook_info.func_posthook[CUDNN_REORDER_FILTER_AND_BIAS] =
        (void *)cudnnReorderFilterAndBias_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionForwardWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionForwardWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionForwardWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_FORWARD] =
        (void *)cudnnConvolutionForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_FORWARD] =
        (void *)cudnnConvolutionForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_FORWARD] =
        (void *)cudnnConvolutionForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD] =
        (void *)cudnnConvolutionBiasActivationForward_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD] =
        (void *)cudnnConvolutionBiasActivationForward_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD] =
        (void *)cudnnConvolutionBiasActivationForward_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionBackwardDataAlgorithmMaxCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionBackwardDataAlgorithmMaxCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionBackwardDataAlgorithmMaxCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM] =
        (void *)cudnnFindConvolutionBackwardDataAlgorithm_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM] =
        (void *)cudnnFindConvolutionBackwardDataAlgorithm_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM] =
        (void *)cudnnFindConvolutionBackwardDataAlgorithm_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionBackwardDataAlgorithmEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionBackwardDataAlgorithmEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionBackwardDataAlgorithmEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionBackwardDataAlgorithm_v7_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionBackwardDataAlgorithm_v7_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionBackwardDataAlgorithm_v7_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionBackwardDataWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionBackwardDataWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionBackwardDataWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BACKWARD_DATA] =
        (void *)cudnnConvolutionBackwardData_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BACKWARD_DATA] =
        (void *)cudnnConvolutionBackwardData_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BACKWARD_DATA] =
        (void *)cudnnConvolutionBackwardData_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS] =
        (void *)cudnnGetFoldedConvBackwardDataDescriptors_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS] =
        (void *)cudnnGetFoldedConvBackwardDataDescriptors_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS] =
        (void *)cudnnGetFoldedConvBackwardDataDescriptors_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CNN_INFER_VERSION_CHECK] =
        (void *)cudnnCnnInferVersionCheck_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CNN_INFER_VERSION_CHECK] =
        (void *)cudnnCnnInferVersionCheck_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CNN_INFER_VERSION_CHECK] =
        (void *)cudnnCnnInferVersionCheck_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT] =
        (void *)cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM] =
        (void *)cudnnFindConvolutionBackwardFilterAlgorithm_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM] =
        (void *)cudnnFindConvolutionBackwardFilterAlgorithm_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM] =
        (void *)cudnnFindConvolutionBackwardFilterAlgorithm_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionBackwardFilterAlgorithmEx_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionBackwardFilterAlgorithmEx_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX] =
        (void *)cudnnFindConvolutionBackwardFilterAlgorithmEx_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionBackwardFilterAlgorithm_v7_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionBackwardFilterAlgorithm_v7_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7] =
        (void *)cudnnGetConvolutionBackwardFilterAlgorithm_v7_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionBackwardFilterWorkspaceSize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionBackwardFilterWorkspaceSize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE] =
        (void *)cudnnGetConvolutionBackwardFilterWorkspaceSize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BACKWARD_FILTER] =
        (void *)cudnnConvolutionBackwardFilter_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BACKWARD_FILTER] =
        (void *)cudnnConvolutionBackwardFilter_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BACKWARD_FILTER] =
        (void *)cudnnConvolutionBackwardFilter_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CONVOLUTION_BACKWARD_BIAS] =
        (void *)cudnnConvolutionBackwardBias_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CONVOLUTION_BACKWARD_BIAS] =
        (void *)cudnnConvolutionBackwardBias_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CONVOLUTION_BACKWARD_BIAS] =
        (void *)cudnnConvolutionBackwardBias_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK] =
        (void *)cudnnCreateFusedOpsConstParamPack_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK] =
        (void *)cudnnCreateFusedOpsConstParamPack_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK] =
        (void *)cudnnCreateFusedOpsConstParamPack_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK] =
        (void *)cudnnDestroyFusedOpsConstParamPack_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK] =
        (void *)cudnnDestroyFusedOpsConstParamPack_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FUSED_OPS_CONST_PARAM_PACK] =
        (void *)cudnnDestroyFusedOpsConstParamPack_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnSetFusedOpsConstParamPackAttribute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnSetFusedOpsConstParamPackAttribute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnSetFusedOpsConstParamPackAttribute_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnGetFusedOpsConstParamPackAttribute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnGetFusedOpsConstParamPackAttribute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnGetFusedOpsConstParamPackAttribute_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK] =
        (void *)cudnnCreateFusedOpsVariantParamPack_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK] =
        (void *)cudnnCreateFusedOpsVariantParamPack_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK] =
        (void *)cudnnCreateFusedOpsVariantParamPack_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK] =
        (void *)cudnnDestroyFusedOpsVariantParamPack_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK] =
        (void *)cudnnDestroyFusedOpsVariantParamPack_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FUSED_OPS_VARIANT_PARAM_PACK] =
        (void *)cudnnDestroyFusedOpsVariantParamPack_posthook;
    cudnn_hook_info.func_prehook[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnSetFusedOpsVariantParamPackAttribute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnSetFusedOpsVariantParamPackAttribute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_SET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnSetFusedOpsVariantParamPackAttribute_posthook;
    cudnn_hook_info.func_prehook[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnGetFusedOpsVariantParamPackAttribute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnGetFusedOpsVariantParamPackAttribute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE] =
        (void *)cudnnGetFusedOpsVariantParamPackAttribute_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CREATE_FUSED_OPS_PLAN] =
        (void *)cudnnCreateFusedOpsPlan_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CREATE_FUSED_OPS_PLAN] =
        (void *)cudnnCreateFusedOpsPlan_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CREATE_FUSED_OPS_PLAN] =
        (void *)cudnnCreateFusedOpsPlan_posthook;
    cudnn_hook_info.func_prehook[CUDNN_DESTROY_FUSED_OPS_PLAN] =
        (void *)cudnnDestroyFusedOpsPlan_prehook;
    cudnn_hook_info.func_proxy[CUDNN_DESTROY_FUSED_OPS_PLAN] =
        (void *)cudnnDestroyFusedOpsPlan_proxy;
    cudnn_hook_info.func_posthook[CUDNN_DESTROY_FUSED_OPS_PLAN] =
        (void *)cudnnDestroyFusedOpsPlan_posthook;
    cudnn_hook_info.func_prehook[CUDNN_MAKE_FUSED_OPS_PLAN] =
        (void *)cudnnMakeFusedOpsPlan_prehook;
    cudnn_hook_info.func_proxy[CUDNN_MAKE_FUSED_OPS_PLAN] =
        (void *)cudnnMakeFusedOpsPlan_proxy;
    cudnn_hook_info.func_posthook[CUDNN_MAKE_FUSED_OPS_PLAN] =
        (void *)cudnnMakeFusedOpsPlan_posthook;
    cudnn_hook_info.func_prehook[CUDNN_FUSED_OPS_EXECUTE] =
        (void *)cudnnFusedOpsExecute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_FUSED_OPS_EXECUTE] =
        (void *)cudnnFusedOpsExecute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_FUSED_OPS_EXECUTE] =
        (void *)cudnnFusedOpsExecute_posthook;
    cudnn_hook_info.func_prehook[CUDNN_CNN_TRAIN_VERSION_CHECK] =
        (void *)cudnnCnnTrainVersionCheck_prehook;
    cudnn_hook_info.func_proxy[CUDNN_CNN_TRAIN_VERSION_CHECK] =
        (void *)cudnnCnnTrainVersionCheck_proxy;
    cudnn_hook_info.func_posthook[CUDNN_CNN_TRAIN_VERSION_CHECK] =
        (void *)cudnnCnnTrainVersionCheck_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_CREATE_DESCRIPTOR] =
        (void *)cudnnBackendCreateDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_CREATE_DESCRIPTOR] =
        (void *)cudnnBackendCreateDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_CREATE_DESCRIPTOR] =
        (void *)cudnnBackendCreateDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_DESTROY_DESCRIPTOR] =
        (void *)cudnnBackendDestroyDescriptor_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_DESTROY_DESCRIPTOR] =
        (void *)cudnnBackendDestroyDescriptor_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_DESTROY_DESCRIPTOR] =
        (void *)cudnnBackendDestroyDescriptor_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_INITIALIZE] =
        (void *)cudnnBackendInitialize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_INITIALIZE] =
        (void *)cudnnBackendInitialize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_INITIALIZE] =
        (void *)cudnnBackendInitialize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_FINALIZE] =
        (void *)cudnnBackendFinalize_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_FINALIZE] =
        (void *)cudnnBackendFinalize_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_FINALIZE] =
        (void *)cudnnBackendFinalize_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_SET_ATTRIBUTE] =
        (void *)cudnnBackendSetAttribute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_SET_ATTRIBUTE] =
        (void *)cudnnBackendSetAttribute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_SET_ATTRIBUTE] =
        (void *)cudnnBackendSetAttribute_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_GET_ATTRIBUTE] =
        (void *)cudnnBackendGetAttribute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_GET_ATTRIBUTE] =
        (void *)cudnnBackendGetAttribute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_GET_ATTRIBUTE] =
        (void *)cudnnBackendGetAttribute_posthook;
    cudnn_hook_info.func_prehook[CUDNN_BACKEND_EXECUTE] =
        (void *)cudnnBackendExecute_prehook;
    cudnn_hook_info.func_proxy[CUDNN_BACKEND_EXECUTE] =
        (void *)cudnnBackendExecute_proxy;
    cudnn_hook_info.func_posthook[CUDNN_BACKEND_EXECUTE] =
        (void *)cudnnBackendExecute_posthook;
}

/* hook function start */
CUDNN_HOOK_GEN(
    CUDNN_CREATE,
    ,
    cudnnCreate,
    (cudnnHandle_t *handle),
    handle)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY,
    ,
    cudnnDestroy,
    (cudnnHandle_t handle),
    handle)

CUDNN_HOOK_GEN(
    CUDNN_QUERY_RUNTIME_ERROR,
    ,
    cudnnQueryRuntimeError,
    (cudnnHandle_t handle,
    cudnnStatus_t *rstatus,
    cudnnErrQueryMode_t mode,
    cudnnRuntimeTag_t *tag),
    handle, rstatus, mode, tag)

CUDNN_HOOK_GEN(
    CUDNN_GET_PROPERTY,
    ,
    cudnnGetProperty,
    (libraryPropertyType type,
    int *value),
    type, value)

CUDNN_HOOK_GEN(
    CUDNN_SET_STREAM,
    ,
    cudnnSetStream,
    (cudnnHandle_t handle,
    cudaStream_t streamId),
    handle, streamId)

CUDNN_HOOK_GEN(
    CUDNN_GET_STREAM,
    ,
    cudnnGetStream,
    (cudnnHandle_t handle,
    cudaStream_t *streamId),
    handle, streamId)

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
    (cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w),
    tensorDesc, format, dataType, n,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_4D_DESCRIPTOR_EX,
    ,
    cudnnSetTensor4dDescriptorEx,
    (cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int n,
    int c,
    int h,
    int w,
    int nStride,
    int cStride,
    int hStride,
    int wStride),
    tensorDesc, dataType, n, c,
    h, w, nStride, cStride,
    hStride, wStride)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_4D_DESCRIPTOR,
    ,
    cudnnGetTensor4dDescriptor,
    (const cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t *dataType,
    int *n,
    int *c,
    int *h,
    int *w,
    int *nStride,
    int *cStride,
    int *hStride,
    int *wStride),
    tensorDesc, dataType, n, c,
    h, w, nStride, cStride,
    hStride, wStride)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_ND_DESCRIPTOR,
    ,
    cudnnSetTensorNdDescriptor,
    (cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const int strideA[]),
    tensorDesc, dataType, nbDims, dimA,
    strideA)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR_ND_DESCRIPTOR_EX,
    ,
    cudnnSetTensorNdDescriptorEx,
    (cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[]),
    tensorDesc, format, dataType, nbDims,
    dimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_ND_DESCRIPTOR,
    ,
    cudnnGetTensorNdDescriptor,
    (const cudnnTensorDescriptor_t tensorDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    int *nbDims,
    int dimA[],
    int strideA[]),
    tensorDesc, nbDimsRequested, dataType, nbDims,
    dimA, strideA)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_SIZE_IN_BYTES,
    ,
    cudnnGetTensorSizeInBytes,
    (const cudnnTensorDescriptor_t tensorDesc,
    size_t *size),
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
    (const cudnnTensorTransformDescriptor_t transformDesc,
    const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc,
    size_t *destSizeInBytes),
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
    (cudnnTensorTransformDescriptor_t transformDesc,
    const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat,
    const int32_t padBeforeA[],
    const int32_t padAfterA[],
    const uint32_t foldA[],
    const cudnnFoldingDirection_t direction),
    transformDesc, nbDims, destFormat, padBeforeA,
    padAfterA, foldA, direction)

CUDNN_HOOK_GEN(
    CUDNN_GET_TENSOR_TRANSFORM_DESCRIPTOR,
    ,
    cudnnGetTensorTransformDescriptor,
    (cudnnTensorTransformDescriptor_t transformDesc,
    uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat,
    int32_t padBeforeA[],
    int32_t padAfterA[],
    uint32_t foldA[],
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
    CUDNN_TRANSFORM_TENSOR,
    ,
    cudnnTransformTensor,
    (cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, alpha, xDesc, x,
    beta, yDesc, y)

CUDNN_HOOK_GEN(
    CUDNN_TRANSFORM_TENSOR_EX,
    ,
    cudnnTransformTensorEx,
    (cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnTensorDescriptor_t destDesc,
    void *destData),
    handle, transDesc, alpha, srcDesc,
    srcData, beta, destDesc, destData)

CUDNN_HOOK_GEN(
    CUDNN_ADD_TENSOR,
    ,
    cudnnAddTensor,
    (cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C),
    handle, alpha, aDesc, A,
    beta, cDesc, C)

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
    (cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType,
    cudnnNanPropagation_t opTensorNanOpt),
    opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)

CUDNN_HOOK_GEN(
    CUDNN_GET_OP_TENSOR_DESCRIPTOR,
    ,
    cudnnGetOpTensorDescriptor,
    (const cudnnOpTensorDescriptor_t opTensorDesc,
    cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType,
    cudnnNanPropagation_t *opTensorNanOpt),
    opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_OP_TENSOR_DESCRIPTOR,
    ,
    cudnnDestroyOpTensorDescriptor,
    (cudnnOpTensorDescriptor_t opTensorDesc),
    opTensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_OP_TENSOR,
    ,
    cudnnOpTensor,
    (cudnnHandle_t handle,
    const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *alpha2,
    const cudnnTensorDescriptor_t bDesc,
    const void *B,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C),
    handle, opTensorDesc, alpha1, aDesc,
    A, alpha2, bDesc, B,
    beta, cDesc, C)

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
    (cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType,
    cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices,
    cudnnIndicesType_t reduceTensorIndicesType),
    reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
    reduceTensorIndices, reduceTensorIndicesType)

CUDNN_HOOK_GEN(
    CUDNN_GET_REDUCE_TENSOR_DESCRIPTOR,
    ,
    cudnnGetReduceTensorDescriptor,
    (const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType),
    reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt,
    reduceTensorIndices, reduceTensorIndicesType)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_REDUCE_TENSOR_DESCRIPTOR,
    ,
    cudnnDestroyReduceTensorDescriptor,
    (cudnnReduceTensorDescriptor_t reduceTensorDesc),
    reduceTensorDesc)

CUDNN_HOOK_GEN(
    CUDNN_GET_REDUCTION_INDICES_SIZE,
    ,
    cudnnGetReductionIndicesSize,
    (cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes),
    handle, reduceTensorDesc, aDesc, cDesc,
    sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_REDUCTION_WORKSPACE_SIZE,
    ,
    cudnnGetReductionWorkspaceSize,
    (cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes),
    handle, reduceTensorDesc, aDesc, cDesc,
    sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_REDUCE_TENSOR,
    ,
    cudnnReduceTensor,
    (cudnnHandle_t handle,
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices,
    size_t indicesSizeInBytes,
    void *workspace,
    size_t workspaceSizeInBytes,
    const void *alpha,
    const cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const cudnnTensorDescriptor_t cDesc,
    void *C),
    handle, reduceTensorDesc, indices, indicesSizeInBytes,
    workspace, workspaceSizeInBytes, alpha, aDesc,
    A, beta, cDesc, C)

CUDNN_HOOK_GEN(
    CUDNN_SET_TENSOR,
    ,
    cudnnSetTensor,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *valuePtr),
    handle, yDesc, y, valuePtr)

CUDNN_HOOK_GEN(
    CUDNN_SCALE_TENSOR,
    ,
    cudnnScaleTensor,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const void *alpha),
    handle, yDesc, y, alpha)

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
    (cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int k,
    int c,
    int h,
    int w),
    filterDesc, dataType, format, k,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_GET_FILTER_4D_DESCRIPTOR,
    ,
    cudnnGetFilter4dDescriptor,
    (const cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *k,
    int *c,
    int *h,
    int *w),
    filterDesc, dataType, format, k,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_SET_FILTER_ND_DESCRIPTOR,
    ,
    cudnnSetFilterNdDescriptor,
    (cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int nbDims,
    const int filterDimA[]),
    filterDesc, dataType, format, nbDims,
    filterDimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_FILTER_ND_DESCRIPTOR,
    ,
    cudnnGetFilterNdDescriptor,
    (const cudnnFilterDescriptor_t filterDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format,
    int *nbDims,
    int filterDimA[]),
    filterDesc, nbDimsRequested, dataType, format,
    nbDims, filterDimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_FILTER_SIZE_IN_BYTES,
    ,
    cudnnGetFilterSizeInBytes,
    (const cudnnFilterDescriptor_t filterDesc,
    size_t *size),
    filterDesc, size)

CUDNN_HOOK_GEN(
    CUDNN_TRANSFORM_FILTER,
    ,
    cudnnTransformFilter,
    (cudnnHandle_t handle,
    const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha,
    const cudnnFilterDescriptor_t srcDesc,
    const void *srcData,
    const void *beta,
    const cudnnFilterDescriptor_t destDesc,
    void *destData),
    handle, transDesc, alpha, srcDesc,
    srcData, beta, destDesc, destData)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_FILTER_DESCRIPTOR,
    ,
    cudnnDestroyFilterDescriptor,
    (cudnnFilterDescriptor_t filterDesc),
    filterDesc)

CUDNN_HOOK_GEN(
    CUDNN_SOFTMAX_FORWARD,
    ,
    cudnnSoftmaxForward,
    (cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, algo, mode, alpha,
    xDesc, x, beta, yDesc,
    y)

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
    (cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt,
    int windowHeight,
    int windowWidth,
    int verticalPadding,
    int horizontalPadding,
    int verticalStride,
    int horizontalStride),
    poolingDesc, mode, maxpoolingNanOpt, windowHeight,
    windowWidth, verticalPadding, horizontalPadding, verticalStride,
    horizontalStride)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_2D_DESCRIPTOR,
    ,
    cudnnGetPooling2dDescriptor,
    (const cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *windowHeight,
    int *windowWidth,
    int *verticalPadding,
    int *horizontalPadding,
    int *verticalStride,
    int *horizontalStride),
    poolingDesc, mode, maxpoolingNanOpt, windowHeight,
    windowWidth, verticalPadding, horizontalPadding, verticalStride,
    horizontalStride)

CUDNN_HOOK_GEN(
    CUDNN_SET_POOLING_ND_DESCRIPTOR,
    ,
    cudnnSetPoolingNdDescriptor,
    (cudnnPoolingDescriptor_t poolingDesc,
    const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt,
    int nbDims,
    const int windowDimA[],
    const int paddingA[],
    const int strideA[]),
    poolingDesc, mode, maxpoolingNanOpt, nbDims,
    windowDimA, paddingA, strideA)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_ND_DESCRIPTOR,
    ,
    cudnnGetPoolingNdDescriptor,
    (const cudnnPoolingDescriptor_t poolingDesc,
    int nbDimsRequested,
    cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims,
    int windowDimA[],
    int paddingA[],
    int strideA[]),
    poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt,
    nbDims, windowDimA, paddingA, strideA)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_ND_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetPoolingNdForwardOutputDim,
    (const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims,
    int outputTensorDimA[]),
    poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_POOLING_2D_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetPooling2dForwardOutputDim,
    (const cudnnPoolingDescriptor_t poolingDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    int *n,
    int *c,
    int *h,
    int *w),
    poolingDesc, inputTensorDesc, n, c,
    h, w)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_POOLING_DESCRIPTOR,
    ,
    cudnnDestroyPoolingDescriptor,
    (cudnnPoolingDescriptor_t poolingDesc),
    poolingDesc)

CUDNN_HOOK_GEN(
    CUDNN_POOLING_FORWARD,
    ,
    cudnnPoolingForward,
    (cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, poolingDesc, alpha, xDesc,
    x, beta, yDesc, y)

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
    (cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt,
    double coef),
    activationDesc, mode, reluNanOpt, coef)

CUDNN_HOOK_GEN(
    CUDNN_GET_ACTIVATION_DESCRIPTOR,
    ,
    cudnnGetActivationDescriptor,
    (const cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt,
    double *coef),
    activationDesc, mode, reluNanOpt, coef)

CUDNN_HOOK_GEN(
    CUDNN_SET_ACTIVATION_DESCRIPTOR_SWISH_BETA,
    ,
    cudnnSetActivationDescriptorSwishBeta,
    (cudnnActivationDescriptor_t activationDesc,
    double swish_beta),
    activationDesc, swish_beta)

CUDNN_HOOK_GEN(
    CUDNN_GET_ACTIVATION_DESCRIPTOR_SWISH_BETA,
    ,
    cudnnGetActivationDescriptorSwishBeta,
    (cudnnActivationDescriptor_t activationDesc,
    double *swish_beta),
    activationDesc, swish_beta)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_ACTIVATION_DESCRIPTOR,
    ,
    cudnnDestroyActivationDescriptor,
    (cudnnActivationDescriptor_t activationDesc),
    activationDesc)

CUDNN_HOOK_GEN(
    CUDNN_ACTIVATION_FORWARD,
    ,
    cudnnActivationForward,
    (cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, activationDesc, alpha, xDesc,
    x, beta, yDesc, y)

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
    (cudnnLRNDescriptor_t normDesc,
    unsigned lrnN,
    double lrnAlpha,
    double lrnBeta,
    double lrnK),
    normDesc, lrnN, lrnAlpha, lrnBeta,
    lrnK)

CUDNN_HOOK_GEN(
    CUDNN_GET_LRN_DESCRIPTOR,
    ,
    cudnnGetLRNDescriptor,
    (cudnnLRNDescriptor_t normDesc,
    unsigned *lrnN,
    double *lrnAlpha,
    double *lrnBeta,
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
    CUDNN_LRN_CROSS_CHANNEL_FORWARD,
    ,
    cudnnLRNCrossChannelForward,
    (cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, normDesc, lrnMode, alpha,
    xDesc, x, beta, yDesc,
    y)

CUDNN_HOOK_GEN(
    CUDNN_DIVISIVE_NORMALIZATION_FORWARD,
    ,
    cudnnDivisiveNormalizationForward,
    (cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, normDesc, mode, alpha,
    xDesc, x, means, temp,
    temp2, beta, yDesc, y)

CUDNN_HOOK_GEN(
    CUDNN_DERIVE_BN_TENSOR_DESCRIPTOR,
    ,
    cudnnDeriveBNTensorDescriptor,
    (cudnnTensorDescriptor_t derivedBnDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode),
    derivedBnDesc, xDesc, mode)

CUDNN_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_FORWARD_INFERENCE,
    ,
    cudnnBatchNormalizationForwardInference,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    const void *estimatedMean,
    const void *estimatedVariance,
    double epsilon),
    handle, mode, alpha, beta,
    xDesc, x, yDesc, y,
    bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
    estimatedVariance, epsilon)

CUDNN_HOOK_GEN(
    CUDNN_DERIVE_NORM_TENSOR_DESCRIPTOR,
    ,
    cudnnDeriveNormTensorDescriptor,
    (cudnnTensorDescriptor_t derivedNormScaleBiasDesc,
    cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc,
    cudnnNormMode_t mode,
    int groupCnt),
    derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode,
    groupCnt)

CUDNN_HOOK_GEN(
    CUDNN_NORMALIZATION_FORWARD_INFERENCE,
    ,
    cudnnNormalizationForwardInference,
    (cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean,
    const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    double epsilon,
    int groupCnt),
    handle, mode, normOps, algo,
    alpha, beta, xDesc, x,
    normScaleBiasDesc, normScale, normBias, normMeanVarDesc,
    estimatedMean, estimatedVariance, zDesc, z,
    activationDesc, yDesc, y, epsilon,
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
    (cudnnSpatialTransformerDescriptor_t stDesc,
    cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType,
    const int nbDims,
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
    CUDNN_SPATIAL_TF_GRID_GENERATOR_FORWARD,
    ,
    cudnnSpatialTfGridGeneratorForward,
    (cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta,
    void *grid),
    handle, stDesc, theta, grid)

CUDNN_HOOK_GEN(
    CUDNN_SPATIAL_TF_SAMPLER_FORWARD,
    ,
    cudnnSpatialTfSamplerForward,
    (cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *grid,
    const void *beta,
    cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, stDesc, alpha, xDesc,
    x, grid, beta, yDesc,
    y)

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
    CUDNN_DROPOUT_GET_STATES_SIZE,
    ,
    cudnnDropoutGetStatesSize,
    (cudnnHandle_t handle,
    size_t *sizeInBytes),
    handle, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_DROPOUT_GET_RESERVE_SPACE_SIZE,
    ,
    cudnnDropoutGetReserveSpaceSize,
    (cudnnTensorDescriptor_t xdesc,
    size_t *sizeInBytes),
    xdesc, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_SET_DROPOUT_DESCRIPTOR,
    ,
    cudnnSetDropoutDescriptor,
    (cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed),
    dropoutDesc, handle, dropout, states,
    stateSizeInBytes, seed)

CUDNN_HOOK_GEN(
    CUDNN_RESTORE_DROPOUT_DESCRIPTOR,
    ,
    cudnnRestoreDropoutDescriptor,
    (cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed),
    dropoutDesc, handle, dropout, states,
    stateSizeInBytes, seed)

CUDNN_HOOK_GEN(
    CUDNN_GET_DROPOUT_DESCRIPTOR,
    ,
    cudnnGetDropoutDescriptor,
    (cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float *dropout,
    void **states,
    unsigned long long *seed),
    dropoutDesc, handle, dropout, states,
    seed)

CUDNN_HOOK_GEN(
    CUDNN_DROPOUT_FORWARD,
    ,
    cudnnDropoutForward,
    (cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc,
    const void *x,
    const cudnnTensorDescriptor_t ydesc,
    void *y,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, dropoutDesc, xdesc, x,
    ydesc, y, reserveSpace, reserveSpaceSizeInBytes)

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
    (cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t algorithm),
    algoDesc, algorithm)

CUDNN_HOOK_GEN(
    CUDNN_GET_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnGetAlgorithmDescriptor,
    (const cudnnAlgorithmDescriptor_t algoDesc,
    cudnnAlgorithm_t *algorithm),
    algoDesc, algorithm)

CUDNN_HOOK_GEN(
    CUDNN_COPY_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnCopyAlgorithmDescriptor,
    (const cudnnAlgorithmDescriptor_t src,
    cudnnAlgorithmDescriptor_t dest),
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
    (cudnnAlgorithmPerformance_t *algoPerf,
    int numberToCreate),
    algoPerf, numberToCreate)

CUDNN_HOOK_GEN(
    CUDNN_SET_ALGORITHM_PERFORMANCE,
    CUDNN_DEPRECATED,
    cudnnSetAlgorithmPerformance,
    (cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status,
    float time,
    size_t memory),
    algoPerf, algoDesc, status, time,
    memory)

CUDNN_HOOK_GEN(
    CUDNN_GET_ALGORITHM_PERFORMANCE,
    CUDNN_DEPRECATED,
    cudnnGetAlgorithmPerformance,
    (const cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status,
    float *time,
    size_t *memory),
    algoPerf, algoDesc, status, time,
    memory)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_ALGORITHM_PERFORMANCE,
    CUDNN_DEPRECATED,
    cudnnDestroyAlgorithmPerformance,
    (cudnnAlgorithmPerformance_t *algoPerf,
    int numberToDestroy),
    algoPerf, numberToDestroy)

CUDNN_HOOK_GEN(
    CUDNN_GET_ALGORITHM_SPACE_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetAlgorithmSpaceSize,
    (cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes),
    handle, algoDesc, algoSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_SAVE_ALGORITHM,
    CUDNN_DEPRECATED,
    cudnnSaveAlgorithm,
    (cudnnHandle_t handle,
    cudnnAlgorithmDescriptor_t algoDesc,
    void *algoSpace,
    size_t algoSpaceSizeInBytes),
    handle, algoDesc, algoSpace, algoSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_RESTORE_ALGORITHM,
    CUDNN_DEPRECATED,
    cudnnRestoreAlgorithm,
    (cudnnHandle_t handle,
    void *algoSpace,
    size_t algoSpaceSizeInBytes,
    cudnnAlgorithmDescriptor_t algoDesc),
    handle, algoSpace, algoSpaceSizeInBytes, algoDesc)

CUDNN_HOOK_GEN(
    CUDNN_SET_CALLBACK,
    ,
    cudnnSetCallback,
    (unsigned mask,
    void *udata,
    cudnnCallback_t fptr),
    mask, udata, fptr)

CUDNN_HOOK_GEN(
    CUDNN_GET_CALLBACK,
    ,
    cudnnGetCallback,
    (unsigned *mask,
    void **udata,
    cudnnCallback_t *fptr),
    mask, udata, fptr)

CUDNN_HOOK_GEN(
    CUDNN_OPS_INFER_VERSION_CHECK,
    ,
    cudnnOpsInferVersionCheck,
    (),
    )

CUDNN_HOOK_GEN(
    CUDNN_SOFTMAX_BACKWARD,
    ,
    cudnnSoftmaxBackward,
    (cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, algo, mode, alpha,
    yDesc, y, dyDesc, dy,
    beta, dxDesc, dx)

CUDNN_HOOK_GEN(
    CUDNN_POOLING_BACKWARD,
    ,
    cudnnPoolingBackward,
    (cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, poolingDesc, alpha, yDesc,
    y, dyDesc, dy, xDesc,
    x, beta, dxDesc, dx)

CUDNN_HOOK_GEN(
    CUDNN_ACTIVATION_BACKWARD,
    ,
    cudnnActivationBackward,
    (cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, activationDesc, alpha, yDesc,
    y, dyDesc, dy, xDesc,
    x, beta, dxDesc, dx)

CUDNN_HOOK_GEN(
    CUDNN_LRN_CROSS_CHANNEL_BACKWARD,
    ,
    cudnnLRNCrossChannelBackward,
    (cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnLRNMode_t lrnMode,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc,
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, normDesc, lrnMode, alpha,
    yDesc, y, dyDesc, dy,
    xDesc, x, beta, dxDesc,
    dx)

CUDNN_HOOK_GEN(
    CUDNN_DIVISIVE_NORMALIZATION_BACKWARD,
    ,
    cudnnDivisiveNormalizationBackward,
    (cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *means,
    const void *dy,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc,
    void *dx,
    void *dMeans),
    handle, normDesc, mode, alpha,
    xDesc, x, means, dy,
    temp, temp2, beta, dXdMeansDesc,
    dx, dMeans)

CUDNN_HOOK_GEN(
    CUDNN_GET_BATCH_NORMALIZATION_FORWARD_TRAINING_EX_WORKSPACE_SIZE,
    ,
    cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes),
    handle, mode, bnOps, xDesc,
    zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc,
    sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_BATCH_NORMALIZATION_BACKWARD_EX_WORKSPACE_SIZE,
    ,
    cudnnGetBatchNormalizationBackwardExWorkspaceSize,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes),
    handle, mode, bnOps, xDesc,
    yDesc, dyDesc, dzDesc, dxDesc,
    dBnScaleBiasDesc, activationDesc, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_BATCH_NORMALIZATION_TRAINING_EX_RESERVE_SPACE_SIZE,
    ,
    cudnnGetBatchNormalizationTrainingExReserveSpaceSize,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes),
    handle, mode, bnOps, activationDesc,
    xDesc, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING,
    ,
    cudnnBatchNormalizationForwardTraining,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance),
    handle, mode, alpha, beta,
    xDesc, x, yDesc, y,
    bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor,
    resultRunningMean, resultRunningVariance, epsilon, resultSaveMean,
    resultSaveInvVariance)

CUDNN_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_FORWARD_TRAINING_EX,
    ,
    cudnnBatchNormalizationForwardTrainingEx,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, mode, bnOps, alpha,
    beta, xDesc, xData, zDesc,
    zData, yDesc, yData, bnScaleBiasMeanVarDesc,
    bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
    resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance,
    activationDesc, workspace, workSpaceSizeInBytes, reserveSpace,
    reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_BACKWARD,
    ,
    cudnnBatchNormalizationBackward,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale,
    void *dBnScaleResult,
    void *dBnBiasResult,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance),
    handle, mode, alphaDataDiff, betaDataDiff,
    alphaParamDiff, betaParamDiff, xDesc, x,
    dyDesc, dy, dxDesc, dx,
    dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult,
    epsilon, savedMean, savedInvVariance)

CUDNN_HOOK_GEN(
    CUDNN_BATCH_NORMALIZATION_BACKWARD_EX,
    ,
    cudnnBatchNormalizationBackwardEx,
    (cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData,
    const void *bnBiasData,
    void *dBnScaleData,
    void *dBnBiasData,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, mode, bnOps, alphaDataDiff,
    betaDataDiff, alphaParamDiff, betaParamDiff, xDesc,
    xData, yDesc, yData, dyDesc,
    dyData, dzDesc, dzData, dxDesc,
    dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData,
    dBnScaleData, dBnBiasData, epsilon, savedMean,
    savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes,
    reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_NORMALIZATION_FORWARD_TRAINING_WORKSPACE_SIZE,
    ,
    cudnnGetNormalizationForwardTrainingWorkspaceSize,
    (cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt),
    handle, mode, normOps, algo,
    xDesc, zDesc, yDesc, normScaleBiasDesc,
    activationDesc, normMeanVarDesc, sizeInBytes, groupCnt)

CUDNN_HOOK_GEN(
    CUDNN_GET_NORMALIZATION_BACKWARD_WORKSPACE_SIZE,
    ,
    cudnnGetNormalizationBackwardWorkspaceSize,
    (cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    size_t *sizeInBytes,
    int groupCnt),
    handle, mode, normOps, algo,
    xDesc, yDesc, dyDesc, dzDesc,
    dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc,
    sizeInBytes, groupCnt)

CUDNN_HOOK_GEN(
    CUDNN_GET_NORMALIZATION_TRAINING_RESERVE_SPACE_SIZE,
    ,
    cudnnGetNormalizationTrainingReserveSpaceSize,
    (cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    int groupCnt),
    handle, mode, normOps, algo,
    activationDesc, xDesc, sizeInBytes, groupCnt)

CUDNN_HOOK_GEN(
    CUDNN_NORMALIZATION_FORWARD_TRAINING,
    ,
    cudnnNormalizationForwardTraining,
    (cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha,
    const void *beta,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale,
    const void *normBias,
    double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    void *resultRunningMean,
    void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean,
    void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
    int groupCnt),
    handle, mode, normOps, algo,
    alpha, beta, xDesc, xData,
    normScaleBiasDesc, normScale, normBias, exponentialAverageFactor,
    normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon,
    resultSaveMean, resultSaveInvVariance, activationDesc, zDesc,
    zData, yDesc, yData, workspace,
    workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt)

CUDNN_HOOK_GEN(
    CUDNN_NORMALIZATION_BACKWARD,
    ,
    cudnnNormalizationBackward,
    (cudnnHandle_t handle,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const void *normScaleData,
    const void *normBiasData,
    void *dNormScaleData,
    void *dNormBiasData,
    double epsilon,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *savedMean,
    const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes,
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

CUDNN_HOOK_GEN(
    CUDNN_SPATIAL_TF_GRID_GENERATOR_BACKWARD,
    ,
    cudnnSpatialTfGridGeneratorBackward,
    (cudnnHandle_t handle,
    const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid,
    void *dtheta),
    handle, stDesc, dgrid, dtheta)

CUDNN_HOOK_GEN(
    CUDNN_SPATIAL_TF_SAMPLER_BACKWARD,
    ,
    cudnnSpatialTfSamplerBackward,
    (cudnnHandle_t handle,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const void *alphaDgrid,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *grid,
    const void *betaDgrid,
    void *dgrid),
    handle, stDesc, alpha, xDesc,
    x, beta, dxDesc, dx,
    alphaDgrid, dyDesc, dy, grid,
    betaDgrid, dgrid)

CUDNN_HOOK_GEN(
    CUDNN_DROPOUT_BACKWARD,
    ,
    cudnnDropoutBackward,
    (cudnnHandle_t handle,
    const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxdesc,
    void *dx,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, dropoutDesc, dydesc, dy,
    dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)

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
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t algo,
    cudnnRNNMode_t cellMode,
    cudnnRNNBiasMode_t biasMode,
    cudnnDirectionMode_t dirMode,
    cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType,
    cudnnDataType_t mathPrec,
    cudnnMathType_t mathType,
    int32_t inputSize,
    int32_t hiddenSize,
    int32_t projSize,
    int32_t numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags),
    rnnDesc, algo, cellMode, biasMode,
    dirMode, inputMode, dataType, mathPrec,
    mathType, inputSize, hiddenSize, projSize,
    numLayers, dropoutDesc, auxFlags)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_DESCRIPTOR_V8,
    ,
    cudnnGetRNNDescriptor_v8,
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode,
    cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode,
    cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType,
    cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType,
    int32_t *inputSize,
    int32_t *hiddenSize,
    int32_t *projSize,
    int32_t *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    uint32_t *auxFlags),
    rnnDesc, algo, cellMode, biasMode,
    dirMode, inputMode, dataType, mathPrec,
    mathType, inputSize, hiddenSize, projSize,
    numLayers, dropoutDesc, auxFlags)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_DESCRIPTOR_V6,
    CUDNN_DEPRECATED,
    cudnnSetRNNDescriptor_v6,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize,
    const int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction,
    cudnnRNNMode_t cellMode,
    cudnnRNNAlgo_t algo,
    cudnnDataType_t mathPrec),
    handle, rnnDesc, hiddenSize, numLayers,
    dropoutDesc, inputMode, direction, cellMode,
    algo, mathPrec)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_DESCRIPTOR_V6,
    CUDNN_DEPRECATED,
    cudnnGetRNNDescriptor_v6,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int *hiddenSize,
    int *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc,
    cudnnRNNInputMode_t *inputMode,
    cudnnDirectionMode_t *direction,
    cudnnRNNMode_t *cellMode,
    cudnnRNNAlgo_t *algo,
    cudnnDataType_t *mathPrec),
    handle, rnnDesc, hiddenSize, numLayers,
    dropoutDesc, inputMode, direction, cellMode,
    algo, mathPrec)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_MATRIX_MATH_TYPE,
    CUDNN_DEPRECATED,
    cudnnSetRNNMatrixMathType,
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t mType),
    rnnDesc, mType)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_MATRIX_MATH_TYPE,
    CUDNN_DEPRECATED,
    cudnnGetRNNMatrixMathType,
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnMathType_t *mType),
    rnnDesc, mType)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_BIAS_MODE,
    CUDNN_DEPRECATED,
    cudnnSetRNNBiasMode,
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t biasMode),
    rnnDesc, biasMode)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_BIAS_MODE,
    CUDNN_DEPRECATED,
    cudnnGetRNNBiasMode,
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNBiasMode_t *biasMode),
    rnnDesc, biasMode)

CUDNN_HOOK_GEN(
    CUDNN_RNN_SET_CLIP_V8,
    ,
    cudnnRNNSetClip_v8,
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip),
    rnnDesc, clipMode, clipNanOpt, lclip,
    rclip)

CUDNN_HOOK_GEN(
    CUDNN_RNN_GET_CLIP_V8,
    ,
    cudnnRNNGetClip_v8,
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip),
    rnnDesc, clipMode, clipNanOpt, lclip,
    rclip)

CUDNN_HOOK_GEN(
    CUDNN_RNN_SET_CLIP,
    CUDNN_DEPRECATED,
    cudnnRNNSetClip,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t clipMode,
    cudnnNanPropagation_t clipNanOpt,
    double lclip,
    double rclip),
    handle, rnnDesc, clipMode, clipNanOpt,
    lclip, rclip)

CUDNN_HOOK_GEN(
    CUDNN_RNN_GET_CLIP,
    CUDNN_DEPRECATED,
    cudnnRNNGetClip,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNClipMode_t *clipMode,
    cudnnNanPropagation_t *clipNanOpt,
    double *lclip,
    double *rclip),
    handle, rnnDesc, clipMode, clipNanOpt,
    lclip, rclip)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_PROJECTION_LAYERS,
    CUDNN_DEPRECATED,
    cudnnSetRNNProjectionLayers,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int recProjSize,
    const int outProjSize),
    handle, rnnDesc, recProjSize, outProjSize)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_PROJECTION_LAYERS,
    CUDNN_DEPRECATED,
    cudnnGetRNNProjectionLayers,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *recProjSize,
    int *outProjSize),
    handle, rnnDesc, recProjSize, outProjSize)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_PERSISTENT_RNN_PLAN,
    CUDNN_DEPRECATED,
    cudnnCreatePersistentRNNPlan,
    (cudnnRNNDescriptor_t rnnDesc,
    const int minibatch,
    const cudnnDataType_t dataType,
    cudnnPersistentRNNPlan_t *plan),
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
    (cudnnRNNDescriptor_t rnnDesc,
    cudnnPersistentRNNPlan_t plan),
    rnnDesc, plan)

CUDNN_HOOK_GEN(
    CUDNN_BUILD_RNN_DYNAMIC,
    ,
    cudnnBuildRNNDynamic,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int miniBatch),
    handle, rnnDesc, miniBatch)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_WORKSPACE_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetRNNWorkspaceSize,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_TRAINING_RESERVE_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetRNNTrainingReserveSize,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_TEMP_SPACE_SIZES,
    ,
    cudnnGetRNNTempSpaceSizes,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fMode,
    cudnnRNNDataDescriptor_t xDesc,
    size_t *workSpaceSize,
    size_t *reserveSpaceSize),
    handle, rnnDesc, fMode, xDesc,
    workSpaceSize, reserveSpaceSize)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_PARAMS_SIZE,
    CUDNN_DEPRECATED,
    cudnnGetRNNParamsSize,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes,
    cudnnDataType_t dataType),
    handle, rnnDesc, xDesc, sizeInBytes,
    dataType)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_WEIGHT_SPACE_SIZE,
    ,
    cudnnGetRNNWeightSpaceSize,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    size_t *weightSpaceSize),
    handle, rnnDesc, weightSpaceSize)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_LIN_LAYER_MATRIX_PARAMS,
    CUDNN_DEPRECATED,
    cudnnGetRNNLinLayerMatrixParams,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc,
    void **linLayerMat),
    handle, rnnDesc, pseudoLayer, xDesc,
    wDesc, w, linLayerID, linLayerMatDesc,
    linLayerMat)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_LIN_LAYER_BIAS_PARAMS,
    CUDNN_DEPRECATED,
    cudnnGetRNNLinLayerBiasParams,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc,
    void **linLayerBias),
    handle, rnnDesc, pseudoLayer, xDesc,
    wDesc, w, linLayerID, linLayerBiasDesc,
    linLayerBias)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_WEIGHT_PARAMS,
    ,
    cudnnGetRNNWeightParams,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int32_t pseudoLayer,
    size_t weightSpaceSize,
    const void *weightSpace,
    int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc,
    void **mAddr,
    cudnnTensorDescriptor_t bDesc,
    void **bAddr),
    handle, rnnDesc, pseudoLayer, weightSpaceSize,
    weightSpace, linLayerID, mDesc, mAddr,
    bDesc, bAddr)

CUDNN_HOOK_GEN(
    CUDNN_RNN_FORWARD_INFERENCE,
    CUDNN_DEPRECATED,
    cudnnRNNForwardInference,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, workSpace, workSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_PADDING_MODE,
    CUDNN_DEPRECATED,
    cudnnSetRNNPaddingMode,
    (cudnnRNNDescriptor_t rnnDesc,
    unsigned paddingMode),
    rnnDesc, paddingMode)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_PADDING_MODE,
    CUDNN_DEPRECATED,
    cudnnGetRNNPaddingMode,
    (cudnnRNNDescriptor_t rnnDesc,
    unsigned *paddingMode),
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
    (cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout,
    int maxSeqLength,
    int batchSize,
    int vectorSize,
    const int seqLengthArray[],
    void *paddingFill),
    rnnDataDesc, dataType, layout, maxSeqLength,
    batchSize, vectorSize, seqLengthArray, paddingFill)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_DATA_DESCRIPTOR,
    ,
    cudnnGetRNNDataDescriptor,
    (cudnnRNNDataDescriptor_t rnnDataDesc,
    cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout,
    int *maxSeqLength,
    int *batchSize,
    int *vectorSize,
    int arrayLengthRequested,
    int seqLengthArray[],
    void *paddingFill),
    rnnDataDesc, dataType, layout, maxSeqLength,
    batchSize, vectorSize, arrayLengthRequested, seqLengthArray,
    paddingFill)

CUDNN_HOOK_GEN(
    CUDNN_RNN_FORWARD_INFERENCE_EX,
    CUDNN_DEPRECATED,
    cudnnRNNForwardInferenceEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, rnnDesc, xDesc, x,
    hxDesc, hx, cxDesc, cx,
    wDesc, w, yDesc, y,
    hyDesc, hy, cyDesc, cy,
    kDesc, keys, cDesc, cAttn,
    iDesc, iAttn, qDesc, queries,
    workSpace, workSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_RNN_FORWARD,
    ,
    cudnnRNNForward,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnRNNDataDescriptor_t yDesc,
    void *y,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    void *hy,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    void *cy,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace),
    handle, rnnDesc, fwdMode, devSeqLengths,
    xDesc, x, yDesc, y,
    hDesc, hx, hy, cDesc,
    cx, cy, weightSpaceSize, weightSpace,
    workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)

CUDNN_HOOK_GEN(
    CUDNN_SET_RNN_ALGORITHM_DESCRIPTOR,
    CUDNN_DEPRECATED,
    cudnnSetRNNAlgorithmDescriptor,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc),
    handle, rnnDesc, algoDesc)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_FORWARD_INFERENCE_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNForwardInferenceAlgorithmMaxCount,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HOOK_GEN(
    CUDNN_FIND_RNN_FORWARD_INFERENCE_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNForwardInferenceAlgorithmEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
    perfResults, workspace, workSpaceSizeInBytes)

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
    (cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t dataType,
    int nbDims,
    const int dimA[],
    const cudnnSeqDataAxis_t axes[],
    size_t seqLengthArraySize,
    const int seqLengthArray[],
    void *paddingFill),
    seqDataDesc, dataType, nbDims, dimA,
    axes, seqLengthArraySize, seqLengthArray, paddingFill)

CUDNN_HOOK_GEN(
    CUDNN_GET_SEQ_DATA_DESCRIPTOR,
    ,
    cudnnGetSeqDataDescriptor,
    (const cudnnSeqDataDescriptor_t seqDataDesc,
    cudnnDataType_t *dataType,
    int *nbDims,
    int nbDimsRequested,
    int dimA[],
    cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize,
    size_t seqLengthSizeRequested,
    int seqLengthArray[],
    void *paddingFill),
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
    (cudnnAttnDescriptor_t attnDesc,
    unsigned attnMode,
    int nHeads,
    double smScaler,
    cudnnDataType_t dataType,
    cudnnDataType_t computePrec,
    cudnnMathType_t mathType,
    cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc,
    int qSize,
    int kSize,
    int vSize,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int oProjSize,
    int qoMaxSeqLength,
    int kvMaxSeqLength,
    int maxBatchSize,
    int maxBeamSize),
    attnDesc, attnMode, nHeads, smScaler,
    dataType, computePrec, mathType, attnDropoutDesc,
    postDropoutDesc, qSize, kSize, vSize,
    qProjSize, kProjSize, vProjSize, oProjSize,
    qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)

CUDNN_HOOK_GEN(
    CUDNN_GET_ATTN_DESCRIPTOR,
    ,
    cudnnGetAttnDescriptor,
    (cudnnAttnDescriptor_t attnDesc,
    unsigned *attnMode,
    int *nHeads,
    double *smScaler,
    cudnnDataType_t *dataType,
    cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType,
    cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc,
    int *qSize,
    int *kSize,
    int *vSize,
    int *qProjSize,
    int *kProjSize,
    int *vProjSize,
    int *oProjSize,
    int *qoMaxSeqLength,
    int *kvMaxSeqLength,
    int *maxBatchSize,
    int *maxBeamSize),
    attnDesc, attnMode, nHeads, smScaler,
    dataType, computePrec, mathType, attnDropoutDesc,
    postDropoutDesc, qSize, kSize, vSize,
    qProjSize, kProjSize, vProjSize, oProjSize,
    qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize)

CUDNN_HOOK_GEN(
    CUDNN_GET_MULTI_HEAD_ATTN_BUFFERS,
    ,
    cudnnGetMultiHeadAttnBuffers,
    (cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes,
    size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes),
    handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes,
    reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_MULTI_HEAD_ATTN_WEIGHTS,
    ,
    cudnnGetMultiHeadAttnWeights,
    (cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind,
    size_t weightSizeInBytes,
    const void *weights,
    cudnnTensorDescriptor_t wDesc,
    void **wAddr),
    handle, attnDesc, wKind, weightSizeInBytes,
    weights, wDesc, wAddr)

CUDNN_HOOK_GEN(
    CUDNN_MULTI_HEAD_ATTN_FORWARD,
    ,
    cudnnMultiHeadAttnForward,
    (cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    int currIdx,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsQO[],
    const int devSeqLengthsKV[],
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t oDesc,
    void *out,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace),
    handle, attnDesc, currIdx, loWinIdx,
    hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc,
    queries, residuals, kDesc, keys,
    vDesc, values, oDesc, out,
    weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace,
    reserveSpaceSizeInBytes, reserveSpace)

CUDNN_HOOK_GEN(
    CUDNN_ADV_INFER_VERSION_CHECK,
    ,
    cudnnAdvInferVersionCheck,
    (),
    )

CUDNN_HOOK_GEN(
    CUDNN_RNN_FORWARD_TRAINING,
    CUDNN_DEPRECATED,
    cudnnRNNForwardTraining,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, workSpace, workSpaceSizeInBytes, reserveSpace,
    reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_RNN_BACKWARD_DATA,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardData,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, yDesc,
    y, dyDesc, dy, dhyDesc,
    dhy, dcyDesc, dcy, wDesc,
    w, hxDesc, hx, cxDesc,
    cx, dxDesc, dx, dhxDesc,
    dhx, dcxDesc, dcx, workSpace,
    workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_RNN_BACKWARD_DATA_V8,
    ,
    cudnnRNNBackwardData_v8,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const void *dy,
    cudnnRNNDataDescriptor_t xDesc,
    void *dx,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    const void *dhy,
    void *dhx,
    cudnnTensorDescriptor_t cDesc,
    const void *cx,
    const void *dcy,
    void *dcx,
    size_t weightSpaceSize,
    const void *weightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace),
    handle, rnnDesc, devSeqLengths, yDesc,
    y, dy, xDesc, dx,
    hDesc, hx, dhy, dhx,
    cDesc, cx, dcy, dcx,
    weightSpaceSize, weightSpace, workSpaceSize, workSpace,
    reserveSpaceSize, reserveSpace)

CUDNN_HOOK_GEN(
    CUDNN_RNN_BACKWARD_WEIGHTS,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardWeights,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, yDesc,
    y, workSpace, workSpaceSizeInBytes, dwDesc,
    dw, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_RNN_BACKWARD_WEIGHTS_V8,
    ,
    cudnnRNNBackwardWeights_v8,
    (cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad,
    const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    cudnnTensorDescriptor_t hDesc,
    const void *hx,
    cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    size_t weightSpaceSize,
    void *dweightSpace,
    size_t workSpaceSize,
    void *workSpace,
    size_t reserveSpaceSize,
    void *reserveSpace),
    handle, rnnDesc, addGrad, devSeqLengths,
    xDesc, x, hDesc, hx,
    yDesc, y, weightSpaceSize, dweightSpace,
    workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)

CUDNN_HOOK_GEN(
    CUDNN_RNN_FORWARD_TRAINING_EX,
    CUDNN_DEPRECATED,
    cudnnRNNForwardTrainingEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnRNNDataDescriptor_t yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const cudnnRNNDataDescriptor_t kDesc,
    const void *keys,
    const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc,
    void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc,
    void *queries,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, xDesc, x,
    hxDesc, hx, cxDesc, cx,
    wDesc, w, yDesc, y,
    hyDesc, hy, cyDesc, cy,
    kDesc, keys, cDesc, cAttn,
    iDesc, iAttn, qDesc, queries,
    workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_RNN_BACKWARD_DATA_EX,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardDataEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    const cudnnRNNDataDescriptor_t dyDesc,
    const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc,
    const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc,
    void *dkeys,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, yDesc, y,
    dyDesc, dy, dcDesc, dcAttn,
    dhyDesc, dhy, dcyDesc, dcy,
    wDesc, w, hxDesc, hx,
    cxDesc, cx, dxDesc, dx,
    dhxDesc, dhx, dcxDesc, dcx,
    dkDesc, dkeys, workSpace, workSpaceSizeInBytes,
    reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_RNN_BACKWARD_WEIGHTS_EX,
    CUDNN_DEPRECATED,
    cudnnRNNBackwardWeightsEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnRNNDataDescriptor_t yDesc,
    const void *y,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, xDesc, x,
    hxDesc, hx, yDesc, y,
    workSpace, workSpaceSizeInBytes, dwDesc, dw,
    reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_FORWARD_TRAINING_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNForwardTrainingAlgorithmMaxCount,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HOOK_GEN(
    CUDNN_FIND_RNN_FORWARD_TRAINING_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNForwardTrainingAlgorithmEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, cxDesc,
    cx, wDesc, w, yDesc,
    y, hyDesc, hy, cyDesc,
    cy, findIntensity, requestedAlgoCount, returnedAlgoCount,
    perfResults, workspace, workSpaceSizeInBytes, reserveSpace,
    reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_BACKWARD_DATA_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNBackwardDataAlgorithmMaxCount,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HOOK_GEN(
    CUDNN_FIND_RNN_BACKWARD_DATA_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNBackwardDataAlgorithmEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const cudnnTensorDescriptor_t *dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnTensorDescriptor_t *dxDesc,
    void *dx,
    const cudnnTensorDescriptor_t dhxDesc,
    void *dhx,
    const cudnnTensorDescriptor_t dcxDesc,
    void *dcx,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, yDesc,
    y, dyDesc, dy, dhyDesc,
    dhy, dcyDesc, dcy, wDesc,
    w, hxDesc, hx, cxDesc,
    cx, dxDesc, dx, dhxDesc,
    dhx, dcxDesc, dcx, findIntensity,
    requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
    workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_RNN_BACKWARD_WEIGHTS_ALGORITHM_MAX_COUNT,
    CUDNN_DEPRECATED,
    cudnnGetRNNBackwardWeightsAlgorithmMaxCount,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    int *count),
    handle, rnnDesc, count)

CUDNN_HOOK_GEN(
    CUDNN_FIND_RNN_BACKWARD_WEIGHTS_ALGORITHM_EX,
    CUDNN_DEPRECATED,
    cudnnFindRNNBackwardWeightsAlgorithmEx,
    (cudnnHandle_t handle,
    const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength,
    const cudnnTensorDescriptor_t *xDesc,
    const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t *yDesc,
    const void *y,
    const float findIntensity,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace,
    size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const void *reserveSpace,
    size_t reserveSpaceSizeInBytes),
    handle, rnnDesc, seqLength, xDesc,
    x, hxDesc, hx, yDesc,
    y, findIntensity, requestedAlgoCount, returnedAlgoCount,
    perfResults, workspace, workSpaceSizeInBytes, dwDesc,
    dw, reserveSpace, reserveSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_MULTI_HEAD_ATTN_BACKWARD_DATA,
    ,
    cudnnMultiHeadAttnBackwardData,
    (cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[],
    const int hiWinIdx[],
    const int devSeqLengthsDQDO[],
    const int devSeqLengthsDKDV[],
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc,
    void *dqueries,
    const void *queries,
    const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys,
    const void *keys,
    const cudnnSeqDataDescriptor_t dvDesc,
    void *dvalues,
    const void *values,
    size_t weightSizeInBytes,
    const void *weights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace),
    handle, attnDesc, loWinIdx, hiWinIdx,
    devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout,
    dqDesc, dqueries, queries, dkDesc,
    dkeys, keys, dvDesc, dvalues,
    values, weightSizeInBytes, weights, workSpaceSizeInBytes,
    workSpace, reserveSpaceSizeInBytes, reserveSpace)

CUDNN_HOOK_GEN(
    CUDNN_MULTI_HEAD_ATTN_BACKWARD_WEIGHTS,
    ,
    cudnnMultiHeadAttnBackwardWeights,
    (cudnnHandle_t handle,
    const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc,
    const void *queries,
    const cudnnSeqDataDescriptor_t kDesc,
    const void *keys,
    const cudnnSeqDataDescriptor_t vDesc,
    const void *values,
    const cudnnSeqDataDescriptor_t doDesc,
    const void *dout,
    size_t weightSizeInBytes,
    const void *weights,
    void *dweights,
    size_t workSpaceSizeInBytes,
    void *workSpace,
    size_t reserveSpaceSizeInBytes,
    void *reserveSpace),
    handle, attnDesc, addGrad, qDesc,
    queries, kDesc, keys, vDesc,
    values, doDesc, dout, weightSizeInBytes,
    weights, dweights, workSpaceSizeInBytes, workSpace,
    reserveSpaceSizeInBytes, reserveSpace)

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
    (cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType),
    ctcLossDesc, compType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CTC_LOSS_DESCRIPTOR_EX,
    ,
    cudnnSetCTCLossDescriptorEx,
    (cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode),
    ctcLossDesc, compType, normMode, gradMode)

CUDNN_HOOK_GEN(
    CUDNN_SET_CTC_LOSS_DESCRIPTOR_V8,
    ,
    cudnnSetCTCLossDescriptor_v8,
    (cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode,
    cudnnNanPropagation_t gradMode,
    int maxLabelLength),
    ctcLossDesc, compType, normMode, gradMode,
    maxLabelLength)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_DESCRIPTOR,
    ,
    cudnnGetCTCLossDescriptor,
    (cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType),
    ctcLossDesc, compType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_DESCRIPTOR_EX,
    ,
    cudnnGetCTCLossDescriptorEx,
    (cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode),
    ctcLossDesc, compType, normMode, gradMode)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_DESCRIPTOR_V8,
    ,
    cudnnGetCTCLossDescriptor_v8,
    (cudnnCTCLossDescriptor_t ctcLossDesc,
    cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode,
    cudnnNanPropagation_t *gradMode,
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
    CUDNN_CTC_LOSS,
    ,
    cudnnCTCLoss,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int hostLabels[],
    const int hostLabelLengths[],
    const int hostInputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace,
    size_t workSpaceSizeInBytes),
    handle, probsDesc, probs, hostLabels,
    hostLabelLengths, hostInputLengths, costs, gradientsDesc,
    gradients, algo, ctcLossDesc, workspace,
    workSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_CTC_LOSS_V8,
    ,
    cudnnCTCLoss_v8,
    (cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const void *probs,
    const int labels[],
    const int labelLengths[],
    const int inputLengths[],
    void *costs,
    const cudnnTensorDescriptor_t gradientsDesc,
    void *gradients,
    size_t workSpaceSizeInBytes,
    void *workspace),
    handle, algo, ctcLossDesc, probsDesc,
    probs, labels, labelLengths, inputLengths,
    costs, gradientsDesc, gradients, workSpaceSizeInBytes,
    workspace)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE,
    ,
    cudnnGetCTCLossWorkspaceSize,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    const int *labels,
    const int *labelLengths,
    const int *inputLengths,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes),
    handle, probsDesc, gradientsDesc, labels,
    labelLengths, inputLengths, algo, ctcLossDesc,
    sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_CTC_LOSS_WORKSPACE_SIZE_V8,
    ,
    cudnnGetCTCLossWorkspaceSize_v8,
    (cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc,
    size_t *sizeInBytes),
    handle, algo, ctcLossDesc, probsDesc,
    gradientsDesc, sizeInBytes)

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
    (cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t mathType),
    convDesc, mathType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_MATH_TYPE,
    ,
    cudnnGetConvolutionMathType,
    (cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t *mathType),
    convDesc, mathType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_GROUP_COUNT,
    ,
    cudnnSetConvolutionGroupCount,
    (cudnnConvolutionDescriptor_t convDesc,
    int groupCount),
    convDesc, groupCount)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_GROUP_COUNT,
    ,
    cudnnGetConvolutionGroupCount,
    (cudnnConvolutionDescriptor_t convDesc,
    int *groupCount),
    convDesc, groupCount)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_REORDER_TYPE,
    ,
    cudnnSetConvolutionReorderType,
    (cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t reorderType),
    convDesc, reorderType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_REORDER_TYPE,
    ,
    cudnnGetConvolutionReorderType,
    (cudnnConvolutionDescriptor_t convDesc,
    cudnnReorderType_t *reorderType),
    convDesc, reorderType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR,
    ,
    cudnnSetConvolution2dDescriptor,
    (cudnnConvolutionDescriptor_t convDesc,
    int pad_h,
    int pad_w,
    int u,
    int v,
    int dilation_h,
    int dilation_w,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType),
    convDesc, pad_h, pad_w, u,
    v, dilation_h, dilation_w, mode,
    computeType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_2D_DESCRIPTOR,
    ,
    cudnnGetConvolution2dDescriptor,
    (const cudnnConvolutionDescriptor_t convDesc,
    int *pad_h,
    int *pad_w,
    int *u,
    int *v,
    int *dilation_h,
    int *dilation_w,
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType),
    convDesc, pad_h, pad_w, u,
    v, dilation_h, dilation_w, mode,
    computeType)

CUDNN_HOOK_GEN(
    CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR,
    ,
    cudnnSetConvolutionNdDescriptor,
    (cudnnConvolutionDescriptor_t convDesc,
    int arrayLength,
    const int padA[],
    const int filterStrideA[],
    const int dilationA[],
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType),
    convDesc, arrayLength, padA, filterStrideA,
    dilationA, mode, computeType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_ND_DESCRIPTOR,
    ,
    cudnnGetConvolutionNdDescriptor,
    (const cudnnConvolutionDescriptor_t convDesc,
    int arrayLengthRequested,
    int *arrayLength,
    int padA[],
    int strideA[],
    int dilationA[],
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType),
    convDesc, arrayLengthRequested, arrayLength, padA,
    strideA, dilationA, mode, computeType)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_2D_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetConvolution2dForwardOutputDim,
    (const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int *n,
    int *c,
    int *h,
    int *w),
    convDesc, inputTensorDesc, filterDesc, n,
    c, h, w)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_ND_FORWARD_OUTPUT_DIM,
    ,
    cudnnGetConvolutionNdForwardOutputDim,
    (const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int nbDims,
    int tensorOuputDimA[]),
    convDesc, inputTensorDesc, filterDesc, nbDims,
    tensorOuputDimA)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_MAX_COUNT,
    ,
    cudnnGetConvolutionForwardAlgorithmMaxCount,
    (cudnnHandle_t handle,
    int *count),
    handle, count)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM_V7,
    ,
    cudnnGetConvolutionForwardAlgorithm_v7,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults),
    handle, srcDesc, filterDesc, convDesc,
    destDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM,
    ,
    cudnnFindConvolutionForwardAlgorithm,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults),
    handle, xDesc, wDesc, convDesc,
    yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_FORWARD_ALGORITHM_EX,
    ,
    cudnnFindConvolutionForwardAlgorithmEx,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, xDesc, x, wDesc,
    w, convDesc, yDesc, y,
    requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
    workSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_IM_2_COL,
    ,
    cudnnIm2Col,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    void *colBuffer),
    handle, xDesc, x, wDesc,
    convDesc, colBuffer)

CUDNN_HOOK_GEN(
    CUDNN_REORDER_FILTER_AND_BIAS,
    ,
    cudnnReorderFilterAndBias,
    (cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType,
    const void *filterData,
    void *reorderedFilterData,
    int reorderBias,
    const void *biasData,
    void *reorderedBiasData),
    handle, filterDesc, reorderType, filterData,
    reorderedFilterData, reorderBias, biasData, reorderedBiasData)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE,
    ,
    cudnnGetConvolutionForwardWorkspaceSize,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes),
    handle, xDesc, wDesc, convDesc,
    yDesc, algo, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_CONVOLUTION_FORWARD,
    ,
    cudnnConvolutionForward,
    (cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, alpha, xDesc, x,
    wDesc, w, convDesc, algo,
    workSpace, workSpaceSizeInBytes, beta, yDesc,
    y)

CUDNN_HOOK_GEN(
    CUDNN_CONVOLUTION_BIAS_ACTIVATION_FORWARD,
    ,
    cudnnConvolutionBiasActivationForward,
    (cudnnHandle_t handle,
    const void *alpha1,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *alpha2,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    const cudnnTensorDescriptor_t biasDesc,
    const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y),
    handle, alpha1, xDesc, x,
    wDesc, w, convDesc, algo,
    workSpace, workSpaceSizeInBytes, alpha2, zDesc,
    z, biasDesc, bias, activationDesc,
    yDesc, y)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_MAX_COUNT,
    ,
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount,
    (cudnnHandle_t handle,
    int *count),
    handle, count)

CUDNN_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM,
    ,
    cudnnFindConvolutionBackwardDataAlgorithm,
    (cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults),
    handle, wDesc, dyDesc, convDesc,
    dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_DATA_ALGORITHM_EX,
    ,
    cudnnFindConvolutionBackwardDataAlgorithmEx,
    (cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, wDesc, w, dyDesc,
    dy, convDesc, dxDesc, dx,
    requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
    workSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_ALGORITHM_V7,
    ,
    cudnnGetConvolutionBackwardDataAlgorithm_v7,
    (cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults),
    handle, filterDesc, diffDesc, convDesc,
    gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE,
    ,
    cudnnGetConvolutionBackwardDataWorkspaceSize,
    (cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes),
    handle, wDesc, dyDesc, convDesc,
    dxDesc, algo, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_CONVOLUTION_BACKWARD_DATA,
    ,
    cudnnConvolutionBackwardData,
    (cudnnHandle_t handle,
    const void *alpha,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx),
    handle, alpha, wDesc, w,
    dyDesc, dy, convDesc, algo,
    workSpace, workSpaceSizeInBytes, beta, dxDesc,
    dx)

CUDNN_HOOK_GEN(
    CUDNN_GET_FOLDED_CONV_BACKWARD_DATA_DESCRIPTORS,
    ,
    cudnnGetFoldedConvBackwardDataDescriptors,
    (const cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc),
    handle, filterDesc, diffDesc, convDesc,
    gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc,
    foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc,
    gradFoldTransDesc, gradUnfoldTransDesc)

CUDNN_HOOK_GEN(
    CUDNN_CNN_INFER_VERSION_CHECK,
    ,
    cudnnCnnInferVersionCheck,
    (),
    )

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_MAX_COUNT,
    ,
    cudnnGetConvolutionBackwardFilterAlgorithmMaxCount,
    (cudnnHandle_t handle,
    int *count),
    handle, count)

CUDNN_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM,
    ,
    cudnnFindConvolutionBackwardFilterAlgorithm,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults),
    handle, xDesc, dyDesc, convDesc,
    dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HOOK_GEN(
    CUDNN_FIND_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_EX,
    ,
    cudnnFindConvolutionBackwardFilterAlgorithmEx,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes),
    handle, xDesc, x, dyDesc,
    y, convDesc, dwDesc, dw,
    requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
    workSpaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_ALGORITHM_V7,
    ,
    cudnnGetConvolutionBackwardFilterAlgorithm_v7,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults),
    handle, srcDesc, diffDesc, convDesc,
    gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults)

CUDNN_HOOK_GEN(
    CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE,
    ,
    cudnnGetConvolutionBackwardFilterWorkspaceSize,
    (cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes),
    handle, xDesc, dyDesc, convDesc,
    gradDesc, algo, sizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_CONVOLUTION_BACKWARD_FILTER,
    ,
    cudnnConvolutionBackwardFilter,
    (cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw),
    handle, alpha, xDesc, x,
    dyDesc, dy, convDesc, algo,
    workSpace, workSpaceSizeInBytes, beta, dwDesc,
    dw)

CUDNN_HOOK_GEN(
    CUDNN_CONVOLUTION_BACKWARD_BIAS,
    ,
    cudnnConvolutionBackwardBias,
    (cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const void *beta,
    const cudnnTensorDescriptor_t dbDesc,
    void *db),
    handle, alpha, dyDesc, dy,
    beta, dbDesc, db)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_FUSED_OPS_CONST_PARAM_PACK,
    ,
    cudnnCreateFusedOpsConstParamPack,
    (cudnnFusedOpsConstParamPack_t *constPack,
    cudnnFusedOps_t ops),
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
    (cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param),
    constPack, paramLabel, param)

CUDNN_HOOK_GEN(
    CUDNN_GET_FUSED_OPS_CONST_PARAM_PACK_ATTRIBUTE,
    ,
    cudnnGetFusedOpsConstParamPackAttribute,
    (const cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param,
    int *isNULL),
    constPack, paramLabel, param, isNULL)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_FUSED_OPS_VARIANT_PARAM_PACK,
    ,
    cudnnCreateFusedOpsVariantParamPack,
    (cudnnFusedOpsVariantParamPack_t *varPack,
    cudnnFusedOps_t ops),
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
    (cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr),
    varPack, paramLabel, ptr)

CUDNN_HOOK_GEN(
    CUDNN_GET_FUSED_OPS_VARIANT_PARAM_PACK_ATTRIBUTE,
    ,
    cudnnGetFusedOpsVariantParamPackAttribute,
    (const cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel,
    void *ptr),
    varPack, paramLabel, ptr)

CUDNN_HOOK_GEN(
    CUDNN_CREATE_FUSED_OPS_PLAN,
    ,
    cudnnCreateFusedOpsPlan,
    (cudnnFusedOpsPlan_t *plan,
    cudnnFusedOps_t ops),
    plan, ops)

CUDNN_HOOK_GEN(
    CUDNN_DESTROY_FUSED_OPS_PLAN,
    ,
    cudnnDestroyFusedOpsPlan,
    (cudnnFusedOpsPlan_t plan),
    plan)

CUDNN_HOOK_GEN(
    CUDNN_MAKE_FUSED_OPS_PLAN,
    ,
    cudnnMakeFusedOpsPlan,
    (cudnnHandle_t handle,
    cudnnFusedOpsPlan_t plan,
    const cudnnFusedOpsConstParamPack_t constPack,
    size_t *workspaceSizeInBytes),
    handle, plan, constPack, workspaceSizeInBytes)

CUDNN_HOOK_GEN(
    CUDNN_FUSED_OPS_EXECUTE,
    ,
    cudnnFusedOpsExecute,
    (cudnnHandle_t handle,
    const cudnnFusedOpsPlan_t plan,
    cudnnFusedOpsVariantParamPack_t varPack),
    handle, plan, varPack)

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
    (cudnnBackendDescriptorType_t descriptorType,
    cudnnBackendDescriptor_t *descriptor),
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
    (cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void *arrayOfElements),
    descriptor, attributeName, attributeType, elementCount,
    arrayOfElements)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_GET_ATTRIBUTE,
    ,
    cudnnBackendGetAttribute,
    (cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t *elementCount,
    void *arrayOfElements),
    descriptor, attributeName, attributeType, requestedElementCount,
    elementCount, arrayOfElements)

CUDNN_HOOK_GEN(
    CUDNN_BACKEND_EXECUTE,
    ,
    cudnnBackendExecute,
    (cudnnHandle_t handle,
    cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack),
    handle, executionPlan, variantPack)
/* hook function end */

#endif /* _CUDNN_HOOK_ENABLE */
