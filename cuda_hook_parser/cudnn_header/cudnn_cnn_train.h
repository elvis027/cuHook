/*
 * Copyright 2017-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/*
 *  cudnn_cnn_train : cuDNN's basic definitions and inference CNN functions.
 */

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#include "cudnn_version.h"
#include "cudnn_ops_infer.h"
#include "cudnn_ops_train.h"
#include "cudnn_cnn_infer.h"

/* These version numbers are autogenerated, do not edit manually. */
#define CUDNN_CNN_TRAIN_MAJOR 8
#define CUDNN_CNN_TRAIN_MINOR 4
#define CUDNN_CNN_TRAIN_PATCH 0

#if (CUDNN_CNN_TRAIN_MAJOR != CUDNN_MAJOR) || (CUDNN_CNN_TRAIN_MINOR != CUDNN_MINOR) || \
    (CUDNN_CNN_TRAIN_PATCH != CUDNN_PATCHLEVEL)
#error Version mismatch in cuDNN CNN INFER!!!
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* helper function to provide the convolution backward filter algo that fit best the requirement */

typedef struct cudnnConvolutionBwdFilterAlgoPerfStruct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionBwdFilterAlgoPerf_t;

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count);

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                            const cudnnTensorDescriptor_t xDesc,
                                            const cudnnTensorDescriptor_t dyDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnFilterDescriptor_t dwDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle,
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
                                              size_t workSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
                                              const cudnnTensorDescriptor_t srcDesc,
                                              const cudnnTensorDescriptor_t diffDesc,
                                              const cudnnConvolutionDescriptor_t convDesc,
                                              const cudnnFilterDescriptor_t gradDesc,
                                              const int requestedAlgoCount,
                                              int *returnedAlgoCount,
                                              cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
                                               const cudnnTensorDescriptor_t xDesc,
                                               const cudnnTensorDescriptor_t dyDesc,
                                               const cudnnConvolutionDescriptor_t convDesc,
                                               const cudnnFilterDescriptor_t gradDesc,
                                               cudnnConvolutionBwdFilterAlgo_t algo,
                                               size_t *sizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
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
                               void *dw);

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                             const void *alpha,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy,
                             const void *beta,
                             const cudnnTensorDescriptor_t dbDesc,
                             void *db);

cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack);

cudnnStatus_t CUDNNWINAPI
cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t constPack,
                                        cudnnFusedOpsConstParamLabel_t paramLabel,
                                        const void *param);

cudnnStatus_t CUDNNWINAPI
cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,
                                        cudnnFusedOpsConstParamLabel_t paramLabel,
                                        void *param,
                                        int *isNULL);

cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack);

cudnnStatus_t CUDNNWINAPI
cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          void *ptr);

cudnnStatus_t CUDNNWINAPI
cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          void *ptr);

cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan);

cudnnStatus_t CUDNNWINAPI
cudnnMakeFusedOpsPlan(cudnnHandle_t handle,
                      cudnnFusedOpsPlan_t plan,
                      const cudnnFusedOpsConstParamPack_t constPack,
                      size_t *workspaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack);

cudnnStatus_t CUDNNWINAPI
cudnnCnnTrainVersionCheck(void);

#if defined(__cplusplus)
}
#endif
