#ifndef _CUDNN_HOOK_FUNC_H_
#define _CUDNN_HOOK_FUNC_H_
#include <cudnn.h>

/**
 * This file is only used for input of parser and codegen.
 *
 */

#define CUDNN_OPS_INFER
#define CUDNN_OPS_TRAIN
#define CUDNN_ADV_INFER
#define CUDNN_ADV_TRAIN
#define CUDNN_CNN_INFER
#define CUDNN_CNN_TRAIN
#define CUDNN_BACKEND

#ifdef CUDNN_OPS_INFER

/* Special function: Ignored
size_t CUDNNWINAPI
cudnnGetVersion(void);
*/

/* Returns CUDA Runtime version statically linked against cudnn */
/* Special function: Ignored
size_t CUDNNWINAPI
cudnnGetCudartVersion(void);
*/

/* human-readable error messages */
/* Special function: Ignored
const char *CUDNNWINAPI
cudnnGetErrorString(cudnnStatus_t status);
*/

cudnnStatus_t CUDNNWINAPI
cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t CUDNNWINAPI
cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t CUDNNWINAPI
cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag);

cudnnStatus_t CUDNNWINAPI
cudnnGetProperty(libraryPropertyType type, int *value);

cudnnStatus_t CUDNNWINAPI
cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t CUDNNWINAPI
cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);

/* Create an instance of a generic Tensor descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnTensorFormat_t format,
                           cudnnDataType_t dataType, /* image data type */
                           int n,                    /* number of inputs (batch size) */
                           int c,                    /* number of input feature maps */
                           int h,                    /* height of input section */
                           int w);                   /* width of input section */

cudnnStatus_t CUDNNWINAPI
cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                             cudnnDataType_t dataType, /* image data type */
                             int n,                    /* number of inputs (batch size) */
                             int c,                    /* number of input feature maps */
                             int h,                    /* height of input section */
                             int w,                    /* width of input section */
                             int nStride,
                             int cStride,
                             int hStride,
                             int wStride);

cudnnStatus_t CUDNNWINAPI
cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t *dataType, /* image data type */
                           int *n,                    /* number of inputs (batch size) */
                           int *c,                    /* number of input feature maps  */
                           int *h,                    /* height of input section */
                           int *w,                    /* width of input section */
                           int *nStride,
                           int *cStride,
                           int *hStride,
                           int *wStride);

cudnnStatus_t CUDNNWINAPI
cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t dataType,
                           int nbDims,
                           const int dimA[],
                           const int strideA[]);

cudnnStatus_t CUDNNWINAPI
cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                             cudnnTensorFormat_t format,
                             cudnnDataType_t dataType,
                             int nbDims,
                             const int dimA[]);

cudnnStatus_t CUDNNWINAPI
cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           int nbDimsRequested,
                           cudnnDataType_t *dataType,
                           int *nbDims,
                           int dimA[],
                           int strideA[]);

cudnnStatus_t CUDNNWINAPI
cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size);

/* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
   input_stride :  c x h x h_stride
   feature_stride : h x h_stride
   h_stride  :  >= w  ( h_stride = w if no padding)
   w_stride  : 1


   2)Example of all images in row major with features maps interleaved
   input_stride :  c x h x h_stride
   feature_stride : 1
   h_stride  :  w x c
   w_stride  : c

   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
   input_stride :  c x w x w_stride
   feature_stride : w x w_stride
   h_stride  :  1
   w_stride  :  >= h

*/

/* Destroy an instance of Tensor4d descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);

/** Create a destination descriptor for cudnnTransformTensor */
cudnnStatus_t CUDNNWINAPI
cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                       const cudnnTensorDescriptor_t srcDesc,
                       cudnnTensorDescriptor_t destDesc,
                       size_t *destSizeInBytes);

/** Create an empty tensor transform descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc);

/** Initialize a previously created tensor transform descriptor. */
cudnnStatus_t CUDNNWINAPI
cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                  const uint32_t nbDims,
                                  const cudnnTensorFormat_t destFormat,
                                  const int32_t padBeforeA[],
                                  const int32_t padAfterA[],
                                  const uint32_t foldA[],
                                  const cudnnFoldingDirection_t direction);

/**
 * Retrieves the values stored in a previously initialized tensor transform
 * descriptor.
 */
cudnnStatus_t CUDNNWINAPI
cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                  uint32_t nbDimsRequested,
                                  cudnnTensorFormat_t *destFormat,
                                  int32_t padBeforeA[],
                                  int32_t padAfterA[],
                                  uint32_t foldA[],
                                  cudnnFoldingDirection_t *direction);

/**
 * Destroys a previously created tensor transform descriptor.
 */
cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc);

/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t CUDNNWINAPI
cudnnTransformTensor(cudnnHandle_t handle,
                     const void *alpha,
                     const cudnnTensorDescriptor_t xDesc,
                     const void *x,
                     const void *beta,
                     const cudnnTensorDescriptor_t yDesc,
                     void *y);

cudnnStatus_t CUDNNWINAPI
cudnnTransformTensorEx(cudnnHandle_t handle,
                       const cudnnTensorTransformDescriptor_t transDesc,
                       const void *alpha,
                       const cudnnTensorDescriptor_t srcDesc,
                       const void *srcData,
                       const void *beta,
                       const cudnnTensorDescriptor_t destDesc,
                       void *destData);

/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t CUDNNWINAPI
cudnnAddTensor(cudnnHandle_t handle,
               const void *alpha,
               const cudnnTensorDescriptor_t aDesc,
               const void *A,
               const void *beta,
               const cudnnTensorDescriptor_t cDesc,
               void *C);

cudnnStatus_t CUDNNWINAPI
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                           cudnnOpTensorOp_t opTensorOp,
                           cudnnDataType_t opTensorCompType,
                           cudnnNanPropagation_t opTensorNanOpt);

cudnnStatus_t CUDNNWINAPI
cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                           cudnnOpTensorOp_t *opTensorOp,
                           cudnnDataType_t *opTensorCompType,
                           cudnnNanPropagation_t *opTensorNanOpt);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc);

/* Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
/* B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT. */
cudnnStatus_t CUDNNWINAPI
cudnnOpTensor(cudnnHandle_t handle,
              const cudnnOpTensorDescriptor_t opTensorDesc,
              const void *alpha1,
              const cudnnTensorDescriptor_t aDesc,
              const void *A,
              const void *alpha2,
              const cudnnTensorDescriptor_t bDesc,
              const void *B,
              const void *beta,
              const cudnnTensorDescriptor_t cDesc,
              void *C);

cudnnStatus_t CUDNNWINAPI
cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t reduceTensorOp,
                               cudnnDataType_t reduceTensorCompType,
                               cudnnNanPropagation_t reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t reduceTensorIndices,
                               cudnnIndicesType_t reduceTensorIndicesType);

cudnnStatus_t CUDNNWINAPI
cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t *reduceTensorOp,
                               cudnnDataType_t *reduceTensorCompType,
                               cudnnNanPropagation_t *reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t *reduceTensorIndices,
                               cudnnIndicesType_t *reduceTensorIndicesType);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc);

/* Helper function to return the minimum size of the index space to be passed to the reduction given the input and
 * output tensors */
cudnnStatus_t CUDNNWINAPI
cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                             const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                             const cudnnTensorDescriptor_t aDesc,
                             const cudnnTensorDescriptor_t cDesc,
                             size_t *sizeInBytes);

/* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output
 * tensors */
cudnnStatus_t CUDNNWINAPI
cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                               const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               const cudnnTensorDescriptor_t aDesc,
                               const cudnnTensorDescriptor_t cDesc,
                               size_t *sizeInBytes);

/* Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
cudnnStatus_t CUDNNWINAPI
cudnnReduceTensor(cudnnHandle_t handle,
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
                  void *C);

/* Set all values of a tensor to a given value : y[i] = value[0] */
cudnnStatus_t CUDNNWINAPI
cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr);

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
cudnnStatus_t CUDNNWINAPI
cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha);

/* Create an instance of FilterStruct */
cudnnStatus_t CUDNNWINAPI
cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t dataType, /* image data type */
                           cudnnTensorFormat_t format,
                           int k,  /* number of output feature maps */
                           int c,  /* number of input feature maps */
                           int h,  /* height of each input filter */
                           int w); /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI
cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t *dataType, /* image data type */
                           cudnnTensorFormat_t *format,
                           int *k,  /* number of output feature maps */
                           int *c,  /* number of input feature maps */
                           int *h,  /* height of each input filter */
                           int *w); /* width of  each input filter */

cudnnStatus_t CUDNNWINAPI
cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t dataType, /* image data type */
                           cudnnTensorFormat_t format,
                           int nbDims,
                           const int filterDimA[]);

cudnnStatus_t CUDNNWINAPI
cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           int nbDimsRequested,
                           cudnnDataType_t *dataType, /* image data type */
                           cudnnTensorFormat_t *format,
                           int *nbDims,
                           int filterDimA[]);
cudnnStatus_t CUDNNWINAPI
cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size);

cudnnStatus_t CUDNNWINAPI
cudnnTransformFilter(cudnnHandle_t handle,
                     const cudnnTensorTransformDescriptor_t transDesc,
                     const void *alpha,
                     const cudnnFilterDescriptor_t srcDesc,
                     const void *srcData,
                     const void *beta,
                     const cudnnFilterDescriptor_t destDesc,
                     void *destData);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward softmax */
cudnnStatus_t CUDNNWINAPI
cudnnSoftmaxForward(cudnnHandle_t handle,
                    cudnnSoftmaxAlgorithm_t algo,
                    cudnnSoftmaxMode_t mode,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y);

/* Create an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                            cudnnPoolingMode_t mode,
                            cudnnNanPropagation_t maxpoolingNanOpt,
                            int windowHeight,
                            int windowWidth,
                            int verticalPadding,
                            int horizontalPadding,
                            int verticalStride,
                            int horizontalStride);

cudnnStatus_t CUDNNWINAPI
cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                            cudnnPoolingMode_t *mode,
                            cudnnNanPropagation_t *maxpoolingNanOpt,
                            int *windowHeight,
                            int *windowWidth,
                            int *verticalPadding,
                            int *horizontalPadding,
                            int *verticalStride,
                            int *horizontalStride);

cudnnStatus_t CUDNNWINAPI
cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                            const cudnnPoolingMode_t mode,
                            const cudnnNanPropagation_t maxpoolingNanOpt,
                            int nbDims,
                            const int windowDimA[],
                            const int paddingA[],
                            const int strideA[]);

cudnnStatus_t CUDNNWINAPI
cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                            int nbDimsRequested,
                            cudnnPoolingMode_t *mode,
                            cudnnNanPropagation_t *maxpoolingNanOpt,
                            int *nbDims,
                            int windowDimA[],
                            int paddingA[],
                            int strideA[]);

cudnnStatus_t CUDNNWINAPI
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int nbDims,
                                  int outputTensorDimA[]);

cudnnStatus_t CUDNNWINAPI
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int *n,
                                  int *c,
                                  int *h,
                                  int *w);

/* Destroy an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc);

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward pooling */
cudnnStatus_t CUDNNWINAPI
cudnnPoolingForward(cudnnHandle_t handle,
                    const cudnnPoolingDescriptor_t poolingDesc,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y);

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
cudnnStatus_t CUDNNWINAPI
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t mode,
                             cudnnNanPropagation_t reluNanOpt,
                             double coef); /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t *mode,
                             cudnnNanPropagation_t *reluNanOpt,
                             double *coef); /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t CUDNNWINAPI
cudnnSetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t activationDesc, double swish_beta);

cudnnStatus_t CUDNNWINAPI
cudnnGetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t activationDesc, double *swish_beta);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc);

/* Function to perform forward activation  */
cudnnStatus_t CUDNNWINAPI
cudnnActivationForward(cudnnHandle_t handle,
                       cudnnActivationDescriptor_t activationDesc,
                       const void *alpha,
                       const cudnnTensorDescriptor_t xDesc,
                       const void *x,
                       const void *beta,
                       const cudnnTensorDescriptor_t yDesc,
                       void *y);

/*
 * Create an instance of LRN (Local Response Normalization) descriptor
 * Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
 */
cudnnStatus_t CUDNNWINAPI
cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc);

/*
 * Uses a window [center-lookBehind, center+lookAhead], where
 * lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
 * Values of double parameters cast to tensor data type.
 */
cudnnStatus_t CUDNNWINAPI
cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK);
/*
 * Retrieve the settings currently stored in an LRN layer descriptor
 * Any of the provided pointers can be NULL (no corresponding value will be returned)
 */
cudnnStatus_t CUDNNWINAPI
cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha, double *lrnBeta, double *lrnK);

/* Destroy an instance of LRN descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc);

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI
cudnnLRNCrossChannelForward(cudnnHandle_t handle,
                            cudnnLRNDescriptor_t normDesc,
                            cudnnLRNMode_t lrnMode,
                            const void *alpha,
                            const cudnnTensorDescriptor_t xDesc,
                            const void *x,
                            const void *beta,
                            const cudnnTensorDescriptor_t yDesc,
                            void *y);

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
cudnnStatus_t CUDNNWINAPI
cudnnDivisiveNormalizationForward(cudnnHandle_t handle,
                                  cudnnLRNDescriptor_t normDesc,
                                  cudnnDivNormMode_t mode,
                                  const void *alpha,
                                  const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
                                  const void *x,
                                  const void *means, /* if NULL, means are assumed to be zero */
                                  void *temp,
                                  void *temp2,
                                  const void *beta,
                                  const cudnnTensorDescriptor_t yDesc,
                                  void *y);

/*
 * Derives a tensor descriptor from layer data descriptor for BatchNormalization
 * scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
 * bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
 */
cudnnStatus_t CUDNNWINAPI
cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                              const cudnnTensorDescriptor_t xDesc,
                              cudnnBatchNormMode_t mode);

/*
 * Performs Batch Normalization during Inference:
 * y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
 * with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
 * according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
 * above for notes on function arguments.
 */
cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,
                                        cudnnBatchNormMode_t mode,
                                        const void *alpha, /* alpha[0] = result blend factor */
                                        const void *beta,  /* beta[0] = dest layer blend factor */
                                        const cudnnTensorDescriptor_t xDesc,
                                        const void *x, /* NxCxHxW */
                                        const cudnnTensorDescriptor_t yDesc,
                                        void *y, /* NxCxHxW */
                                        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        const void *bnScale,
                                        const void *bnBias,
                                        const void *estimatedMean,
                                        const void *estimatedVariance,
                                        double epsilon);

/*
 * Derives a tensor descriptor from layer data descriptor for Normalization
 * scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
 * normScaleBiasMeanVarDesc and normScaleBiasDiffDesc in Normalization forward and backward functions.
 */
cudnnStatus_t CUDNNWINAPI
cudnnDeriveNormTensorDescriptor(cudnnTensorDescriptor_t derivedNormScaleBiasDesc,
                                cudnnTensorDescriptor_t derivedNormMeanVarDesc,
                                const cudnnTensorDescriptor_t xDesc,
                                cudnnNormMode_t mode,
                                int groupCnt); /* Place hold for future work, should be set to 1 now*/

/*
 * Performs Normalization during Inference:
 * y[i] = normScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + normBias[k]
 * with normScale, normBias, runningMean, runningInvVariance tensors indexed
 * according to per-channel or per-activation mode. Refer to cudnnNormalizationForwardTraining
 * above for notes on function arguments.
 */
cudnnStatus_t CUDNNWINAPI
cudnnNormalizationForwardInference(cudnnHandle_t handle,
                                   cudnnNormMode_t mode,
                                   cudnnNormOps_t normOps,
                                   cudnnNormAlgo_t algo,
                                   const void *alpha, /* alpha[0] = result blend factor */
                                   const void *beta,  /* beta[0] = dest layer blend factor */
                                   const cudnnTensorDescriptor_t xDesc,
                                   const void *x, /* NxCxHxW */
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
                                   void *y, /* NxCxHxW */
                                   double epsilon,
                                   int groupCnt); /* Place hold for future work*/

cudnnStatus_t CUDNNWINAPI
cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                       cudnnSamplerType_t samplerType,
                                       cudnnDataType_t dataType,
                                       const int nbDims,
                                       const int dimA[]);

cudnnStatus_t CUDNNWINAPI
cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
                                   const cudnnSpatialTransformerDescriptor_t stDesc,
                                   const void *theta,
                                   void *grid);

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfSamplerForward(cudnnHandle_t handle,
                             cudnnSpatialTransformerDescriptor_t stDesc,
                             const void *alpha,
                             const cudnnTensorDescriptor_t xDesc,
                             const void *x,
                             const void *grid,
                             const void *beta,
                             cudnnTensorDescriptor_t yDesc,
                             void *y);

cudnnStatus_t CUDNNWINAPI
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc);

/*helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
cudnnStatus_t CUDNNWINAPI
cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes);

/*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
cudnnStatus_t CUDNNWINAPI
cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                          cudnnHandle_t handle,
                          float dropout,
                          void *states,
                          size_t stateSizeInBytes,
                          unsigned long long seed);

/* Restores the dropout descriptor to a previously saved-off state */
cudnnStatus_t CUDNNWINAPI
cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                              cudnnHandle_t handle,
                              float dropout,
                              void *states,
                              size_t stateSizeInBytes,
                              unsigned long long seed);

cudnnStatus_t CUDNNWINAPI
cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                          cudnnHandle_t handle,
                          float *dropout,
                          void **states,
                          unsigned long long *seed);

cudnnStatus_t CUDNNWINAPI
cudnnDropoutForward(cudnnHandle_t handle,
                    const cudnnDropoutDescriptor_t dropoutDesc,
                    const cudnnTensorDescriptor_t xdesc,
                    const void *x,
                    const cudnnTensorDescriptor_t ydesc,
                    void *y,
                    void *reserveSpace,
                    size_t reserveSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                             cudnnAlgorithmDescriptor_t algoDesc,
                             cudnnStatus_t status,
                             float time,
                             size_t memory);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                             cudnnAlgorithmDescriptor_t *algoDesc,
                             cudnnStatus_t *status,
                             float *time,
                             size_t *memory);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSaveAlgorithm(cudnnHandle_t handle,
                   cudnnAlgorithmDescriptor_t algoDesc,
                   void *algoSpace,
                   size_t algoSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRestoreAlgorithm(cudnnHandle_t handle,
                      void *algoSpace,
                      size_t algoSpaceSizeInBytes,
                      cudnnAlgorithmDescriptor_t algoDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr);

cudnnStatus_t CUDNNWINAPI
cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr);

/*
 * \brief Cross-library version checker.
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * \returns CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 */
cudnnStatus_t CUDNNWINAPI
cudnnOpsInferVersionCheck(void);

#endif /* CUDNN_OPS_INFER */

#ifdef CUDNN_OPS_TRAIN

/* Function to perform backward softmax */
cudnnStatus_t CUDNNWINAPI
cudnnSoftmaxBackward(cudnnHandle_t handle,
                     cudnnSoftmaxAlgorithm_t algo,
                     cudnnSoftmaxMode_t mode,
                     const void *alpha,
                     const cudnnTensorDescriptor_t yDesc,
                     const void *y,
                     const cudnnTensorDescriptor_t dyDesc,
                     const void *dy,
                     const void *beta,
                     const cudnnTensorDescriptor_t dxDesc,
                     void *dx);

/* Function to perform backward pooling */
cudnnStatus_t CUDNNWINAPI
cudnnPoolingBackward(cudnnHandle_t handle,
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
                     void *dx);

/* Function to perform backward activation  */
cudnnStatus_t CUDNNWINAPI
cudnnActivationBackward(cudnnHandle_t handle,
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
                        void *dx);

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI
cudnnLRNCrossChannelBackward(cudnnHandle_t handle,
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
                             void *dx);

cudnnStatus_t CUDNNWINAPI
cudnnDivisiveNormalizationBackward(cudnnHandle_t handle,
                                   cudnnLRNDescriptor_t normDesc,
                                   cudnnDivNormMode_t mode,
                                   const void *alpha,
                                   const cudnnTensorDescriptor_t xDesc, /* same desc for x, means, dy, temp, temp2 */
                                   const void *x,
                                   const void *means, /* if NULL, means are assumed to be zero */
                                   const void *dy,
                                   void *temp,
                                   void *temp2,
                                   const void *beta,
                                   const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
                                   void *dx,                                   /* output x differential */
                                   void *dMeans); /* output means differential, can be NULL */

cudnnStatus_t CUDNNWINAPI
cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t handle,
                                                         cudnnBatchNormMode_t mode,
                                                         cudnnBatchNormOps_t bnOps,
                                                         const cudnnTensorDescriptor_t xDesc,
                                                         const cudnnTensorDescriptor_t zDesc,
                                                         const cudnnTensorDescriptor_t yDesc,
                                                         const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                                         const cudnnActivationDescriptor_t activationDesc,
                                                         size_t *sizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t handle,
                                                  cudnnBatchNormMode_t mode,
                                                  cudnnBatchNormOps_t bnOps,
                                                  const cudnnTensorDescriptor_t xDesc,
                                                  const cudnnTensorDescriptor_t yDesc,
                                                  const cudnnTensorDescriptor_t dyDesc,
                                                  const cudnnTensorDescriptor_t dzDesc,
                                                  const cudnnTensorDescriptor_t dxDesc,
                                                  const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                                  const cudnnActivationDescriptor_t activationDesc,
                                                  size_t *sizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t handle,
                                                     cudnnBatchNormMode_t mode,
                                                     cudnnBatchNormOps_t bnOps,
                                                     const cudnnActivationDescriptor_t activationDesc,
                                                     const cudnnTensorDescriptor_t xDesc,
                                                     size_t *sizeInBytes);

/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc,
    const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc,
    void *y, /* NxCxHxW */

    /* Shared desc for the next 6 tensors in the argument list.
       Data type to be set as follows:
       type = (typeOf(x) == double) ? double : float
       Dimensions for this descriptor depend on normalization mode
       - Spatial Normalization : tensors are expected to have dims 1xCx1x1
        (normalization is performed across NxHxW)
       - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
        (normalization is performed across N) */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

    /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
    const void *bnScale,
    const void *bnBias,

    /* MUST use factor=1 in the very first call of a complete training cycle.
       Use a factor=1/(1+n) at N-th call to the function to get
       Cumulative Moving Average (CMA) behavior
       CMA[n] = (x[1]+...+x[n])/n
       Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
       ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
       CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
    double exponentialAverageFactor,

    /* Used in Training phase only.
       runningMean = newMean*factor + runningMean*(1-factor) */
    void *resultRunningMean,
    /* Output in training mode, input in inference. Is the moving average
       of  variance[x] (factor is applied in the same way as for runningMean) */
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean,
    void *resultSaveInvVariance);

/* Computes y = relu(BN(x) + z). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

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

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean,
    void *resultSaveInvVariance,

    cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes);

/* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationBackward(cudnnHandle_t handle,
                                cudnnBatchNormMode_t mode,
                                const void *alphaDataDiff,
                                const void *betaDataDiff,
                                const void *alphaParamDiff,
                                const void *betaParamDiff,
                                const cudnnTensorDescriptor_t xDesc, /* same desc for x, dx, dy */
                                const void *x,
                                const cudnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const cudnnTensorDescriptor_t dxDesc,
                                void *dx,
                                /* Shared tensor desc for the 4 tensors below */
                                const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                const void *bnScale, /* bnBias doesn't affect backpropagation */
                                /* scale and bias diff are not backpropagated below this layer */
                                void *dBnScaleResult,
                                void *dBnBiasResult,
                                /* Same epsilon as forward pass */
                                double epsilon,

                                /* Optionally cached intermediate results from
                                   forward pass */
                                const void *savedMean,
                                const void *savedInvVariance);

cudnnStatus_t CUDNNWINAPI
cudnnBatchNormalizationBackwardEx(cudnnHandle_t handle,
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

                                  /* Shared tensor desc for the 4 tensors below */
                                  const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                  const void *bnScaleData,
                                  const void *bnBiasData, /* needed if there is activation */
                                  void *dBnScaleData,
                                  void *dBnBiasData,
                                  double epsilon, /* Same epsilon as forward pass */

                                  /* Optionally cached intermediate results from
                                     forward pass */
                                  const void *savedMean,
                                  const void *savedInvVariance,
                                  cudnnActivationDescriptor_t activationDesc,
                                  void *workSpace,
                                  size_t workSpaceSizeInBytes,
                                  void *reserveSpace,
                                  size_t reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnGetNormalizationForwardTrainingWorkspaceSize(cudnnHandle_t handle,
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
                                                  int groupCnt); /* Place hold for future work, should be set to 1 now*/

cudnnStatus_t CUDNNWINAPI
cudnnGetNormalizationBackwardWorkspaceSize(cudnnHandle_t handle,
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
                                           int groupCnt); /* Place hold for future work, should be set to 1 now*/

cudnnStatus_t CUDNNWINAPI
cudnnGetNormalizationTrainingReserveSpaceSize(cudnnHandle_t handle,
                                              cudnnNormMode_t mode,
                                              cudnnNormOps_t normOps,
                                              cudnnNormAlgo_t algo,
                                              const cudnnActivationDescriptor_t activationDesc,
                                              const cudnnTensorDescriptor_t xDesc,
                                              size_t *sizeInBytes,
                                              int groupCnt); /* Place hold for future work, should be set to 1 now*/

/* Computes y = relu(Norm(x) + z). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI
cudnnNormalizationForwardTraining(cudnnHandle_t handle,
                                  cudnnNormMode_t mode,
                                  cudnnNormOps_t normOps,
                                  cudnnNormAlgo_t algo,
                                  const void *alpha, /* alpha[0] = result blend factor */
                                  const void *beta,  /* beta[0] = dest layer blend factor */
                                  const cudnnTensorDescriptor_t xDesc,
                                  const void *xData,
                                  const cudnnTensorDescriptor_t normScaleBiasDesc,
                                  const void *normScale,
                                  const void *normBias,
                                  double exponentialAverageFactor,
                                  const cudnnTensorDescriptor_t normMeanVarDesc,
                                  void *resultRunningMean,
                                  void *resultRunningVariance,
                                  /* Has to be >= 0. Should be the same in forward and backward functions. */
                                  double epsilon,
                                  /* Optionally save intermediate results from the forward pass here
                                     - can be reused to speed up backward pass. NULL if unused */
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
                                  int groupCnt); /* Place hold for future work, should be set to 1 now*/

cudnnStatus_t CUDNNWINAPI
cudnnNormalizationBackward(cudnnHandle_t handle,
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
                           /* Shared tensor desc for the 4 tensors below */
                           const cudnnTensorDescriptor_t dNormScaleBiasDesc,
                           const void *normScaleData,
                           const void *normBiasData, /* needed if there is activation */
                           void *dNormScaleData,
                           void *dNormBiasData,
                           double epsilon, /* Same epsilon as forward pass */
                           const cudnnTensorDescriptor_t normMeanVarDesc,
                           /* Optionally cached intermediate results from
                              forward pass */
                           const void *savedMean,
                           const void *savedInvVariance,
                           cudnnActivationDescriptor_t activationDesc,
                           void *workSpace,
                           size_t workSpaceSizeInBytes,
                           void *reserveSpace,
                           size_t reserveSpaceSizeInBytes,
                           int groupCnt); /* Place hold for future work, should be set to 1 now*/

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                    const cudnnSpatialTransformerDescriptor_t stDesc,
                                    const void *dgrid,
                                    void *dtheta);

cudnnStatus_t CUDNNWINAPI
cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
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
                              void *dgrid);

cudnnStatus_t CUDNNWINAPI
cudnnDropoutBackward(cudnnHandle_t handle,
                     const cudnnDropoutDescriptor_t dropoutDesc,
                     const cudnnTensorDescriptor_t dydesc,
                     const void *dy,
                     const cudnnTensorDescriptor_t dxdesc,
                     void *dx,
                     void *reserveSpace,
                     size_t reserveSpaceSizeInBytes);

/*
 * \brief Cross-library version checker.
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * \returns CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 */
cudnnStatus_t CUDNNWINAPI
cudnnOpsTrainVersionCheck(void);

#endif /* CUDNN_OPS_TRAIN */

#ifdef CUDNN_ADV_INFER

/* BASIC RNN API */

cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc,
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
                         uint32_t auxFlags);

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc,
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
                         uint32_t *auxFlags);

/*
 * mathPrec in cudnnSetRNNDescriptor_v6() specifies compute precision
 * compute precision is further modified by cudnnSetRNNMatrixMathType()
 * dataType in cudnnGetRNNParamsSize() and wDesc specify weight storage
 * dropout is between RNN layers, not between recurrent steps
 */
CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
                         cudnnRNNDescriptor_t rnnDesc,
                         const int hiddenSize,
                         const int numLayers,
                         cudnnDropoutDescriptor_t dropoutDesc,
                         cudnnRNNInputMode_t inputMode,
                         cudnnDirectionMode_t direction,
                         cudnnRNNMode_t cellMode,
                         cudnnRNNAlgo_t algo,
                         cudnnDataType_t mathPrec);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNDescriptor_v6(cudnnHandle_t handle,
                         cudnnRNNDescriptor_t rnnDesc,
                         int *hiddenSize,
                         int *numLayers,
                         cudnnDropoutDescriptor_t *dropoutDesc,
                         cudnnRNNInputMode_t *inputMode,
                         cudnnDirectionMode_t *direction,
                         cudnnRNNMode_t *cellMode,
                         cudnnRNNAlgo_t *algo,
                         cudnnDataType_t *mathPrec);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode);

cudnnStatus_t CUDNNWINAPI
cudnnRNNSetClip_v8(cudnnRNNDescriptor_t rnnDesc,
                   cudnnRNNClipMode_t clipMode,
                   cudnnNanPropagation_t clipNanOpt,
                   double lclip,
                   double rclip);

cudnnStatus_t CUDNNWINAPI
cudnnRNNGetClip_v8(cudnnRNNDescriptor_t rnnDesc,
                   cudnnRNNClipMode_t *clipMode,
                   cudnnNanPropagation_t *clipNanOpt,
                   double *lclip,
                   double *rclip);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNSetClip(cudnnHandle_t handle,
                cudnnRNNDescriptor_t rnnDesc,
                cudnnRNNClipMode_t clipMode,
                cudnnNanPropagation_t clipNanOpt,
                double lclip,
                double rclip);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNGetClip(cudnnHandle_t handle,
                cudnnRNNDescriptor_t rnnDesc,
                cudnnRNNClipMode_t *clipMode,
                cudnnNanPropagation_t *clipNanOpt,
                double *lclip,
                double *rclip);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                            cudnnRNNDescriptor_t rnnDesc,
                            const int recProjSize,
                            const int outProjSize);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                            const cudnnRNNDescriptor_t rnnDesc,
                            int *recProjSize,
                            int *outProjSize);

/* Expensive. Creates the plan for the specific settings. */
CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                             const int minibatch,
                             const cudnnDataType_t dataType,
                             cudnnPersistentRNNPlan_t *plan);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan);

cudnnStatus_t CUDNNWINAPI
cudnnBuildRNNDynamic(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int miniBatch);

/* dataType in weight descriptors and input descriptors is used to describe storage */
CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                         const cudnnRNNDescriptor_t rnnDesc,
                         const int seqLength,
                         const cudnnTensorDescriptor_t *xDesc,
                         size_t *sizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
                               const cudnnRNNDescriptor_t rnnDesc,
                               const int seqLength,
                               const cudnnTensorDescriptor_t *xDesc,
                               size_t *sizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNTempSpaceSizes(cudnnHandle_t handle,
                          cudnnRNNDescriptor_t rnnDesc,
                          cudnnForwardMode_t fMode,
                          cudnnRNNDataDescriptor_t xDesc,
                          size_t *workSpaceSize,
                          size_t *reserveSpaceSize);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNParamsSize(cudnnHandle_t handle,
                      const cudnnRNNDescriptor_t rnnDesc,
                      const cudnnTensorDescriptor_t xDesc,
                      size_t *sizeInBytes,
                      cudnnDataType_t dataType);

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNWeightSpaceSize(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, size_t *weightSpaceSize);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
                                const cudnnRNNDescriptor_t rnnDesc,
                                const int pseudoLayer,
                                const cudnnTensorDescriptor_t xDesc,
                                const cudnnFilterDescriptor_t wDesc,
                                const void *w,
                                const int linLayerID,
                                cudnnFilterDescriptor_t linLayerMatDesc,
                                void **linLayerMat);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
                              const cudnnRNNDescriptor_t rnnDesc,
                              const int pseudoLayer,
                              const cudnnTensorDescriptor_t xDesc,
                              const cudnnFilterDescriptor_t wDesc,
                              const void *w,
                              const int linLayerID,
                              cudnnFilterDescriptor_t linLayerBiasDesc,
                              void **linLayerBias);

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNWeightParams(cudnnHandle_t handle,
                        cudnnRNNDescriptor_t rnnDesc,
                        int32_t pseudoLayer,
                        size_t weightSpaceSize,
                        const void *weightSpace,
                        int32_t linLayerID,
                        cudnnTensorDescriptor_t mDesc,
                        void **mAddr,
                        cudnnTensorDescriptor_t bDesc,
                        void **bAddr);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardInference(cudnnHandle_t handle,
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
                         size_t workSpaceSizeInBytes);

/* RNN EX API */

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, unsigned paddingMode);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, unsigned *paddingMode);

cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          cudnnDataType_t dataType,
                          cudnnRNNDataLayout_t layout,
                          int maxSeqLength,
                          int batchSize,
                          int vectorSize,
                          const int seqLengthArray[], /* length of each sequence in the batch */
                          void *paddingFill);         /* symbol for filling padding position in output */

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          cudnnDataType_t *dataType,
                          cudnnRNNDataLayout_t *layout,
                          int *maxSeqLength,
                          int *batchSize,
                          int *vectorSize,
                          int arrayLengthRequested,
                          int seqLengthArray[],
                          void *paddingFill);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardInferenceEx(cudnnHandle_t handle,
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
                           const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
                           const void *keys,                     /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
                           void *cAttn,                          /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
                           void *iAttn,                          /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
                           void *queries,                        /* reserved, should pass NULL */
                           void *workSpace,
                           size_t workSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnRNNForward(cudnnHandle_t handle,
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
                void *reserveSpace);

/* RNN FIND API */

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,
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
                                        size_t workSpaceSizeInBytes);

/* Sequence data descriptor */

cudnnStatus_t CUDNNWINAPI
cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc);

cudnnStatus_t CUDNNWINAPI
cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t dataType,
                          int nbDims,
                          const int dimA[],
                          const cudnnSeqDataAxis_t axes[],
                          size_t seqLengthArraySize,
                          const int seqLengthArray[],
                          void *paddingFill);

cudnnStatus_t CUDNNWINAPI
cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t *dataType,
                          int *nbDims,
                          int nbDimsRequested,
                          int dimA[],
                          cudnnSeqDataAxis_t axes[],
                          size_t *seqLengthArraySize,
                          size_t seqLengthSizeRequested,
                          int seqLengthArray[],
                          void *paddingFill);

/* Multihead Attention */

cudnnStatus_t CUDNNWINAPI
cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
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
                       int maxBeamSize);

cudnnStatus_t CUDNNWINAPI
cudnnGetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
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
                       int *maxBeamSize);

cudnnStatus_t CUDNNWINAPI
cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             size_t *weightSizeInBytes,
                             size_t *workSpaceSizeInBytes,
                             size_t *reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             cudnnMultiHeadAttnWeightKind_t wKind,
                             size_t weightSizeInBytes,
                             const void *weights,
                             cudnnTensorDescriptor_t wDesc,
                             void **wAddr);

cudnnStatus_t CUDNNWINAPI
cudnnMultiHeadAttnForward(cudnnHandle_t handle,
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
                          void *reserveSpace);

/*
 * \brief Cross-library version checker.
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * \returns CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 */
cudnnStatus_t CUDNNWINAPI
cudnnAdvInferVersionCheck(void);

#endif /* CUDNN_ADV_INFER */

#ifdef CUDNN_ADV_TRAIN

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardTraining(cudnnHandle_t handle,
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
                        size_t reserveSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardData(cudnnHandle_t handle,
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
                     size_t reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardData_v8(cudnnHandle_t handle,
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
                        void *reserveSpace);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardWeights(cudnnHandle_t handle,
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
                        size_t reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardWeights_v8(cudnnHandle_t handle,
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
                           void *reserveSpace);

/* RNN EX API */

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNForwardTrainingEx(cudnnHandle_t handle,
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
                          const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
                          const void *keys,                     /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
                          void *cAttn,                          /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
                          void *iAttn,                          /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
                          void *queries,                        /* reserved, should pass NULL */
                          void *workSpace,
                          size_t workSpaceSizeInBytes,
                          void *reserveSpace,
                          size_t reserveSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardDataEx(cudnnHandle_t handle,
                       const cudnnRNNDescriptor_t rnnDesc,
                       const cudnnRNNDataDescriptor_t yDesc,
                       const void *y,
                       const cudnnRNNDataDescriptor_t dyDesc,
                       const void *dy,
                       const cudnnRNNDataDescriptor_t dcDesc, /* reserved, should pass NULL */
                       const void *dcAttn,                    /* reserved, should pass NULL */
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
                       const cudnnRNNDataDescriptor_t dkDesc, /* reserved, should pass NULL */
                       void *dkeys,                           /* reserved, should pass NULL */
                       void *workSpace,
                       size_t workSpaceSizeInBytes,
                       void *reserveSpace,
                       size_t reserveSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardWeightsEx(cudnnHandle_t handle,
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
                          size_t reserveSpaceSizeInBytes);

/* RNN FIND API */

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,
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
                                       size_t reserveSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,
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
                                    size_t reserveSpaceSizeInBytes);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,
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
                                       size_t reserveSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnMultiHeadAttnBackwardData(cudnnHandle_t handle,
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
                               void *reserveSpace);

cudnnStatus_t CUDNNWINAPI
cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t handle,
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
                                  void *reserveSpace);

/*
* CTC (Connectionist Temporal Classification) loss descriptor create/destory/set/get functions
*/

cudnnStatus_t CUDNNWINAPI
cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType);

cudnnStatus_t CUDNNWINAPI
cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                            cudnnDataType_t compType,
                            cudnnLossNormalizationMode_t normMode,
                            cudnnNanPropagation_t gradMode);

cudnnStatus_t CUDNNWINAPI
cudnnSetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t ctcLossDesc,
                             cudnnDataType_t compType,
                             cudnnLossNormalizationMode_t normMode,
                             cudnnNanPropagation_t gradMode,
                             int maxLabelLength);

cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType);

cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                            cudnnDataType_t *compType,
                            cudnnLossNormalizationMode_t *normMode,
                            cudnnNanPropagation_t *gradMode);

cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t ctcLossDesc,
                             cudnnDataType_t *compType,
                             cudnnLossNormalizationMode_t *normMode,
                             cudnnNanPropagation_t *gradMode,
                             int *maxLabelLength);

cudnnStatus_t CUDNNWINAPI
cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc);

/* return the ctc costs and gradients, given the probabilities and labels */
cudnnStatus_t CUDNNWINAPI
cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the
                          mini batch size, A is the alphabet size)  */
    const void *probs, /* probabilities after softmax, in GPU memory */
    const int hostLabels[],                      /* labels, in CPU memory */
    const int hostLabelLengths[],                /* the length of each label, in CPU memory */
    const int hostInputLengths[],                /* the lengths of timing steps in each batch, in CPU memory */
    void *costs,                                 /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
    void *gradients,         /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace,              /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes); /* size of the workspace */

/* return the ctc costs and gradients, given the probabilities and labels */
cudnnStatus_t CUDNNWINAPI
cudnnCTCLoss_v8(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t
        probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the
                          mini batch size, A is the alphabet size)  */
    const void *probs, /* probabilities after softmax, in GPU memory */
    const int labels[],                          /* labels, in GPU memory */
    const int labelLengths[],                    /* the length of each label, in GPU memory */
    const int inputLengths[],                    /* the lengths of timing steps in each batch, in GPU memory */
    void *costs,                                 /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
    void *gradients,             /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
    size_t workSpaceSizeInBytes, /* size of the workspace */
    void *workspace);            /* pointer to the workspace, in GPU memory */

/* return the workspace size needed for ctc */
cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size) */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the
                                                    dimensions are T,N,A. To compute costs
                                                    only, set it to NULL */
    const int *labels,                           /* labels, in CPU memory */
    const int *labelLengths,                     /* the length of each label, in CPU memory */
    const int *inputLengths,                     /* the lengths of timing steps in each batch, in CPU memory */
    cudnnCTCLossAlgo_t algo,                     /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes); /* pointer to the returned workspace size */

/* return the workspace size needed for ctc */
cudnnStatus_t CUDNNWINAPI
cudnnGetCTCLossWorkspaceSize_v8(
    cudnnHandle_t handle,
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size) */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the
                                                    dimensions are T,N,A. To compute costs
                                                    only, set it to NULL */
    size_t *sizeInBytes);                        /* pointer to the returned workspace size */

/*
 * \brief Cross-library version checker.
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * \returns CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_VERSION_MISMATCH if the versions are inconsistent.
 */
cudnnStatus_t CUDNNWINAPI
cudnnAdvTrainVersionCheck(void);

#endif /* CUDNN_ADV_TRAIN */

#ifdef CUDNN_CNN_INFER

/* Create an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc);

/* Destroy an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType);

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType);

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount);

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount);

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType);

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType);

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                int pad_h,      /* zero-padding height */
                                int pad_w,      /* zero-padding width */
                                int u,          /* vertical filter stride */
                                int v,          /* horizontal filter stride */
                                int dilation_h, /* filter dilation in the vertical dimension */
                                int dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t mode,
                                cudnnDataType_t computeType);

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                int *pad_h,      /* zero-padding height */
                                int *pad_w,      /* zero-padding width */
                                int *u,          /* vertical filter stride */
                                int *v,          /* horizontal filter stride */
                                int *dilation_h, /* filter dilation in the vertical dimension */
                                int *dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t *mode,
                                cudnnDataType_t *computeType);

cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                int arrayLength, /* nbDims-2 size */
                                const int padA[],
                                const int filterStrideA[],
                                const int dilationA[],
                                cudnnConvolutionMode_t mode,
                                cudnnDataType_t computeType); /* convolution data type */

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                int arrayLengthRequested,
                                int *arrayLength,
                                int padA[],
                                int strideA[],
                                int dilationA[],
                                cudnnConvolutionMode_t *mode,
                                cudnnDataType_t *computeType); /* convolution data type */

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      const cudnnFilterDescriptor_t filterDesc,
                                      int *n,
                                      int *c,
                                      int *h,
                                      int *w);

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      const cudnnFilterDescriptor_t filterDesc,
                                      int nbDims,
                                      int tensorOuputDimA[]);

/* helper function to provide the convolution forward algo that fit best the requirement */
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count);

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                       const cudnnTensorDescriptor_t srcDesc,
                                       const cudnnFilterDescriptor_t filterDesc,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       const cudnnTensorDescriptor_t destDesc,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnConvolutionFwdAlgoPerf_t *perfResults);

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                     const cudnnTensorDescriptor_t xDesc,
                                     const cudnnFilterDescriptor_t wDesc,
                                     const cudnnConvolutionDescriptor_t convDesc,
                                     const cudnnTensorDescriptor_t yDesc,
                                     const int requestedAlgoCount,
                                     int *returnedAlgoCount,
                                     cudnnConvolutionFwdAlgoPerf_t *perfResults);

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,
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
                                       size_t workSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnIm2Col(cudnnHandle_t handle,
            const cudnnTensorDescriptor_t xDesc,
            const void *x,
            const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc,
            void *colBuffer);

cudnnStatus_t CUDNNWINAPI
cudnnReorderFilterAndBias(cudnnHandle_t handle,
                          const cudnnFilterDescriptor_t filterDesc,
                          cudnnReorderType_t reorderType,
                          const void *filterData,
                          void *reorderedFilterData,
                          int reorderBias,
                          const void *biasData,
                          void *reorderedBiasData);

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                        const cudnnTensorDescriptor_t xDesc,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const cudnnConvolutionDescriptor_t convDesc,
                                        const cudnnTensorDescriptor_t yDesc,
                                        cudnnConvolutionFwdAlgo_t algo,
                                        size_t *sizeInBytes);

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform the forward pass for batch convolution */
cudnnStatus_t CUDNNWINAPI
cudnnConvolutionForward(cudnnHandle_t handle,
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
                        void *y);

/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBiasActivationForward(cudnnHandle_t handle,
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
                                      void *y);

/* helper function to provide the convolution backward data algo that fit best the requirement */
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count);

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                          const cudnnFilterDescriptor_t wDesc,
                                          const cudnnTensorDescriptor_t dyDesc,
                                          const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t dxDesc,
                                          const int requestedAlgoCount,
                                          int *returnedAlgoCount,
                                          cudnnConvolutionBwdDataAlgoPerf_t *perfResults);

cudnnStatus_t CUDNNWINAPI
cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle,
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
                                            size_t workSpaceSizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
                                            const cudnnFilterDescriptor_t filterDesc,
                                            const cudnnTensorDescriptor_t diffDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t gradDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdDataAlgoPerf_t *perfResults);

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
                                             const cudnnFilterDescriptor_t wDesc,
                                             const cudnnTensorDescriptor_t dyDesc,
                                             const cudnnConvolutionDescriptor_t convDesc,
                                             const cudnnTensorDescriptor_t dxDesc,
                                             cudnnConvolutionBwdDataAlgo_t algo,
                                             size_t *sizeInBytes);

cudnnStatus_t CUDNNWINAPI
cudnnConvolutionBackwardData(cudnnHandle_t handle,
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
                             void *dx);

/* Helper function to calculate folding descriptors for dgrad */
cudnnStatus_t CUDNNWINAPI
cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t handle,
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
                                          cudnnTensorTransformDescriptor_t gradUnfoldTransDesc);

cudnnStatus_t CUDNNWINAPI
cudnnCnnInferVersionCheck(void);

#endif /* CUDNN_CNN_INFER */

#ifdef CUDNN_CNN_TRAIN

/* helper function to provide the convolution backward filter algo that fit best the requirement */
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

#endif /* CUDNN_CNN_TRAIN */

#ifdef CUDNN_BACKEND

cudnnStatus_t CUDNNWINAPI
cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t elementCount,
                         const void *arrayOfElements);

cudnnStatus_t CUDNNWINAPI
cudnnBackendGetAttribute(cudnnBackendDescriptor_t const descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount,
                         int64_t *elementCount,
                         void *arrayOfElements);

cudnnStatus_t CUDNNWINAPI
cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack);

#endif /* CUDNN_BACKEND */

#endif /* _CUDNN_HOOK_FUNC_H_ */
