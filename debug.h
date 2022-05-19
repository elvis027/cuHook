#ifndef _DEBUG_H_
#define _DEBUG_H_

#ifdef _DEBUG_ENABLE
#define DEBUG(...)                              \
    do {                                        \
        fprintf(stderr, "DEBUG: " __VA_ARGS__); \
        fflush(stderr);                         \
    } while (0)
#else
#define DEBUG(...)
#endif /* _DEBUG_ENABLE */

#ifdef _INFO_ENABLE
#define INFO(stream, ...)                       \
    do {                                        \
        fprintf(stream, "INFO: " __VA_ARGS__);  \
        fflush(stream);                         \
    } while (0)
#else
#define INFO(stream, ...)
#endif /* _INFO_ENABLE */

#define WARN(...)                               \
    do {                                        \
        fprintf(stderr, "WARN: " __VA_ARGS__);  \
        fflush(stderr);                         \
    } while (0)

#define ERROR(...)                              \
    do {                                        \
        fprintf(stderr, "ERROR: " __VA_ARGS__); \
        fflush(stderr);                         \
    } while (0)

#define DRIVER_API_CALL(call)                                       \
    do {                                                            \
        CUresult _status = call;                                    \
        if (_status != CUDA_SUCCESS) {                              \
            const char *_errstr;                                    \
            cuGetErrorString(_status, &_errstr);                    \
            ERROR("%s:%d: function %s failed with error %s.\n",     \
                __FILE__, __LINE__, #call, _errstr);                \
            exit(-1);                                               \
        }                                                           \
    } while (0)

#define RUNTIME_API_CALL(call)                                      \
    do {                                                            \
        cudaError_t _status = call;                                 \
        if (_status != cudaSuccess) {                               \
            const char *_errstr = cudaGetErrorString(_status);      \
            ERROR("%s:%d: function %s failed with error %s.\n",     \
                __FILE__, __LINE__, #call, _errstr);                \
            exit(-1);                                               \
        }                                                           \
    } while (0)

#define CUDNN_API_CALL(call)                                        \
    do {                                                            \
        cudnnStatus_t _status = call;                               \
        if (_status != CUDNN_STATUS_SUCCESS) {                      \
            const char *_errstr = cudnnGetErrorString(_status);     \
            ERROR("%s:%d: function %s failed with error %s.\n",     \
                __FILE__, __LINE__, #call, _errstr);                \
            exit(-1);                                               \
        }                                                           \
    } while (0)

#endif /* _DEBUG_H_ */
