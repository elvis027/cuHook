#ifndef _HOOK_H_
#define _HOOK_H_

#define STRINGIFY(x) #x
#define SYMBOL_STRING(x) STRINGIFY(x)

extern void *libdlHandle;
extern void *libcudaHandle;
extern void *libcudnnHandle;
void *actualDlsym(void *handle, const char *symbol);

#ifdef _TRACE_DUMP_ENABLE
extern FILE *fp_trace;
#define DUMP_TRACE(...)                     \
    do {                                    \
        fprintf(fp_trace, __VA_ARGS__);     \
        fflush(fp_trace);                   \
    } while (0)
#else
#define DUMP_TRACE(...)
#endif /* _TRACE_DUMP_ENABLE */

#endif /* _HOOK_H_ */
