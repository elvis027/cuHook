#ifndef _HOOK_HPP_
#define _HOOK_HPP_

#include "logging.hpp"

#define STRINGIFY(x) #x
#define SYMBOL_STRING(x) STRINGIFY(x)

extern void *libdl_handle;
extern void *libcuda_handle;
extern void *libcudnn_handle;
void *actual_dlsym(void *handle, const char *symbol);

extern logging::log hook_log;
extern logging::log trace_dump;

#endif /* _HOOK_HPP_ */
