# cuda Hook
Hook cuda/cudnn api functions using LD_PRELOAD.

## Usage
```bash=
make
LD_PRELOAD=libcuhook.so ./app
```

## Source Info
- `hook.[h|cpp]`: Handle basic library functions loading and hook mechanism.
- `cuda_hook.[h|cpp]`: Hook and proxy cuda api functions.
- `cudnn_hook.[h|cpp]`: Hook and proxy cudnn api functions.
- `debug.h`: Debug header.
- `Makefile` Variable
    - `DEBUG_ENABLE`: Enable DEBUG messages.
    - `INFO_ENABLE`: Enable INFO messages.
    - `TRACE_DUMP_ENABLE`: Dump hooked functions trace.
    - `CUDA_HOOK_ENABLE`: Enable cuda hook mechanism.
    - `CUDA_HOOK_PROXY_ENABLE`: Enable cuda proxy mechanism.
    - `CUDNN_HOOK_ENABLE`: Enable cudnn hook mechanism.
    - `CUDNN_HOOK_PROXY_ENABLE`: Enable cudnn proxy mechanism.
- `cuda_hook_parser`: Parse cuda/cudnn api functions and generate primary hook code.
