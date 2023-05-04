# cuda Hook
Hook cuda/cudnn api functions using LD_PRELOAD.

## Usage
```bash=
make
LD_PRELOAD=libcuhook.so ./app
```

## Source Info
- `hook.[hpp|cpp]`: dlsym hook mechanism and handle basic library functions loading.
- `cuda_hook.[hpp|cpp]`: Hook and proxy cuda api functions.
- `cudnn_hook.[hpp|cpp]`: Hook and proxy cudnn api functions.
- `logging.hpp`: Logging.
- `Makefile` Variable
    - `DEBUG_ENABLE`: Enable DEBUG messages.
    - `DUMP_ENABLE`: Dump hooked functions trace.
    - `CUDA_HOOK_EFFECT_ENABLE`: Enable cuda hook effect.
    - `CUDNN_HOOK_EFFECT_ENABLE`: Enable cudnn hook effect.
- `cuda_hook_parser/`: Parse cuda/cudnn api functions and generate primary hook code.
