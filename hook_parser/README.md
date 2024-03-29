# Hook Parser
Parse cuda/cudnn api functions and generate primary hook code.

## Usage
```bash=
make -C codegen
./codegen/codegen < input_function/cuda_hook_func.h > output_codegen/cuda_hook_code.cpp
./codegen/codegen < input_function/cudnn_hook_func.h > output_codegen/cudnn_hook_code.cpp
```
or
```bash=
sh codegen.sh
```

## Source Info
- `[cuda|cudnn]_header/`: The official header files of cuda/cudnn api functions.
- `input_function/[cuda|cudnn]_hook_func.h`: List of cuda/cudnn api functions to be hooked.
- `output_codegen/[cuda|cudnn]_hook_code.cpp`: The primary hook code generated by parser.
- `codegen/`:
    - `scanner.l`: Lexer.
    - `parser.y`: Parser.
    - `codegen.[h|c]`: Codegen.

## Hooked Function Info
- cuda v11.7.64
    - Hooked cuda driver api functions related to **device memory** and **kernel launch**.
    - cuda special function
        - `CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);`
            - This function is the entry of cuda api functions for cuda version >= 11.3.
            - Hook mechanism must be applied in this function.
- cudnn v8.4.0
    - Hooked all cudnn api functions.
    - cudnn special function
        - `size_t CUDNNWINAPI cudnnGetVersion(void);`: Ignored.
        - `size_t CUDNNWINAPI cudnnGetCudartVersion(void);`: Ignored.
        - `const char *CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status);`: Ignored.

## Dependencies
- `gcc`
- `flex`
- `byacc`
