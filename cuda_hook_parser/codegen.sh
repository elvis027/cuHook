#!/bin/sh
set -x

make -C codegen
./codegen/codegen < input_function/cuda_hook_func.h > output_codegen/cuda_hook_code.cpp
./codegen/codegen < input_function/cudnn_hook_func.h > output_codegen/cudnn_hook_code.cpp
make clean -C codegen
