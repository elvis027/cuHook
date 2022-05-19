DEBUG_ENABLE ?= 1
INFO_ENABLE ?= 1
TRACE_DUMP_ENABLE ?= 1

CUDA_HOOK_ENABLE ?= 1
CUDA_HOOK_PROXY_ENABLE ?= 0
CUDNN_HOOK_ENABLE ?= 1
CUDNN_HOOK_PROXY_ENABLE ?= 0

CUDA_PATH ?= /opt/cuda
CUDNN_PATH ?= /usr

CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

SMS ?= 35 37 50 52 60 61 70 75 80 86
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

CXXFLAGS += -std=c++11 -fPIC -pthread
ifeq ($(DEBUG_ENABLE),1)
CXXFLAGS += -D_DEBUG_ENABLE -g -Wall
else
CXXFLAGS += -O2
endif
ifeq ($(INFO_ENABLE),1)
CXXFLAGS += -D_INFO_ENABLE
endif
ifeq ($(TRACE_DUMP_ENABLE),1)
CXXFLAGS += -D_TRACE_DUMP_ENABLE
endif

ifeq ($(CUDA_HOOK_ENABLE),1)
CXXFLAGS += -D_CUDA_HOOK_ENABLE
endif
ifeq ($(CUDA_HOOK_PROXY_ENABLE),1)
CXXFLAGS += -D_CUDA_HOOK_PROXY_ENABLE
endif
ifeq ($(CUDNN_HOOK_ENABLE),1)
CXXFLAGS += -D_CUDNN_HOOK_ENABLE
endif
ifeq ($(CUDNN_HOOK_PROXY_ENABLE),1)
CXXFLAGS += -D_CUDNN_HOOK_PROXY_ENABLE
endif

CUDA_INCLUDE += -I$(CUDA_PATH)/include
CUDNN_INCLUDE += -I$(CUDNN_PATH)/include
NVCC_INCLUDE += $(CUDA_INCLUDE) $(CUDNN_INCLUDE)

LDFLAGS += -ldl -lrt
CUDA_LDFLAGS += -lcuda -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs
CUDNN_LDFLAGS += -lcudnn -L$(CUDNN_PATH)/lib64
NVCC_LDFLAGS += $(CUDA_LDFLAGS) $(CUDNN_LDFLAGS)

TARGETS = libcuhook.so

.PHONY: all
all: $(TARGETS)

libcuhook.so: hook.o cuda_hook.o cudnn_hook.o
	$(NVCC) -shared -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ $^ $(NVCC_INCLUDE) $(NVCC_LDFLAGS) $(LDFLAGS)

hook.o: hook.cpp hook.h debug.h
	$(NVCC) -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $< $(NVCC_INCLUDE)

cuda_hook.o: cuda_hook.cpp cuda_hook.h hook.h debug.h
	$(NVCC) -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $< $(NVCC_INCLUDE)

cudnn_hook.o: cudnn_hook.cpp cudnn_hook.h hook.h debug.h
	$(NVCC) -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $< $(NVCC_INCLUDE)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o
