DEBUG_ENABLE ?= 1
DUMP_ENABLE ?= 0

CUDA_HOOK_EFFECT_ENABLE ?= 1
CUDNN_HOOK_EFFECT_ENABLE ?= 1

CUDA_PATH ?= /usr/local/cuda
CUDNN_PATH ?= /usr

CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

SMS ?= 50 52 60 61 70 75 80 86
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM := $(lastword $(sort $(SMS)))
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

CXXFLAGS += -std=c++17 -fPIC -pthread
ifeq ($(DEBUG_ENABLE),1)
CXXFLAGS += -D_DEBUG_ENABLE -g -Wall
else
CXXFLAGS += -O2
endif
ifeq ($(DUMP_ENABLE),1)
CXXFLAGS += -D_DUMP_ENABLE
endif
ifeq ($(CUDA_HOOK_EFFECT_ENABLE),1)
CXXFLAGS += -D_CUDA_HOOK_EFFECT_ENABLE
endif
ifeq ($(CUDNN_HOOK_EFFECT_ENABLE),1)
CXXFLAGS += -D_CUDNN_HOOK_EFFECT_ENABLE
endif

HOST_INCLUDE +=
HOST_LDFLAGS += -ldl -lrt

CUDA_INCLUDE += -I$(CUDA_PATH)/include
CUDA_LDFLAGS += -lcuda -lcudart -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs

CUDNN_INCLUDE += -I$(CUDNN_PATH)/include
CUDNN_LDFLAGS += -lcudnn -L$(CUDNN_PATH)/lib64

ALL_INCLUDE += $(HOST_INCLUDE) $(CUDA_INCLUDE) $(CUDNN_INCLUDE)
ALL_LDFLAGS += $(HOST_LDFLAGS) $(CUDA_LDFLAGS) $(CUDNN_LDFLAGS)

TARGETS = libcuhook.so

.PHONY: all
all: $(TARGETS)

libcuhook.so: cuda_hook.o cudnn_hook.o hook.o
	$(NVCC) -shared -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ $^ $(ALL_INCLUDE) $(ALL_LDFLAGS)

cuda_hook.o: cuda_hook.cpp cuda_hook.hpp hook.hpp logging.hpp
	$(NVCC) -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $< $(ALL_INCLUDE)

cudnn_hook.o: cudnn_hook.cpp cudnn_hook.hpp hook.hpp logging.hpp
	$(NVCC) -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $< $(ALL_INCLUDE)

hook.o: hook.cpp hook.hpp logging.hpp
	$(NVCC) -m64 -Xcompiler "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $< $(ALL_INCLUDE)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o
