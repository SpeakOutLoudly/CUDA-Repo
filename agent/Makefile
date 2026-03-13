# Copyright 2026

HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
ifneq ($(wildcard $(CUDA_PATH)/bin/nvcc),)
NVCC_BIN ?= $(CUDA_PATH)/bin/nvcc
else
NVCC_BIN ?= $(shell command -v nvcc 2>/dev/null)
endif
ifeq ($(NVCC_BIN),)
NVCC_BIN := $(CUDA_PATH)/bin/nvcc
endif
CUDA_INCLUDE_PATH ?= $(CUDA_PATH)/include
ifeq ($(wildcard $(CUDA_INCLUDE_PATH)),)
CUDA_INCLUDE_PATH := /usr/include
endif
NVCC := $(NVCC_BIN) -ccbin $(HOST_COMPILER)

NVCCFLAGS := -m$(shell getconf LONG_BIT)
CCFLAGS := -fPIC
LDFLAGS :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += --threads 0 --std=c++14
ALL_CCFLAGS += -maxrregcount=255
ALL_CCFLAGS += --use_fast_math -lineinfo
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

INCLUDES := -I$(CUDA_INCLUDE_PATH) -I.
LIBRARIES := -lcublas -lcusparse -lcusparseLt

SMS ?= 86
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm) -gencode arch=compute_$(sm),code=compute_$(sm)))

BIN_DIR ?= ./bin
OBJ_DIR ?= ./obj

TARGET := $(BIN_DIR)/kernelTest
OBJECTS := $(OBJ_DIR)/kernelTest.o $(OBJ_DIR)/benchmark_runner.o $(OBJ_DIR)/SpMM_API.o
HEAD_FILES := kernelTest.cu benchmark_runner.cu benchmark_runner.h SpMM_API.cu SpMM_API.h utils.h backends/cuSparseLt_backend.cuh

.PHONY: all clean

all: $(TARGET)

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

$(OBJ_DIR)/kernelTest.o: kernelTest.cu benchmark_runner.h | $(OBJ_DIR)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ_DIR)/benchmark_runner.o: benchmark_runner.cu $(HEAD_FILES) | $(OBJ_DIR)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ_DIR)/SpMM_API.o: SpMM_API.cu SpMM_API.h utils.h backends/cuSparseLt_backend.cuh | $(OBJ_DIR)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)
