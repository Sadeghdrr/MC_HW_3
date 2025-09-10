# Makefile for HW3 – High-Performance Multithreaded Regex Matching
#
# Build targets:
#   make              - Optimized build (default)
#   make debug        - Debug build with symbols
#   make clean        - Remove binaries & result files content
#   make run          - Quick demo run (default parameters)
#   make perf-test-cpu- Comprehensive CPU performance sweep
#   make perf-test-gpu- Comprehensive GPU performance tests
#   make perf-test-all- Run both CPU and GPU performance tests
#   make help         - Print this help message
#
# Notes:
#  - Uses cuDF/RMM from the ACTIVE conda env (CONDA_PREFIX).
#  - Auto-detects CUDA root from 'which nvcc' (works with conda cudatoolkit).
#  - Hyperscan is linked WITHOUT pkg-config. It tries these locations (in order):
#       $(HOME)/.local, /usr/local, /usr/local/hyperscan/build
#    You can change HS_PREFIX below if needed.
#  - GPU arch set to sm_75 for NVIDIA T4.
#
# ---------------------------------------------------------------------------
# Toolchain
NVCC       := nvcc
NVCCFLAGS  := -O3 -std=c++20 -arch=sm_75 -Xcompiler -Wall,-Wextra,-fPIC \
              -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

# ---------------------------------------------------------------------------
# RAPIDS/cuDF (from active conda env) + CUDA (from nvcc path)
RAPIDS_ROOT ?= $(CONDA_PREFIX)
ifeq ($(strip $(RAPIDS_ROOT)),)
  $(error No conda environment active. Run: 'conda activate rapids-25.08')
endif
CUDA_ROOT ?= $(shell dirname $$(dirname $$(realpath $$(which nvcc))))

# ---------------------------------------------------------------------------
# Hyperscan (no pkg-config): search order -> $(HOME)/.local, /usr/local, /usr/local/hyperscan/build
# You can override HS_PREFIX at 'make' time, but it's NOT required.
HS_PREFIX   ?= $(HOME)/.local
HS_INC_DIRS := \
  $(HS_PREFIX)/include \
  $(HS_PREFIX)/include/hs \
  /usr/local/include \
  /usr/local/include/hs \
  /usr/local/hyperscan/include \
  /usr/local/hyperscan/include/hs
HS_LIB_DIRS := \
  $(HS_PREFIX)/lib \
  /usr/local/lib \
  /usr/local/hyperscan/build/lib

# Compose include & lib flags (keep both include and include/hs for safety)
HS_INC := $(foreach d,$(HS_INC_DIRS),-I$(d))
HS_LIB := $(foreach d,$(HS_LIB_DIRS),-L$(d)) -lhs

# ---------------------------------------------------------------------------
# Includes / Libs
INCLUDES := -I$(RAPIDS_ROOT)/include \
            -I$(RAPIDS_ROOT)/include/libcudf/libcudacxx \
            -I$(CUDA_ROOT)/include \
            $(HS_INC)

# Library search paths + libraries
LDFLAGS  := -L$(RAPIDS_ROOT)/lib \
            -L$(RAPIDS_ROOT)/lib64 \
            -L$(CUDA_ROOT)/lib64 \
            -L/usr/lib/x86_64-linux-gnu \
            $(HS_LIB) \
            -lcudf -lrmm -lrapids_logger -lcudart -lpthread -ldl

# nvcc does not accept '-Wl,....' directly; pass rpath via -Xlinker
RPATH_DIRS := $(RAPIDS_ROOT)/lib $(RAPIDS_ROOT)/lib64 $(CUDA_ROOT)/lib64 \
              $(HS_PREFIX)/lib /usr/local/lib /usr/local/hyperscan/build/lib
RPFLAGS := $(foreach d,$(RPATH_DIRS),-Xlinker -rpath -Xlinker $(d))

# ---------------------------------------------------------------------------
# Layout
SRC_DIR     := src
BIN_DIR     := bin
BUILD_DIR   := build
RESULTS_DIR := results

# Final binary name (kept)
TARGET := $(BIN_DIR)/HW3_MCC_030402_401106039

# Only compile files under src/
CU_SRCS   := $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS  := $(wildcard $(SRC_DIR)/*.cpp)
C_SRCS    := $(wildcard $(SRC_DIR)/*.c)

# Map source files to build/*.o (preserve filenames, change dir)
CU_OBJS   := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS  := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
C_OBJS    := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
OBJS      := $(CU_OBJS) $(CPP_OBJS) $(C_OBJS)

# ---------------------------------------------------------------------------
# Defaults for 'make run'
DEFAULT_RULES   := rules.txt
DEFAULT_INPUT   := set1.txt
DEFAULT_THREADS := 4
DEFAULT_MODE    := cpu
GPU_STREAMS		:= 6
GPU_WAVE 		:= 32
GPU_CHUNK_MB	:= 64

# ---------------------------------------------------------------------------
# Build rules
.PHONY: all debug clean run perf-test-cpu perf-test-gpu perf-test-all help

all: $(TARGET)

debug: NVCCFLAGS := -g -G -O0 -std=c++20 -arch=sm_75 -Xcompiler -Wall,-Wextra,-fPIC \
                    -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
debug: clean $(TARGET)

# Ensure dirs exist
dirs:
	@mkdir -p $(BIN_DIR) $(BUILD_DIR) $(RESULTS_DIR)

# Compile rules write .o into build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x c++ -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x c -c $< -o $@

$(TARGET): $(OBJS) | dirs
	@echo "Linking $(TARGET) ..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OBJS) -o $@ $(LDFLAGS) $(RPFLAGS)
	@echo "Build completed: $@"

clean:
	@echo "Cleaning up..."
	@rm -f $(BUILD_DIR)/*
	@rm -f $(BIN_DIR)/*
	@rm -f $(RESULTS_DIR)/*

# ---------------------------------------------------------------------------
run: $(TARGET) | dirs
	@echo "Running default demo with $(DEFAULT_MODE) mode and $(DEFAULT_THREADS) threads..."
	@$(TARGET) --mode $(DEFAULT_MODE) --rules sample_rules.txt --input sample_input.txt --threads $(DEFAULT_THREADS)
	@echo "Demo run complete."

perf-test-cpu: $(TARGET) | dirs
	@echo "Running comprehensive CPU performance tests..."
	@for dataset in set1 set2 set3; do \
		echo "================================================="; \
		echo "              Testing Dataset: $${dataset}.txt   "; \
		echo "================================================="; \
		for threads in 1 2 4 8 16 32; do \
			echo "Testing with $$threads thread(s)..."; \
			$(TARGET) --mode cpu --rules $(DEFAULT_RULES) --input $${dataset}.txt --threads $$threads; \
		done; \
	done
	@echo "CPU performance tests complete."

perf-test-gpu: $(TARGET) | dirs
	@echo "Running comprehensive GPU performance tests..."
	@for dataset in set1.txt set2.txt set3.txt; do \
		echo "================================================="; \
		echo "              Testing Dataset: $${dataset}       "; \
		echo "================================================="; \
		echo "Testing with GPU..."; \
		$(TARGET) --mode gpu --rules $(DEFAULT_RULES) --input $${dataset} --gpu-streams $(GPU_STREAMS) --gpu-wave $(GPU_WAVE) --gpu-chunk-mb $(GPU_CHUNK_MB); \
			echo \"Using GPU flags: streams=$(GPU_STREAMS) wave=$(GPU_WAVE) chunk=$(GPU_CHUNK_MB)MB\"; \
	done
	@echo "GPU performance tests complete."

perf-test-all: perf-test-cpu perf-test-gpu

help:
	@echo "Targets:"
	@echo "  make              - Build optimized binary"
	@echo "  debug             - Build debug binary"
	@echo "  clean             - Remove generated files"
	@echo "  run               - Run a quick demo with default settings"
	@echo "  perf-test-cpu     - Run a performance sweep for the CPU version"
	@echo "  perf-test-gpu     - Run GPU performance tests with cuDF/RAPIDS"
	@echo "  perf-test-all     - Run both CPU and GPU performance tests"
	@echo ""
	@echo "Usage examples:"
	@echo "  make"
	@echo "  make run"
	@echo "  make perf-test-cpu"
	@echo "  make perf-test-gpu"
	@echo "  make perf-test-all"
	@echo "  ./bin/HW3_MCC_030402_401106039 --mode cpu --rules rules.txt --input set1.txt --threads 4"
	@echo "  ./bin/HW3_MCC_030402_401106039 --mode gpu --rules rules.txt --input set1.txt"