# Makefile for CUDA K-Means Implementation

# CUDA and C++ compiler
NVCC = nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -O3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70
CXX_FLAGS = -O3 -Wall -Wextra

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories
LIB_DIRS = -L/usr/local/cuda/lib64

# Libraries
LIBS = -lcudart -lcuda

# Source files
SOURCES = kmeans_cuda.cu

# Executables
EXE = kmeans_cuda

# Default target
all: $(EXE)

# Build the main executable
$(EXE): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIB_DIRS) $(LIBS) -o $@ $<

# Clean up
clean:
	rm -f $(EXE)

# Run with test data
run_test: $(EXE)
	./$(EXE) test_data.txt 10000 2 10 50 1e-4

# Generate random test data
generate_data:
	$(CXX) $(CXX_FLAGS) -o generate_data generate_data.cpp
	./generate_data 10000 2 test_data.txt

.PHONY: all clean run_test generate_data