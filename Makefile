# Makefile for CUDA K-Means Implementation with Visualization

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
DATA_GEN = generate_data.cpp
VISUALIZATION = visualize_kmeans.py
RUN_SCRIPT = run_visualization.sh

# Executables
EXE = kmeans_cuda
DATA_GEN_EXE = generate_data

# Python requirements for visualization
PY_REQUIREMENTS = numpy matplotlib

# Default target
all: $(EXE) $(DATA_GEN_EXE) setup_visualization

# Build the main CUDA executable
$(EXE): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIB_DIRS) $(LIBS) -o $@ $<

# Build the data generator
$(DATA_GEN_EXE): $(DATA_GEN)
	$(CXX) $(CXX_FLAGS) -o $@ $<

# Make the run script executable
setup_visualization: 
	@echo "Setting up visualization environment..."
	@chmod +x $(RUN_SCRIPT)
	@echo "Checking Python dependencies..."
	@for pkg in $(PY_REQUIREMENTS); do \
		python3 -c "import $$pkg" 2>/dev/null || (echo "Please install $$pkg with: pip install $$pkg" && exit 1); \
	done
	@echo "Setup complete. You can now run './run_visualization.sh' to generate and visualize clusters."

# Clean up
clean:
	rm -f $(EXE) $(DATA_GEN_EXE)
	rm -f kmeans_out_*.txt
	rm -f kmeans_out_*.png
	rm -f test_data.txt

# Run with visualization (2D example)
run_visualization: all
	./run_visualization.sh --dimensions 2 --points 10000 --clusters 5

# Run with 3D visualization
run_visualization_3d: all
	./run_visualization.sh --dimensions 3 --points 10000 --clusters 5

# Run with 1D visualization
run_visualization_1d: all
	./run_visualization.sh --dimensions 1 --points 10000 --clusters 5

# Run with random data (not clustered)
run_visualization_random: all
	./run_visualization.sh --dimensions 2 --points 10000 --clusters 5 --random

# Run with 3D random data
run_visualization_random_3d: all
	./run_visualization.sh --dimensions 3 --points 10000 --clusters 5 --random

# Generate test data (clustered)
generate_data: $(DATA_GEN_EXE)
	./$(DATA_GEN_EXE) 10000 2 test_data.txt 5

# Generate random test data
generate_random_data: $(DATA_GEN_EXE)
	./$(DATA_GEN_EXE) 10000 2 test_data.txt 0 --random

.PHONY: all clean run_visualization run_visualization_3d run_visualization_1d run_visualization_random run_visualization_random_3d generate_data generate_random_data setup_visualization