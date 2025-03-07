# CUDA K-Means Implementation with Visualization

This project implements an efficient k-means clustering algorithm on GPUs using CUDA, based on the paper "Speeding up k-Means algorithm by GPUs" by You Li, Kaiyong Zhao, Xiaowen Chu, and Jiming Liu (Journal of Computer and System Sciences 79 (2013) 216–229).

## Key Features

- **Dimensionality-based strategies**:
  - **Low-dimensional strategy**: Utilizes GPU on-chip registers for data points with dimensions ≤ 16
  - **High-dimensional strategy**: Uses shared memory with a tile-based approach similar to matrix multiplication

- **Flexible data generation**:
  - Generate structured data with clear clusters
  - Generate completely random data for testing algorithm robustness
  
- **Interactive visualization**:
  - Visualize clustering results in 1D, 2D, and 3D
  - Works both locally and on remote/headless servers
  - Customizable visualization options

## Project Structure

- `kmeans_cuda.cu`: Main CUDA implementation of k-means
- `generate_data.cpp`: Utility to generate test data (structured or random)
- `visualize_kmeans.py`: Python script for visualizing clustering results
- `run_visualization.sh`: Shell script to automate clustering and visualization
- `Makefile`: Build configuration

## Requirements

- CUDA Toolkit (tested with CUDA 10.0+)
- C++ compiler (supporting C++11 or newer)
- Make
- Python 3 with:
  - NumPy
  - Matplotlib

## Building the Project

```bash
make
```

This will compile:
- The CUDA k-means implementation
- The data generator utility
- Set up the visualization environment

## Quick Start

The easiest way to run the entire pipeline is to use the provided script:

```bash
./run_visualization.sh
```

### Script Options

```
Usage: ./run_visualization.sh [OPTIONS]
Run k-means clustering and visualize the results.

Options:
  -d, --dimensions NUM     Set data dimensionality (1-3, default: 2)
  -n, --points NUM         Set number of data points (default: 10000)
  -k, --clusters NUM       Set number of clusters (default: 5)
  -i, --iterations NUM     Set maximum iterations (default: 50)
  -t, --threshold NUM      Set convergence threshold (default: 0.0001)
  -o, --output PREFIX      Set output file prefix (default: kmeans_out)
  -s, --save               Force save image even on local environment
  -r, --random             Generate random data instead of clustered data
  -h, --help               Show this help message
```

### Using Make Shortcuts

```bash
# For 2D visualization with clustered data
make run_visualization

# For 3D visualization with clustered data
make run_visualization_3d

# For 1D visualization with clustered data
make run_visualization_1d

# For 2D visualization with random data
make run_visualization_random

# For 3D visualization with random data
make run_visualization_random_3d
```

## Using the Components Individually

### 1. Generating Test Data

```bash
# For clustered data
./generate_data <num_points> <dimensions> <output_file> <num_clusters>

# For random data
./generate_data <num_points> <dimensions> <output_file> 0 --random
```

Examples:
```bash
# Generate 10,000 points in 2D with 5 clusters
./generate_data 10000 2 test_data.txt 5

# Generate 10,000 random points in 3D
./generate_data 10000 3 test_data.txt 0 --random
```

### 2. Running K-Means

```bash
./kmeans_cuda <data_file> <num_points> <dimensions> <num_clusters> [max_iterations] [threshold] [output_prefix]
```

Example:
```bash
./kmeans_cuda test_data.txt 10000 2 5 50 1e-4 kmeans_out
```

### 3. Visualizing Results

```bash
python3 visualize_kmeans.py --data <data_file> --assignments <assignments_file> --centroids <centroids_file> --dimensions <1-3> [--output <output_image>]
```

Example:
```bash
python3 visualize_kmeans.py --data kmeans_out_data.txt --assignments kmeans_out_assignments.txt --centroids kmeans_out_centroids.txt --dimensions 2 --output visualization.png
```

## Remote/Headless Execution

The visualization system automatically detects when running on a remote server without a display and saves output images instead of displaying interactive plots. You can download these images using `scp` or any file transfer tool.

## Implementation Details

### Low-Dimensional Strategy

For datasets with low dimensionality (d ≤ 16), we use the GPU on-chip registers to minimize memory access latency. Each thread:
1. Loads a single data point into registers
2. Computes distances to all centroids
3. Finds the minimum distance and assigns the point to that cluster

This approach significantly reduces global memory access and achieves high performance for low-dimensional data.

### High-Dimensional Strategy

For high-dimensional datasets (d > 16), we use shared memory with a tile-based approach:
1. Data is divided into tiles and loaded into shared memory
2. Each thread computes partial distances
3. Results are accumulated in registers
4. Final assignments are determined

This matrix-multiplication-like approach optimizes memory access patterns and computational efficiency for high-dimensional data.

## Performance

As described in the paper, this implementation achieves significant speedups compared to CPU and previous GPU implementations:
- 3-8× faster than previous GPU implementations for low-dimensional data
- 4-8× faster than previous GPU implementations for high-dimensional data

## Citation

```
@article{li2013speeding,
  title={Speeding up k-Means algorithm by GPUs},
  author={Li, You and Zhao, Kaiyong and Chu, Xiaowen and Liu, Jiming},
  journal={Journal of Computer and System Sciences},
  volume={79},
  number={2},
  pages={216--229},
  year={2013},
  publisher={Elsevier}
}
```

## License

MIT
