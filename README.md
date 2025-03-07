# CUDA K-Means Implementation

This project implements the k-means clustering algorithm on GPUs using CUDA, based on the paper "Speeding up k-Means algorithm by GPUs" by You Li, Kaiyong Zhao, Xiaowen Chu, and Jiming Liu (Journal of Computer and System Sciences 79 (2013) 216–229).

## Key Features

- Two different strategies for k-means clustering:
  - **Low-dimensional strategy**: Utilizes GPU on-chip registers for data points with dimensions ≤ 16
  - **High-dimensional strategy**: Uses shared memory with a tile-based approach similar to matrix multiplication

## Project Structure

- `kmeans_cuda.cu`: Main CUDA implementation of k-means
- `Makefile`: Build configuration
- `generate_data.cpp`: Utility to generate test data
- `README.md`: This file

## Requirements

- CUDA Toolkit (tested with CUDA 10.0+)
- C++ compiler (supporting C++11 or newer)
- Make

## Building the Project

```bash
make
```

## Generating Test Data

```bash
make generate_data
./generate_data 10000 2 test_data.txt 5
```

Arguments:
- Number of data points
- Dimensions
- Output file
- (Optional) Number of clusters for synthetic data

## Running the Algorithm

```bash
./kmeans_cuda test_data.txt 10000 2 5 50 1e-4
```

Arguments:
- Data file path
- Number of data points (n)
- Dimensions (d)
- Number of clusters (k)
- (Optional) Maximum iterations (default: 100)
- (Optional) Convergence threshold (default: 1e-4)

## Implementation Details

### Low-Dimensional Strategy

For datasets with low dimensionality (d ≤ 16), we use the GPU on-chip registers to minimize memory access latency. Each thread:
1. Loads a single data point into registers
2. Computes distances to all centroids
3. Finds the minimum distance and assigns the point to that cluster

### High-Dimensional Strategy

For high-dimensional datasets (d > 16), we use shared memory with a tile-based approach:
1. Data is divided into tiles and loaded into shared memory
2. Each thread computes partial distances
3. Results are accumulated in registers
4. Final assignments are determined

### Performance

As described in the paper, this implementation achieves a significant speedup compared to CPU implementations:
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
