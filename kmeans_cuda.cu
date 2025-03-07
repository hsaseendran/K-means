#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Constants
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define TDIM_Y 2
#define BLOCK_SIZE 256

/**
 * K-means clustering algorithm implementation based on:
 * "Speeding up k-Means algorithm by GPUs" by Li et al.
 *
 * This implementation contains two strategies:
 * 1. Register-based approach for low-dimensional data
 * 2. Shared memory approach for high-dimensional data
 */

// Structure to hold dataset information
typedef struct {
    float* data;
    int n;       // number of data points
    int d;       // dimensionality
    int k;       // number of clusters
    float* centroids;
    int* assignments;  // cluster assignment for each point
    int* counts;       // count of points in each cluster
    int max_iterations;
    float threshold;
} KMeansData;

// Utility functions
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Allocate memory for K-means data
void allocateMemory(KMeansData* data) {
    // Host memory
    data->centroids = (float*)malloc(data->k * data->d * sizeof(float));
    data->assignments = (int*)malloc(data->n * sizeof(int));
    data->counts = (int*)malloc(data->k * sizeof(int));
    
    if (!data->centroids || !data->assignments || !data->counts) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

// Initialize centroids randomly from data points
void initializeCentroids(KMeansData* data) {
    // Use the first k data points as initial centroids
    // In a real implementation, you might want to use k-means++ or random selection
    for (int i = 0; i < data->k; i++) {
        for (int j = 0; j < data->d; j++) {
            data->centroids[i * data->d + j] = data->data[i * data->d + j];
        }
    }
}

// Free allocated memory
void freeMemory(KMeansData* data, float* d_data, float* d_centroids, int* d_assignments) {
    free(data->centroids);
    free(data->assignments);
    free(data->counts);
    
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
}

/***************** Low-Dimensional Strategy (Register-Based) ******************/

// CUDA kernel for finding the closest centroid (register-based for low dimensional data)
__global__ void findClosestCentroidLowDim(float* data, float* centroids, int* assignments, 
                                       int n, int d, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float min_dist = FLT_MAX;
        int closest = 0;
        
        // Load data point into registers
        float point[16];  // Assuming max dimension of 16 for registers
        for (int j = 0; j < d; j++) {
            point[j] = data[tid * d + j];
        }
        
        // Find closest centroid
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            
            // Calculate Euclidean distance
            for (int j = 0; j < d; j++) {
                float diff = point[j] - centroids[c * d + j];
                dist += diff * diff;
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                closest = c;
            }
        }
        
        assignments[tid] = closest;
    }
}

/***************** High-Dimensional Strategy (Shared Memory) ******************/

// CUDA kernel for finding the closest centroid (shared memory-based for high dimensional data)
__global__ void findClosestCentroidHighDim(float* data, float* centroids, int* assignments,
                                        int n, int d, int k) {
    __shared__ float SMData[TILE_WIDTH][TILE_HEIGHT];
    
    // Calculate indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global indices
    int indexD = data + (by * TILE_HEIGHT * d) + (ty * d) + tx;
    int indexC = centroids + (bx * TILE_WIDTH) + (ty * blockDim.x) + tx;
    int indexR = assignments + (by * TILE_HEIGHT * k) + (bx * TILE_WIDTH) + (ty * blockDim.x) + tx;
    
    // Temporary result stored in registers
    float TResult[TILE_WIDTH];
    for (int i = 0; i < TILE_WIDTH; i++) {
        TResult[i] = 0.0f;
    }
    
    // Process data in tiles
    float* Alast = indexD + d;
    while (indexD < Alast) {
        // Load tile of data into shared memory
        if (tx < d && (by * TILE_HEIGHT + ty) < n) {
            SMData[tx][ty] = *indexD;
        }
        __syncthreads();
        
        // Compute distance for this tile
        for (int i = 0; i < TILE_WIDTH && i < d; i++) {
            if ((by * TILE_HEIGHT + ty) < n && (bx * TILE_WIDTH + tx) < k) {
                float diff = SMData[i][ty] - centroids[bx * TILE_WIDTH + tx + i * k];
                TResult[tx] += diff * diff;
            }
        }
        
        indexD += TILE_WIDTH;
        __syncthreads();
    }
    
    // Find minimum distance and assign cluster
    if ((by * TILE_HEIGHT + ty) < n && (bx * TILE_WIDTH + tx) < k) {
        float minDist = TResult[0];
        int minIndex = 0;
        
        for (int i = 1; i < TILE_WIDTH && i < k; i++) {
            if (TResult[i] < minDist) {
                minDist = TResult[i];
                minIndex = i;
            }
        }
        
        assignments[by * TILE_HEIGHT + ty] = bx * TILE_WIDTH + minIndex;
    }
}

// CUDA kernel to update centroids
__global__ void computeNewCentroidsKernel(float* data, int* assignments, float* centroids, 
                                       int* counts, int n, int d, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < k * d) {
        int centroid_idx = tid / d;
        int dim = tid % d;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int i = 0; i < n; i++) {
            if (assignments[i] == centroid_idx) {
                sum += data[i * d + dim];
                count++;
            }
        }
        
        if (count > 0) {
            centroids[tid] = sum / count;
        }
        
        if (dim == 0) {
            counts[centroid_idx] = count;
        }
    }
}

// Compute new centroids on CPU
void computeNewCentroidsCPU(KMeansData* data, float* d_data, int* d_assignments, float* d_centroids) {
    // Copy assignments back to host
    cudaMemcpy(data->assignments, d_assignments, data->n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Reset counts and centroids
    for (int i = 0; i < data->k; i++) {
        data->counts[i] = 0;
        for (int j = 0; j < data->d; j++) {
            data->centroids[i * data->d + j] = 0.0f;
        }
    }
    
    // Sum up all points assigned to each centroid
    for (int i = 0; i < data->n; i++) {
        int cluster = data->assignments[i];
        data->counts[cluster]++;
        
        for (int j = 0; j < data->d; j++) {
            data->centroids[cluster * data->d + j] += data->data[i * data->d + j];
        }
    }
    
    // Divide by count to get means
    for (int i = 0; i < data->k; i++) {
        if (data->counts[i] > 0) {
            for (int j = 0; j < data->d; j++) {
                data->centroids[i * data->d + j] /= data->counts[i];
            }
        }
    }
    
    // Copy updated centroids back to device
    cudaMemcpy(d_centroids, data->centroids, data->k * data->d * sizeof(float), cudaMemcpyHostToDevice);
}

// Check for convergence
bool hasConverged(float* old_centroids, float* new_centroids, int k, int d, float threshold) {
    for (int i = 0; i < k; i++) {
        float distance = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = old_centroids[i * d + j] - new_centroids[i * d + j];
            distance += diff * diff;
        }
        distance = sqrt(distance);
        if (distance > threshold) {
            return false;
        }
    }
    return true;
}

// Main K-means function for low-dimensional data
void kmeansLowDim(KMeansData* data) {
    float *d_data, *d_centroids;
    int *d_assignments;
    
    // Allocate device memory
    cudaMalloc((void**)&d_data, data->n * data->d * sizeof(float));
    cudaMalloc((void**)&d_centroids, data->k * data->d * sizeof(float));
    cudaMalloc((void**)&d_assignments, data->n * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_data, data->data, data->n * data->d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, data->centroids, data->k * data->d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Previous centroids for convergence check
    float* old_centroids = (float*)malloc(data->k * data->d * sizeof(float));
    
    // Configure kernel
    int blockSize = BLOCK_SIZE;
    int gridSize = (data->n + blockSize - 1) / blockSize;
    
    // Main loop
    for (int iter = 0; iter < data->max_iterations; iter++) {
        // Save current centroids for convergence check
        memcpy(old_centroids, data->centroids, data->k * data->d * sizeof(float));
        
        // Find closest centroid for each data point
        findClosestCentroidLowDim<<<gridSize, blockSize>>>(d_data, d_centroids, d_assignments, 
                                                       data->n, data->d, data->k);
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Kernel execution failed");
        
        // Compute new centroids
        computeNewCentroidsCPU(data, d_data, d_assignments, d_centroids);
        
        // Check for convergence
        if (hasConverged(old_centroids, data->centroids, data->k, data->d, data->threshold)) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }
    
    // Copy final assignments back to host
    cudaMemcpy(data->assignments, d_assignments, data->n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Clean up
    free(old_centroids);
    freeMemory(data, d_data, d_centroids, d_assignments);
}

// Main K-means function for high-dimensional data
void kmeansHighDim(KMeansData* data) {
    float *d_data, *d_centroids;
    int *d_assignments;
    
    // Allocate device memory
    cudaMalloc((void**)&d_data, data->n * data->d * sizeof(float));
    cudaMalloc((void**)&d_centroids, data->k * data->d * sizeof(float));
    cudaMalloc((void**)&d_assignments, data->n * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_data, data->data, data->n * data->d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, data->centroids, data->k * data->d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Previous centroids for convergence check
    float* old_centroids = (float*)malloc(data->k * data->d * sizeof(float));
    
    // Configure kernel for high-dimensional strategy
    dim3 dimBlock(TILE_WIDTH, TDIM_Y);
    dim3 dimGrid((data->k + TILE_WIDTH - 1) / TILE_WIDTH, 
                (data->n + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    // Main loop
    for (int iter = 0; iter < data->max_iterations; iter++) {
        // Save current centroids for convergence check
        memcpy(old_centroids, data->centroids, data->k * data->d * sizeof(float));
        
        // Find closest centroid for each data point using shared memory strategy
        findClosestCentroidHighDim<<<dimGrid, dimBlock>>>(d_data, d_centroids, d_assignments, 
                                                       data->n, data->d, data->k);
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Kernel execution failed");
        
        // Compute new centroids
        computeNewCentroidsCPU(data, d_data, d_assignments, d_centroids);
        
        // Check for convergence
        if (hasConverged(old_centroids, data->centroids, data->k, data->d, data->threshold)) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }
    
    // Copy final assignments back to host
    cudaMemcpy(data->assignments, d_assignments, data->n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Clean up
    free(old_centroids);
    freeMemory(data, d_data, d_centroids, d_assignments);
}

// Main entry point
int main(int argc, char** argv) {
    // Check for command line arguments
    if (argc < 5) {
        printf("Usage: %s <data_file> <n> <d> <k> [max_iterations] [threshold]\n", argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    char* filename = argv[1];
    int n = atoi(argv[2]);  // number of data points
    int d = atoi(argv[3]);  // dimensionality
    int k = atoi(argv[4]);  // number of clusters
    int max_iterations = (argc > 5) ? atoi(argv[5]) : 100;
    float threshold = (argc > 6) ? atof(argv[6]) : 1e-4;
    
    // Initialize k-means data
    KMeansData kmeans_data;
    kmeans_data.n = n;
    kmeans_data.d = d;
    kmeans_data.k = k;
    kmeans_data.max_iterations = max_iterations;
    kmeans_data.threshold = threshold;
    
    // Allocate memory for data
    kmeans_data.data = (float*)malloc(n * d * sizeof(float));
    if (!kmeans_data.data) {
        fprintf(stderr, "Error: Memory allocation failed for data\n");
        return 1;
    }
    
    // Read data from file
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        free(kmeans_data.data);
        return 1;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            if (fscanf(file, "%f", &kmeans_data.data[i * d + j]) != 1) {
                fprintf(stderr, "Error: Invalid data format\n");
                fclose(file);
                free(kmeans_data.data);
                return 1;
            }
        }
    }
    fclose(file);
    
    // Allocate memory for results
    allocateMemory(&kmeans_data);
    
    // Initialize centroids
    initializeCentroids(&kmeans_data);
    
    // Start timer
    clock_t start = clock();
    
    // Run k-means algorithm based on dimensionality
    if (d <= 16) { // Low-dimensional strategy using registers
        printf("Using low-dimensional strategy (register-based)\n");
        kmeansLowDim(&kmeans_data);
    } else { // High-dimensional strategy using shared memory
        printf("Using high-dimensional strategy (shared memory)\n");
        kmeansHighDim(&kmeans_data);
    }
    
    // End timer
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("K-means completed in %.3f seconds\n", elapsed);
    
    // Output results
    printf("Final centroids:\n");
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < d; j++) {
            printf("%.4f ", kmeans_data.centroids[i * d + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(kmeans_data.data);
    free(kmeans_data.centroids);
    free(kmeans_data.assignments);
    free(kmeans_data.counts);
    
    return 0;
}