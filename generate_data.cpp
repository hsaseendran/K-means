#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <cstdlib>

/**
 * Simple data generator for k-means testing
 * Generates random data points with optional clustering
 */
int main(int argc, char** argv) {
    // Check arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <num_points> <dimensions> <output_file> [num_clusters]" << std::endl;
        return 1;
    }

    // Parse command line arguments
    int n = std::atoi(argv[1]);         // Number of data points
    int d = std::atoi(argv[2]);         // Dimensions
    const char* filename = argv[3];     // Output file
    int k = (argc > 4) ? std::atoi(argv[4]) : 0;  // Optional: number of clusters

    // Validate input
    if (n <= 0 || d <= 0) {
        std::cerr << "Error: Number of points and dimensions must be positive" << std::endl;
        return 1;
    }

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 100.0); // Range of values

    // Open output file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    if (k <= 0) {
        // Generate completely random data
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                file << dist(gen);
                if (j < d - 1) file << " ";
            }
            file << std::endl;
        }
    } else {
        // Generate clustered data
        // Create k cluster centers
        float** centers = new float*[k];
        for (int i = 0; i < k; i++) {
            centers[i] = new float[d];
            for (int j = 0; j < d; j++) {
                centers[i][j] = dist(gen);
            }
        }

        // Generate points around the centers
        std::normal_distribution<float> cluster_dist(0.0, 5.0); // Variance around cluster centers
        
        for (int i = 0; i < n; i++) {
            // Randomly select a cluster
            int cluster = i % k;  // Distribute points evenly among clusters
            
            // Generate point around the cluster center
            for (int j = 0; j < d; j++) {
                file << centers[cluster][j] + cluster_dist(gen);
                if (j < d - 1) file << " ";
            }
            file << std::endl;
        }

        // Cleanup cluster centers
        for (int i = 0; i < k; i++) {
            delete[] centers[i];
        }
        delete[] centers;
    }

    file.close();
    std::cout << "Generated " << n << " data points with " << d << " dimensions to " << filename << std::endl;
    
    return 0;
}