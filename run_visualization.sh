#!/bin/bash
# Shell script to generate test data and run the k-means visualization
# with automatic detection of remote environment

# Default settings
DIMENSIONS=2
POINTS=10000
CLUSTERS=5
MAX_ITERATIONS=50
THRESHOLD=0.0001
OUTPUT_PREFIX="kmeans_out"
FORCE_SAVE=false
RANDOM_DATA=false

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run k-means clustering and visualize the results."
    echo
    echo "Options:"
    echo "  -d, --dimensions NUM     Set data dimensionality (1-3, default: 2)"
    echo "  -n, --points NUM         Set number of data points (default: 10000)"
    echo "  -k, --clusters NUM       Set number of clusters (default: 5)"
    echo "  -i, --iterations NUM     Set maximum iterations (default: 50)"
    echo "  -t, --threshold NUM      Set convergence threshold (default: 0.0001)"
    echo "  -o, --output PREFIX      Set output file prefix (default: kmeans_out)"
    echo "  -s, --save               Force save image even on local environment"
    echo "  -r, --random             Generate random data instead of clustered data"
    echo "  -h, --help               Show this help message"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dimensions)
            DIMENSIONS="$2"
            shift 2
            ;;
        -n|--points)
            POINTS="$2"
            shift 2
            ;;
        -k|--clusters)
            CLUSTERS="$2"
            shift 2
            ;;
        -i|--iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        -s|--save)
            FORCE_SAVE=true
            shift
            ;;
        -r|--random)
            RANDOM_DATA=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate dimensions
if [[ "$DIMENSIONS" -lt 1 || "$DIMENSIONS" -gt 3 ]]; then
    echo "Error: Dimensions must be between 1 and 3"
    exit 1
fi

# Check if running in a remote/headless environment
REMOTE_ENV=false
if [[ -z "$DISPLAY" ]]; then
    REMOTE_ENV=true
    echo "Detected headless environment - will save output files automatically"
fi

# Generate test data
echo "Generating test data with $DIMENSIONS dimensions..."
if [[ "$RANDOM_DATA" == "true" ]]; then
    ./generate_data $POINTS $DIMENSIONS test_data.txt 0 --random
else
    ./generate_data $POINTS $DIMENSIONS test_data.txt $CLUSTERS
fi

# Run k-means
echo "Running k-means with $CLUSTERS clusters..."
./kmeans_cuda test_data.txt $POINTS $DIMENSIONS $CLUSTERS $MAX_ITERATIONS $THRESHOLD $OUTPUT_PREFIX

# Define visualization output file
VISUALIZATION_FILE="${OUTPUT_PREFIX}_${DIMENSIONS}d_visualization.png"

# Run visualization
echo "Generating visualization..."
if [[ "$REMOTE_ENV" == "true" || "$FORCE_SAVE" == "true" ]]; then
    # Always save the image in remote environment
    python3 visualize_kmeans.py \
        --data ${OUTPUT_PREFIX}_data.txt \
        --assignments ${OUTPUT_PREFIX}_assignments.txt \
        --centroids ${OUTPUT_PREFIX}_centroids.txt \
        --dimensions $DIMENSIONS \
        --output $VISUALIZATION_FILE
    
    echo "Visualization complete! Output saved to $VISUALIZATION_FILE"
    if [[ "$REMOTE_ENV" == "true" ]]; then
        echo "Use 'scp' or another tool to download the visualization from the remote server."
    fi
else
    # On local environment, show interactive visualization by default
    python3 visualize_kmeans.py \
        --data ${OUTPUT_PREFIX}_data.txt \
        --assignments ${OUTPUT_PREFIX}_assignments.txt \
        --centroids ${OUTPUT_PREFIX}_centroids.txt \
        --dimensions $DIMENSIONS
    
    echo "Interactive visualization complete!"
fi