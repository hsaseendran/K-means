#!/usr/bin/env python3
"""
K-means Clustering Visualization
--------------------------------
This script visualizes k-means clustering results in 1D, 2D, or 3D.
It reads the data points and cluster assignments from files and produces
visualizations with distinct colors for each cluster.
"""
import numpy as np
import matplotlib
# Set non-interactive backend for headless environments
import os
if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from matplotlib.colors import TABLEAU_COLORS

def read_data(data_file, dimensions):
    """Read data points from file."""
    try:
        return np.loadtxt(data_file, usecols=range(dimensions))
    except Exception as e:
        print(f"Error reading data file: {e}")
        sys.exit(1)

def read_assignments(assignment_file):
    """Read cluster assignments from file."""
    try:
        return np.loadtxt(assignment_file, dtype=int)
    except Exception as e:
        print(f"Error reading assignment file: {e}")
        sys.exit(1)

def read_centroids(centroid_file, dimensions):
    """Read cluster centroids from file."""
    try:
        return np.loadtxt(centroid_file, usecols=range(dimensions))
    except Exception as e:
        print(f"Error reading centroid file: {e}")
        sys.exit(1)

def visualize_1d(data, assignments, centroids=None, output_file=None):
    """Create 1D visualization of clustered data."""
    plt.figure(figsize=(12, 6))
    
    # Get unique clusters and a list of colors
    unique_clusters = np.unique(assignments)
    colors = list(TABLEAU_COLORS.values())
    
    # Plot each cluster with a different color
    for i, cluster in enumerate(unique_clusters):
        cluster_points = data[assignments == cluster]
        color = colors[i % len(colors)]
        # Plot data points on x-axis, y-axis is just for visualization
        plt.scatter(cluster_points[:, 0], np.zeros_like(cluster_points[:, 0]) + i*0.1, 
                   color=color, alpha=0.6, label=f'Cluster {cluster}')
    
    # Plot centroids if provided
    if centroids is not None:
        plt.scatter(centroids[:, 0], np.zeros_like(centroids[:, 0]) - 0.5, 
                   color='black', marker='X', s=100, label='Centroids')
    
    plt.title('1D K-means Clustering Results')
    plt.xlabel('Value')
    plt.yticks([])  # Hide y-axis ticks
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"1D visualization saved to {output_file}")
    else:
        plt.show()

def visualize_2d(data, assignments, centroids=None, output_file=None):
    """Create 2D scatter plot of clustered data."""
    plt.figure(figsize=(10, 8))
    
    # Get unique clusters and a list of colors
    unique_clusters = np.unique(assignments)
    colors = list(TABLEAU_COLORS.values())
    
    # Plot each cluster with a different color
    for i, cluster in enumerate(unique_clusters):
        cluster_points = data[assignments == cluster]
        color = colors[i % len(colors)]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=color, alpha=0.6, label=f'Cluster {cluster}')
    
    # Plot centroids if provided
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   color='black', marker='X', s=100, label='Centroids')
    
    plt.title('2D K-means Clustering Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"2D visualization saved to {output_file}")
    else:
        plt.show()

def visualize_3d(data, assignments, centroids=None, output_file=None):
    """Create 3D scatter plot of clustered data."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique clusters and a list of colors
    unique_clusters = np.unique(assignments)
    colors = list(TABLEAU_COLORS.values())
    
    # Plot each cluster with a different color
    for i, cluster in enumerate(unique_clusters):
        cluster_points = data[assignments == cluster]
        color = colors[i % len(colors)]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                  color=color, alpha=0.6, label=f'Cluster {cluster}')
    
    # Plot centroids if provided
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                  color='black', marker='X', s=100, label='Centroids')
    
    ax.set_title('3D K-means Clustering Results')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.legend()
    
    # Set the same scale for all axes
    max_range = np.array([
        data[:, 0].max() - data[:, 0].min(),
        data[:, 1].max() - data[:, 1].min(),
        data[:, 2].max() - data[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (data[:, 0].max() + data[:, 0].min()) / 2
    mid_y = (data[:, 1].max() + data[:, 1].min()) / 2
    mid_z = (data[:, 2].max() + data[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add some good default viewing angles for 3D plot
    ax.view_init(elev=30, azim=45)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D visualization saved to {output_file}")
    else:
        plt.show()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize k-means clustering results')
    parser.add_argument('--data', required=True, help='Data points file')
    parser.add_argument('--assignments', required=True, help='Cluster assignments file')
    parser.add_argument('--centroids', help='Cluster centroids file (optional)')
    parser.add_argument('--dimensions', type=int, default=2, choices=[1, 2, 3],
                        help='Number of dimensions to visualize (1, 2, or 3)')
    parser.add_argument('--output', help='Output file path for the visualization (optional)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Always use output file when running headless
    if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
        if not args.output:
            args.output = f"kmeans_{args.dimensions}d_visualization.png"
            print(f"No display detected: Automatically saving output to {args.output}")
    
    # Read data and assignments
    data = read_data(args.data, args.dimensions)
    assignments = read_assignments(args.assignments)
    
    # Ensure data and assignments have the same length
    if len(data) != len(assignments):
        print(f"Error: Data ({len(data)} points) and assignments ({len(assignments)} points) have different lengths")
        sys.exit(1)
    
    # Read centroids if provided
    centroids = None
    if args.centroids:
        centroids = read_centroids(args.centroids, args.dimensions)
    
    # Create appropriate visualization based on dimensions
    if args.dimensions == 1:
        visualize_1d(data, assignments, centroids, args.output)
    elif args.dimensions == 2:
        visualize_2d(data, assignments, centroids, args.output)
    elif args.dimensions == 3:
        visualize_3d(data, assignments, centroids, args.output)

if __name__ == "__main__":
    main()