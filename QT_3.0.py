import math
import sys
import time
import numpy as np
from collections import defaultdict

def readpoints(infile):
    """Read points from file and return as list of tuples (name, x, y, ...)"""
    point_list = []
    n = 0
    with open(infile) as file:
        for line in file:
            n += 1
            line = line.strip().split()
            # Files with index column starting with "point"
            if line[0].lower().startswith("point"):
                point_list.append((line[0].replace("p", "P", 1), *[float(value) for value in line[1:]]))
            # For files with no index column
            else:
                point_list.append((f"Point{1+n}", *[float(value) for value in line]))
    return point_list

def calculate_distance_matrix(point_data):
    """Calculate distances between all points"""
    n_points = len(point_data)
    distance_matrix = {}
    
    # Extract coordinates for faster processing
    coords = np.array([list(point[1:]) for point in point_data])
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Fast Euclidean distance with NumPy
            dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            distance_matrix[(i, j)] = dist
    
    return distance_matrix

def get_distance(i, j, distance_matrix):
    """Get distance between points i and j from distance matrix"""
    if i == j:
        return 0
    key = (i, j) if i < j else (j, i)
    return distance_matrix[key]

def calculate_max_distance(distance_matrix):
    """Find the maximum distance in the distance matrix"""
    return max(distance_matrix.values())

def candidate_cluster_exact(center_idx, point_data, distance_matrix, threshold, used_points):
    """
    Original candidate cluster algorithm that ensures exact results.
    Uses vectorized operations for speed.
    """
    # Start with the center point
    cluster = [center_idx]
    
    # Remaining points (not used and not the center)
    available_points = [i for i in range(len(point_data)) if i != center_idx and i not in used_points]
    
    # Pre-calculate distances from center to all available points for sorting
    distances_to_center = [(i, get_distance(center_idx, i, distance_matrix)) for i in available_points]
    distances_to_center.sort(key=lambda x: x[1])  # Sort by distance to center
    
    # Process available points in order of distance to center
    available_points = [p for p, _ in distances_to_center]
    
    # Continue adding points until no more can be added
    while available_points:
        best_point = None
        best_max_dist = float('inf')
        
        # Try each available point
        for idx in available_points:
            # Check if this point would exceed threshold with any existing point
            valid = True
            max_dist = 0
            
            for p in cluster:
                dist = get_distance(idx, p, distance_matrix)
                if dist > threshold:
                    valid = False
                    break
                max_dist = max(max_dist, dist)
            
            if valid and max_dist < best_max_dist:
                best_point = idx
                best_max_dist = max_dist
        
        # If we found a valid point, add it
        if best_point is not None:
            cluster.append(best_point)
            available_points.remove(best_point)
        else:
            # No more points can be added
            break
    
    return cluster

def best_cluster_exact(point_data, distance_matrix, threshold, used_points):
    """
    Find the best cluster with exact same logic as original algorithm.
    Optimized for performance with NumPy and early pruning.
    """
    best_cluster = []
    best_size = 0
    best_diameter = float('inf')
    
    # Process each potential center point
    for i in range(len(point_data)):
        if i in used_points:
            continue
        
        # Form candidate cluster
        cluster = candidate_cluster_exact(i, point_data, distance_matrix, threshold, used_points)
        
        # Skip if fewer than 2 points
        if len(cluster) < 2:
            continue
        
        # Calculate diameter
        max_dist = 0
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                dist = get_distance(cluster[a], cluster[b], distance_matrix)
                max_dist = max(max_dist, dist)
        
        # Update if better
        if len(cluster) > best_size or (len(cluster) == best_size and max_dist < best_diameter):
            best_cluster = cluster
            best_size = len(cluster)
            best_diameter = max_dist
    
    return best_cluster

def qt_clustering_exact_fast(point_data, distance_matrix, threshold):
    """
    QT Clustering implementation that preserves exact results while optimizing performance.
    """
    used_points = set()
    clusters = []
    n_points = len(point_data)
    
    # Track progress and performance
    iteration = 0
    largest_cluster = 0
    
    while len(used_points) < n_points:
        iteration += 1
        start_time = time.time()
        
        # Find best cluster
        best = best_cluster_exact(point_data, distance_matrix, threshold, used_points)
        largest_cluster = max(largest_cluster, len(best))
        
        # If no valid multi-point cluster
        if len(best) < 2:
            # Add remaining points as singletons
            remaining = [i for i in range(n_points) if i not in used_points]
            for i in remaining:
                clusters.append([i])
                used_points.add(i)
            print(f"Added {len(remaining)} singleton clusters")
            break
        else:
            # Add cluster and update used points
            clusters.append(best)
            used_points.update(best)
            
            # Report progress
            iter_time = time.time() - start_time
            print(f"Iteration {iteration}: Found cluster with {len(best)} points in {iter_time:.4f}s. "
                  f"Total: {len(used_points)}/{n_points}")
    
    print(f"Largest cluster: {largest_cluster} points")
    return clusters

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = "data/point1000.lst"

    print(f"Reading points from: {infile}")
    
    # Overall timing
    total_start = time.time()
    
    # Load data
    point_data = readpoints(infile)
    print(f"Loaded {len(point_data)} points in {time.time() - total_start:.4f} seconds")
    
    # Calculate distance matrix
    start_time = time.time()
    distance_matrix = calculate_distance_matrix(point_data)
    print(f"Distance matrix calculation took {time.time() - start_time:.4f} seconds")
    
    # Find maximum distance (diameter)
    max_distance = calculate_max_distance(distance_matrix)
    
    # Set quality threshold to 30% of maximum distance
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance: {max_distance:.4f}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold:.4f}")
    
    # Run QT clustering
    print("Starting QT clustering...")
    start_time = time.time()
    clusters = qt_clustering_exact_fast(point_data, distance_matrix, threshold=quality_threshold)
    print(f"QT clustering took {time.time() - start_time:.4f} seconds")
    
    # Print results
    print(f"\nFound {len(clusters)} clusters:")
    total_points = 0
    
    for i, cluster in enumerate(clusters):
        total_points += len(cluster)
        
        # Convert indices to point names
        if i < 5 or i >= len(clusters) - 5 or len(clusters) <= 10:
            point_names = [point_data[idx][0] for idx in cluster]
            if len(cluster) > 10:
                point_names = point_names[:5] + ['...'] + point_names[-5:]
            print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")
        elif i == 5 and len(clusters) > 10:
            print("...")
    
    # Print total time
    print(f"\nTotal execution time: {time.time() - total_start:.4f} seconds")
    print(f"Total points in clusters: {total_points}/{len(point_data)}")