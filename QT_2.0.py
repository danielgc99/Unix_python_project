import math
import sys
import time
from collections import defaultdict

def readpoints(infile):
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

def euclidean_dist(pointA, pointB):
    """Takes two points as input and calculates the euclidean distance between them"""
    coords1 = pointA[1:]
    coords2 = pointB[1:]
    
    # Verify both points have the same number of dimensions
    if len(coords1) != len(coords2):
        raise ValueError("Points must have the same number of dimensions")
    
    # Calculate square sum of differences (using faster approach)
    square_sum = sum((coords2[i] - coords1[i]) ** 2 for i in range(len(coords1)))
    
    # Get the euclidean distance
    return math.sqrt(square_sum)

def calculate_distance_matrix(point_data):
    """Pre-calculate all pairwise distances between points"""
    distance_matrix = {}
    for i in range(len(point_data)):
        for j in range(i + 1, len(point_data)):
            dist = euclidean_dist(point_data[i], point_data[j])
            distance_matrix[(i, j)] = dist
    return distance_matrix

def get_distance(i, j, distance_matrix):
    """Get distance between two points from the distance matrix"""
    if i == j:
        return 0
    key = (i, j) if i < j else (j, i)
    return distance_matrix[key]

def candidate_cluster_optimized(center_idx, point_data, distance_matrix, threshold, used_set):
    """
    Optimized version of candidate cluster formation.
    Uses a greedy approach with pre-sorting of points by distance to center.
    """
    # Start with the center point
    cluster = [center_idx]
    
    # Pre-calculate distances to center for all unused points
    distances_to_center = []
    for i in range(len(point_data)):
        if i != center_idx and i not in used_set:
            dist = get_distance(center_idx, i, distance_matrix)
            distances_to_center.append((i, dist))
    
    # Sort points by distance to center (closest first)
    distances_to_center.sort(key=lambda x: x[1])
    
    # For each candidate point, track the maximum distance to any point in cluster
    max_distances = defaultdict(float)
    
    # Try adding points in order of increasing distance to center
    for idx, _ in distances_to_center:
        # Calculate max distance from this point to any point in cluster
        max_dist_to_cluster = 0
        for point in cluster:
            dist = get_distance(idx, point, distance_matrix)
            max_dist_to_cluster = max(max_dist_to_cluster, dist)
            
        # If this point would keep the cluster within threshold
        if max_dist_to_cluster <= threshold:
            # Add it to cluster
            cluster.append(idx)
            
            # Update max distances for remaining candidates
            for other_idx, _ in distances_to_center:
                if other_idx not in cluster and other_idx not in used_set:
                    dist = get_distance(idx, other_idx, distance_matrix)
                    max_distances[other_idx] = max(max_distances[other_idx], dist)
    
    return cluster

def best_cluster_optimized(point_data, distance_matrix, threshold, used_set):
    """
    Optimized version of best cluster finding.
    Uses cached calculations where possible.
    """
    best_cluster = []
    best_size = 0
    best_diameter = float("inf")
    
    # Track points we've already considered as centers to avoid redundant work
    tried_centers = set()
    
    # Start with unused points
    for i in range(len(point_data)):
        if i in used_set or i in tried_centers:
            continue
            
        # Form candidate cluster
        cluster = candidate_cluster_optimized(i, point_data, distance_matrix, threshold, used_set)
        tried_centers.add(i)
        
        # Only consider clusters with at least 2 points
        if len(cluster) < 2:
            continue
            
        # Calculate diameter only once
        diameter = 0
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                dist = get_distance(cluster[a], cluster[b], distance_matrix)
                diameter = max(diameter, dist)
        
        # Update best cluster if this one is better
        if len(cluster) > best_size or (len(cluster) == best_size and diameter < best_diameter):
            best_cluster = cluster
            best_size = len(cluster)
            best_diameter = diameter
            
            # Optimization: If we find a cluster with many points, mark all its points as tried centers
            # since any cluster centered at those points would be similar or worse
            if len(cluster) > 3:
                tried_centers.update(cluster)
    
    return best_cluster

def qt_clustering_optimized(point_data, distance_matrix, threshold):
    """
    Optimized QT clustering algorithm.
    """
    used_points = set()
    clusters = []
    
    # Process until all points are used
    while len(used_points) < len(point_data):
        # Find best cluster from remaining points
        start_time = time.time()
        best = best_cluster_optimized(point_data, distance_matrix, threshold, used_points)
        end_time = time.time()
        
        # Debug timing
        if len(clusters) < 5:  # Only print timing for first few clusters
            print(f"Finding cluster {len(clusters)+1} took {end_time - start_time:.4f} seconds")
        
        # If no valid multi-point cluster found
        if len(best) < 2:
            # Add remaining points as singletons
            for i in range(len(point_data)):
                if i not in used_points:
                    clusters.append([i])
                    used_points.add(i)
            break
        else:
            # Add cluster and update used points
            clusters.append(best)
            used_points.update(best)
            print(f"Found cluster with {len(best)} points. Total points used: {len(used_points)}/{len(point_data)}")
    
    return clusters

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = "data/point1000.lst"

    print(f"Reading points from: {infile}")
    
    # Start timing
    total_start = time.time()
    
    # Load data
    point_data = readpoints(infile)
    print(f"Loaded {len(point_data)} points")
    
    # Calculate distances between all points
    start_time = time.time()
    distance_matrix = calculate_distance_matrix(point_data)
    end_time = time.time()
    print(f"Distance matrix calculation took {end_time - start_time:.4f} seconds")
    
    # Find maximum distance (diameter)
    max_distance = max(distance_matrix.values())
    
    # Set quality threshold to 30% of maximum distance
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance: {max_distance}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold}")
    
    # Run the optimized QT clustering algorithm
    start_time = time.time()
    clusters = qt_clustering_optimized(point_data, distance_matrix, quality_threshold)
    end_time = time.time()
    print(f"QT clustering took {end_time - start_time:.4f} seconds")
    
    # Display results
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        # Convert indices to point names
        point_names = [point_data[idx][0] for idx in cluster]
        print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")
    
    # Print total time
    total_end = time.time()
    print(f"\nTotal execution time: {total_end - total_start:.4f} seconds")