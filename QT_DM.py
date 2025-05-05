##16:52 min

import math
import sys

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
    
    # Calculate square sum of differences
    square_sum = sum((coords2[i] - coords1[i]) ** 2 for i in range(len(coords1)))
    
    # Get the euclidean distance
    distance = math.sqrt(square_sum)
    return distance

def create_filtered_distance_matrix(point_data, quality_threshold):
    """
    Calculate distances between all points and filter out pairs that exceed
    the quality threshold.
    
    Returns:
    - distance_matrix: Dictionary with only distances <= quality_threshold
    - point_neighbors: Dictionary mapping each point to its potential neighbors
    """
    distance_matrix = {}
    point_neighbors = {i: set() for i in range(len(point_data))}
    
    for i in range(len(point_data)):
        for j in range(i + 1, len(point_data)):
            dist = euclidean_dist(point_data[i], point_data[j])
            
            # Only keep distances that are within the threshold
            if dist <= quality_threshold:
                distance_matrix[(i, j)] = dist
                # Add to the neighbors list for both points
                point_neighbors[i].add(j)
                point_neighbors[j].add(i)
    
    return distance_matrix, point_neighbors

def candidate_cluster(center_idx, point_data, distance_matrix, threshold, used_set, point_neighbors):
    """
    Forms a candidate cluster starting from center_idx.
    Only considers points that are neighbors (within threshold distance).
    """
    # Start with the center point
    cluster = [center_idx]
    
    # Make a list of all unused neighbors of the center point
    available_points = [i for i in point_neighbors[center_idx] if i not in used_set]
    
    # Continue adding points as long as possible
    while available_points:
        best_point = None
        best_max_dist = float("inf")
        
        # Try each available point
        for point_idx in available_points:
            # Check if this point is a neighbor of all points in the current cluster
            valid_point = True
            for existing_point in cluster:
                key = (existing_point, point_idx) if existing_point < point_idx else (point_idx, existing_point)
                if key not in distance_matrix:
                    valid_point = False
                    break
            
            if not valid_point:
                continue
                
            # Create a temporary cluster with this point added
            temp_cluster = cluster + [point_idx]
            
            # Calculate maximum distance in this temporary cluster
            max_dist = 0
            for i in range(len(temp_cluster)):
                for j in range(i + 1, len(temp_cluster)):
                    a, b = temp_cluster[i], temp_cluster[j]
                    key = (a, b) if a < b else (b, a)
                    dist = distance_matrix[key]
                    max_dist = max(max_dist, dist)
            
            # If this point keeps the cluster within threshold and has the smallest diameter
            if max_dist <= threshold and max_dist < best_max_dist:
                best_point = point_idx
                best_max_dist = max_dist
        
        # If we found a point to add, add it and remove from available points
        if best_point is not None:
            cluster.append(best_point)
            available_points.remove(best_point)
            
            # Update available points to only include neighbors of all cluster points
            new_available = []
            for point in available_points:
                valid = True
                for cluster_point in cluster:
                    key = (cluster_point, point) if cluster_point < point else (point, cluster_point)
                    if key not in distance_matrix:
                        valid = False
                        break
                if valid:
                    new_available.append(point)
            available_points = new_available
        else:
            # If no point can be added without exceeding threshold, we're done
            break
            
    return cluster

def best_cluster(point_data, distance_matrix, threshold, used_set, point_neighbors):
    """
    Tries to form a cluster from each unused point.
    Returns the best cluster found: the one with the most points (and tightest if tied).
    Only considers points that have potential neighbors.
    """
    best_cluster = []
    best_diameter = float("inf")    

    # Try to form a candidate cluster from every unused point
    for i in range(len(point_data)):
        if i in used_set:
            continue  # Skip if point has already been used
        
        # Skip points that don't have enough neighbors to form a cluster
        if len(point_neighbors[i] - used_set) == 0:
            continue
            
        # Form a candidate cluster starting from point i
        cluster = candidate_cluster(i, point_data, distance_matrix, threshold, used_set, point_neighbors)

        # Only consider clusters that have more than 1 point
        if len(cluster) < 2:
            continue

        # Compute the diameter of this cluster 
        max_dist = 0
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                p1, p2 = cluster[a], cluster[b]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                d = distance_matrix[key]
                max_dist = max(max_dist, d)

        # Compare this cluster with the current best one
        if (len(cluster) > len(best_cluster)) or (len(cluster) == len(best_cluster) and max_dist < best_diameter):
            best_cluster = cluster
            best_diameter = max_dist
    
    # Return after checking ALL potential clusters
    return best_cluster

def qt_clustering(point_data, distance_matrix, threshold, point_neighbors):
    """
    Main QT clustering algorithm.
    Repeatedly finds and removes the best cluster until all points are used.
    """
    used_points = set()
    clusters = []
    
    while len(used_points) < len(point_data):
        # Find the best cluster from remaining points
        best = best_cluster(point_data, distance_matrix, threshold, used_points, point_neighbors)
        
        # If no valid cluster with at least 2 points could be formed
        if len(best) < 2:
            # Add remaining points as single-point clusters
            for i in range(len(point_data)):
                if i not in used_points:
                    clusters.append([i])
                    used_points.add(i)
            break  # We're done
        else:
            # Add this cluster and update used points
            clusters.append(best)
            used_points.update(best)
    
    return clusters

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = "data/point1000.lst"

    print(f"Reading points from: {infile}")
    
    # Load data
    point_data = readpoints(infile)
    
    # Calculate ALL distances for finding the maximum distance
    all_distances = {}
    for i in range(len(point_data)):
        for j in range(i + 1, len(point_data)):
            dist = euclidean_dist(point_data[i], point_data[j])
            all_distances[(i, j)] = dist
    
    # Find maximum distance (diameter)
    max_distance = max(all_distances.values())
    
    # Set quality threshold to 30% of maximum distance
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance: {max_distance}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold}")
    
    # Create a filtered distance matrix that only contains distances <= threshold
    filtered_distance_matrix, point_neighbors = create_filtered_distance_matrix(
        point_data, quality_threshold
    )
    
    print(f"Full distance matrix size: {len(all_distances)}")
    print(f"Filtered distance matrix size: {len(filtered_distance_matrix)}")
    print(f"Reduction: {100 * (1 - len(filtered_distance_matrix) / len(all_distances)):.2f}%")
    
    # Run the QT clustering algorithm with the filtered matrix
    clusters = qt_clustering(point_data, filtered_distance_matrix, quality_threshold, point_neighbors)
    
    # Display results
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        # Convert indices to point names
        point_names = [point_data[idx][0] for idx in cluster]
        print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")