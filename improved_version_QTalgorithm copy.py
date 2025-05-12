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

def candidate_cluster(center_idx, point_data, distance_matrix, threshold, used_set):
    """
    Forms a candidate cluster starting from center_idx.
    Only includes points that are not in used_set.
    Adds points that minimize cluster diameter while staying within threshold.
    """
    # Start with the center point
    cluster = [center_idx]
    
    # Make a list of all unused points
    available_points = [i for i in range(len(point_data)) if i != center_idx and i not in used_set]
    
    # Continue adding points as long as possible
    while available_points:
        best_point = None
        best_max_dist = float("inf")
        
        # Try each available point
        for point_idx in available_points:
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
        else:
            # If no point can be added without exceeding threshold, we're done
            break
            
    return cluster

def best_cluster(point_data, distance_matrix, threshold, used_set):
    """
    Tries to form a cluster from each unused point.
    Returns the best cluster found: the one with the most points (and tightest if tied).
    """
    best_cluster = []
    best_diameter = float("inf")    

    # Try to form a candidate cluster from every unused point
    for i in range(len(point_data)):
        if i in used_set:
            continue  # Skip if point has already been used
            
        # Form a candidate cluster starting from point i
        cluster = candidate_cluster(i, point_data, distance_matrix, threshold, used_set)

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

def qt_clustering(point_data, distance_matrix, threshold):
    """
    Main QT clustering algorithm.
    Repeatedly finds and removes the best cluster until all points are used.
    """
    used_points = set()
    clusters = []
    
    while len(used_points) < len(point_data):
        # Find the best cluster from remaining points
        best = best_cluster(point_data, distance_matrix, threshold, used_points)
        
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
    
    # Calculate distances between all points
    distance_matrix = {}
    for i in range(len(point_data)):
        for j in range(i + 1, len(point_data)):
            dist = euclidean_dist(point_data[i], point_data[j])
            distance_matrix[(i, j)] = dist
    
    # Find maximum distance (diameter)
    max_distance = max(distance_matrix.values())
    
    # Set quality threshold to 30% of maximum distance
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance: {max_distance}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold}")
    
    # Run the QT clustering algorithm
    clusters = qt_clustering(point_data, distance_matrix, quality_threshold)
    
    # Display results
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        # Convert indices to point names
        point_names = [point_data[idx][0] for idx in cluster]
        print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")