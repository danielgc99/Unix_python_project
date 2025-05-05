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

def generate_all_candidate_clusters(point_data, distance_matrix, threshold):
    """
    Generates all possible candidate clusters for every point as a center.
    Returns a list of lists where each inner list contains the indices of points in a cluster.
    The first point in each inner list is always the center point.
    """
    all_candidates = []
    
    # For each potential center point
    for center_idx in range(len(point_data)):
        # Start with the center point
        cluster = [center_idx]
        
        # Make a list of all other points
        available_points = [i for i in range(len(point_data)) if i != center_idx]
        
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
                
        # Add this candidate cluster to our list of all candidates
        all_candidates.append(cluster)

    return all_candidates



def find_best_cluster(candidate_clusters, distance_matrix):
    """
    Finds the best cluster from the list of candidate clusters.
    The best cluster is the one with the most points.
    In case of a tie, the function will return the cluster with the minimum diameter.
    """
    if not candidate_clusters:
        return []
    
    # Find the maximum number of points in any cluster
    max_points = max(len(cluster) for cluster in candidate_clusters)
    
    # Get all clusters with the maximum number of points
    largest_clusters = [cluster for cluster in candidate_clusters if len(cluster) == max_points]
    
    # If there's only one largest cluster, return it
    if len(largest_clusters) == 1:
        return largest_clusters[0]
    
    # Otherwise, find the one with minimum diameter
    best_cluster = None
    min_diameter = float("inf")
    
    for cluster in largest_clusters:
        # Calculate the maximum distance (diameter) within this cluster
        cluster_diameter = 0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a, b = cluster[i], cluster[j]
                key = (a, b) if a < b else (b, a)
                dist = distance_matrix[key]
                cluster_diameter = max(cluster_diameter, dist)
        
        # Update best cluster if this one has a smaller diameter
        if cluster_diameter < min_diameter:
            min_diameter = cluster_diameter
            best_cluster = cluster
    
    return best_cluster

def update_candidate_clusters(candidate_clusters, best_cluster, distance_matrix, threshold, used_points):
    """
    Updates the list of candidate clusters by:
    1. Removing the best cluster itself
    2. Removing clusters whose center point is in the best cluster
    3. Regenerating clusters for center points that are not in the best cluster
       but whose original clusters contained points from the best cluster
    
    Returns the updated list of candidate clusters
    """
    # Create a set of points that are in the best cluster for faster lookups
    best_cluster_points = set(best_cluster)
    
    # Add best cluster points to the overall used points set
    used_points.update(best_cluster_points)
    
    # Create a list to store updated candidate clusters
    updated_candidates = []
    
    # Get total number of points in the data (assuming points are indexed from 0 to n-1)
    # Instead of trying to calculate this, we'll pass it from the calling function
    total_points = max(max(cluster) for cluster in candidate_clusters) + 1
    
    # Get all available points (points not in any best cluster yet)
    available_points = [i for i in range(total_points) if i not in used_points]
    
    # For each original cluster
    for cluster in candidate_clusters:
        center_point = cluster[0]
        
        # Skip this cluster if its center is in the best cluster
        if center_point in best_cluster_points:
            continue
        
        # Check if this center point is still available
        if center_point not in used_points:
            # Regenerate the cluster for this center using only available points
            regenerated_points = available_points + [center_point]  # Center is always included
            
            # Start with the center point
            new_cluster = [center_point]
            points_to_try = [p for p in regenerated_points if p != center_point]
            
            # Continue adding points as long as possible
            while points_to_try:
                best_point = None
                best_max_dist = float("inf")
                
                # Try each available point
                for point_idx in points_to_try:
                    # Create a temporary cluster with this point added
                    temp_cluster = new_cluster + [point_idx]
                    
                    # Calculate maximum distance in this temporary cluster
                    max_dist = 0
                    valid_cluster = True
                    
                    for i in range(len(temp_cluster)):
                        for j in range(i + 1, len(temp_cluster)):
                            a, b = temp_cluster[i], temp_cluster[j]
                            # Ensure we're using the correct key format for the distance matrix
                            key = (min(a, b), max(a, b))
                            
                            # Check if the key exists in the distance matrix
                            if key not in distance_matrix:
                                valid_cluster = False
                                break
                                
                            dist = distance_matrix[key]
                            max_dist = max(max_dist, dist)
                            
                        if not valid_cluster:
                            break
                    
                    # If this point keeps the cluster within threshold and has the smallest diameter
                    if valid_cluster and max_dist <= threshold and max_dist < best_max_dist:
                        best_point = point_idx
                        best_max_dist = max_dist
                
                # If we found a point to add, add it and remove from points to try
                if best_point is not None:
                    new_cluster.append(best_point)
                    points_to_try.remove(best_point)
                else:
                    # If no point can be added without exceeding threshold, we're done
                    break
            
            # Only add if the new cluster has at least 2 points
            if len(new_cluster) >= 2:
                updated_candidates.append(new_cluster)
    
    return updated_candidates

def optimized_qt_clustering(point_data, distance_matrix, threshold):
    """
    Optimized QT clustering algorithm that avoids recalculating all candidate clusters.
    Pre-computes all possible candidate clusters and then repeatedly finds and removes the best cluster.
    """
    # Generate all possible candidate clusters
    print("Generating initial candidate clusters...")
    candidate_clusters = generate_all_candidate_clusters(point_data, distance_matrix, threshold)
    print(f"Generated {len(candidate_clusters)} initial candidate clusters")
    
    # Keep track of used points and final clusters
    used_points = set()
    final_clusters = []
    iteration = 0
    
    # Continue until all points are used or no more valid clusters can be formed
    while candidate_clusters and len(used_points) < len(point_data):
        iteration += 1
        print(f"\nIteration {iteration}:")
        print(f"- Points used so far: {len(used_points)}/{len(point_data)}")
        print(f"- Candidate clusters remaining: {len(candidate_clusters)}")
        
        # Find the best cluster
        best = find_best_cluster(candidate_clusters, distance_matrix)
        
        if not best:
            print("- No valid cluster found, breaking...")
            break
            
        # Add this cluster to final results
        final_clusters.append(best)
        
        print(f"- Selected best cluster with {len(best)} points: {[point_data[idx][0] for idx in best]}")
        
        # Store current used points count
        prev_used_count = len(used_points)
        
        # Update the candidate clusters with available points
        candidate_clusters = update_candidate_clusters(
            candidate_clusters,
            best,
            distance_matrix,
            threshold,
            used_points
        )
        
        print(f"- Added {len(used_points) - prev_used_count} new points to used points")
        print(f"- Regenerated {len(candidate_clusters)} candidate clusters")
    
    # Add any remaining points as single-point clusters
    remaining_points = [i for i in range(len(point_data)) if i not in used_points]
    if remaining_points:
        print(f"\nAdding {len(remaining_points)} remaining points as single-point clusters")
        for i in remaining_points:
            final_clusters.append([i])
    
    return final_clusters

# Main execution
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # Parse command line arguments
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = "data/point100.lst"

    print(f"Reading points from: {infile}")
    
    # Load data
    point_data = readpoints(infile)
    print(f"Loaded {len(point_data)} points")
    
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
    
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    # Run the optimized QT clustering algorithm
    clustering_start_time = time.time()
    clusters = optimized_qt_clustering(point_data, distance_matrix, quality_threshold)
    clustering_time = time.time() - clustering_start_time
    
    # Display results
    print(f"\nFound {len(clusters)} clusters in {clustering_time:.2f} seconds:")
    for i, cluster in enumerate(clusters):
        # Convert indices to point names
        point_names = [point_data[idx][0] for idx in cluster]
        print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")