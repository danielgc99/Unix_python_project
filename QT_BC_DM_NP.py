import math
import sys
import time

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
    the quality threshold. Also identify closest point pairs to merge as a pre-clustering step.
    
    Returns:
    - distance_matrix: Dictionary with only distances <= quality_threshold
    - point_neighbors: Dictionary mapping each point to its potential neighbors
    - point_pairs: Dictionary mapping merged point indices to their constituent points
    """
    print("Creating filtered distance matrix...")
    start_time = time.time()
    
    distance_matrix = {}
    point_neighbors = {i: set() for i in range(len(point_data))}
    
    # Track closest neighbors for each point
    closest_neighbor = {}  # Maps point index to (neighbor_index, distance)
    
    total_pairs = len(point_data) * (len(point_data) - 1) // 2
    included_pairs = 0
    
    for i in range(len(point_data)):
        if i % 100 == 0 and i > 0:
            progress = (i * (len(point_data) - 1) - (i * (i - 1) // 2)) / total_pairs * 100
            print(f"Processing point {i}/{len(point_data)} ({progress:.1f}% complete)")
        
        for j in range(i + 1, len(point_data)):
            dist = euclidean_dist(point_data[i], point_data[j])
            
            # Only keep distances that are within the threshold
            if dist <= quality_threshold:
                distance_matrix[(i, j)] = dist
                # Add to the neighbors list for both points
                point_neighbors[i].add(j)
                point_neighbors[j].add(i)
                included_pairs += 1
                
                # Track closest neighbor for each point
                if i not in closest_neighbor or dist < closest_neighbor[i][1]:
                    closest_neighbor[i] = (j, dist)
                if j not in closest_neighbor or dist < closest_neighbor[j][1]:
                    closest_neighbor[j] = (i, dist)
    
    # Identify mutually closest pairs (bidirectional closest neighbors)
    mutual_closest_pairs = []
    used_points = set()
    
    # First pass: find mutual closest pairs
    for i in range(len(point_data)):
        if i in used_points or i not in closest_neighbor:
            continue
            
        neighbor, _ = closest_neighbor[i]
        
        # Check if this is a mutual closest relationship
        if neighbor in closest_neighbor and closest_neighbor[neighbor][0] == i:
            mutual_closest_pairs.append((i, neighbor))
            used_points.add(i)
            used_points.add(neighbor)
    
    # Create point_pairs mapping
    point_pairs = {}
    for idx, (i, j) in enumerate(mutual_closest_pairs):
        # Create a virtual merged point index starting after the last real point
        merged_idx = len(point_data) + idx
        point_pairs[merged_idx] = (i, j)
        
        # Update the neighbor sets for the merged points
        # Union of neighbors excluding the points being merged
        merged_neighbors = (point_neighbors[i] | point_neighbors[j]) - {i, j}
        point_neighbors[merged_idx] = merged_neighbors
        
        # Update the neighbors of other points to include the merged point
        for neighbor in merged_neighbors:
            point_neighbors[neighbor].add(merged_idx)
            
            # Remove the original points from the neighbor's set
            if i in point_neighbors[neighbor]:
                point_neighbors[neighbor].remove(i)
            if j in point_neighbors[neighbor]:
                point_neighbors[neighbor].remove(j)
        
        # Update distance matrix with distances to the merged point
        for neighbor in merged_neighbors:
            # Use the maximum distance as the conservative estimate
            dist_i_neighbor = distance_matrix.get((min(i, neighbor), max(i, neighbor)), float('inf'))
            dist_j_neighbor = distance_matrix.get((min(j, neighbor), max(j, neighbor)), float('inf'))
            merged_dist = max(dist_i_neighbor, dist_j_neighbor)
            
            # Add to distance matrix
            distance_matrix[(min(merged_idx, neighbor), max(merged_idx, neighbor))] = merged_dist
    
    elapsed_time = time.time() - start_time
    print(f"Filtered distance matrix created in {elapsed_time:.2f} seconds")
    print(f"Total pairs considered: {total_pairs}")
    print(f"Pairs included in filtered matrix: {included_pairs} ({included_pairs/total_pairs*100:.2f}%)")
    print(f"Reduction: {100 * (1 - included_pairs / total_pairs):.2f}%")
    print(f"Identified {len(mutual_closest_pairs)} mutual closest point pairs for pre-clustering")
    
    return distance_matrix, point_neighbors, point_pairs

def generate_all_candidate_clusters(point_data, distance_matrix, threshold, point_neighbors, point_pairs, max_points_per_cluster=None):
    """
    Generates all possible candidate clusters for every point as a center.
    Uses pre-clustered point pairs to reduce computation.
    """
    print("Generating all candidate clusters...")
    start_time = time.time()
    
    all_candidates = []
    # Include both original points and merged point pairs as potential centers
    total_centers = len(point_data) + len(point_pairs)
    
    # For each potential center point (including merged pairs)
    for center_idx in range(total_centers):
        if center_idx % 100 == 0 and center_idx > 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / center_idx) * (total_centers - center_idx)
            print(f"Processing center {center_idx}/{total_centers} ({center_idx/total_centers*100:.1f}%), " +
                  f"elapsed: {elapsed:.2f}s, est. remaining: {remaining:.2f}s")
        
        # Skip points with no neighbors
        if center_idx not in point_neighbors or len(point_neighbors[center_idx]) == 0:
            # If this is a merged pair, add both constituent points
            if center_idx in point_pairs:
                all_candidates.append([center_idx])  # The pair will be expanded later
            else:
                all_candidates.append([center_idx])  # Single-point cluster
            continue
        
        # Start with the center point
        cluster = [center_idx]
        
        # Make a list of all neighbors of the center
        available_points = list(point_neighbors[center_idx])
        
        # Optional limit on cluster size
        effective_max = max_points_per_cluster
        if max_points_per_cluster and center_idx in point_pairs:
            # If this is a merged pair, adjust the max to account for the two points
            effective_max = max_points_per_cluster - 1
        
        if effective_max and len(available_points) > effective_max - 1:
            available_points = available_points[:effective_max-1]
        
        # Continue adding points as long as possible
        while available_points:
            best_point = None
            best_max_dist = float("inf")
            
            # Try each available point
            for point_idx in available_points:
                # Check if this point is a neighbor of all points in the current cluster
                valid_point = True
                
                for existing_idx in cluster:
                    if existing_idx == center_idx:
                        continue  # Skip the center, we already checked this
                        
                    key = (min(existing_idx, point_idx), max(existing_idx, point_idx))
                    if key not in distance_matrix:
                        valid_point = False
                        break
                
                if not valid_point:
                    continue
                    
                # Calculate maximum distance if we add this point
                current_max_dist = 0
                for existing_idx in cluster:
                    key = (min(existing_idx, point_idx), max(existing_idx, point_idx))
                    dist = distance_matrix[key]
                    current_max_dist = max(current_max_dist, dist)
                
                # If this point keeps the cluster within threshold and has the smallest diameter
                if current_max_dist <= threshold and current_max_dist < best_max_dist:
                    best_point = point_idx
                    best_max_dist = current_max_dist
            
            # If we found a point to add, add it and remove from available points
            if best_point is not None:
                cluster.append(best_point)
                available_points.remove(best_point)
                
                # Update available points to only include common neighbors
                new_available = []
                for point in available_points:
                    # The point must be a neighbor of all cluster points
                    valid = True
                    for cluster_point in cluster:
                        key = (min(cluster_point, point), max(cluster_point, point))
                        if key not in distance_matrix:
                            valid = False
                            break
                    if valid:
                        new_available.append(point)
                available_points = new_available
                
                # Check if we've reached the maximum cluster size
                if effective_max and len(cluster) >= effective_max:
                    break
            else:
                # If no point can be added without exceeding threshold, we're done
                break
                
        # Add this candidate cluster to our list of all candidates
        all_candidates.append(cluster)
    
    elapsed_time = time.time() - start_time
    
    # Report final statistics
    cluster_sizes = [len(c) for c in all_candidates]
    avg_cluster_size = sum(cluster_sizes) / len(all_candidates)
    max_cluster_size = max(cluster_sizes)
    clusters_with_multiple_points = sum(1 for c in all_candidates if len(c) > 1)
    
    print(f"Generated {len(all_candidates)} candidate clusters in {elapsed_time:.2f} seconds")
    print(f"Clusters with multiple points: {clusters_with_multiple_points} ({clusters_with_multiple_points/len(all_candidates)*100:.2f}%)")
    print(f"Average cluster size: {avg_cluster_size:.2f}, Maximum: {max_cluster_size}")
    
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
                key = (min(a, b), max(a, b))
                dist = distance_matrix[key]
                cluster_diameter = max(cluster_diameter, dist)
        
        # Update best cluster if this one has a smaller diameter
        if cluster_diameter < min_diameter:
            min_diameter = cluster_diameter
            best_cluster = cluster
    
    return best_cluster

def update_candidate_clusters(candidate_clusters, best_cluster, distance_matrix, threshold, used_points, point_neighbors, point_pairs):
    """
    Updates the list of candidate clusters, accounting for pre-clustered point pairs.
    """
    # Create a set of points that are in the best cluster for faster lookups
    best_cluster_points = set(best_cluster)
    
    # Add best cluster points to the overall used points set
    used_points.update(best_cluster_points)
    
    # Create a list to store updated candidate clusters
    updated_candidates = []
    
    # For each original cluster
    for cluster in candidate_clusters:
        center_point = cluster[0]
        
        # Skip this cluster if its center is in the best cluster
        if center_point in best_cluster_points:
            continue
        
        # If this center is a merged pair and one of its points is in best cluster, skip it
        if center_point in point_pairs:
            i, j = point_pairs[center_point]
            if i in best_cluster_points or j in best_cluster_points:
                continue
        
        # Check if this center point is still available
        if center_point not in used_points:
            # Start with the center point
            new_cluster = [center_point]
            
            # Get available neighbor points (not used and neighbors of center)
            available_neighbors = [p for p in point_neighbors[center_point] if p not in used_points]
            
            # Continue adding points as long as possible
            while available_neighbors:
                best_point = None
                best_max_dist = float("inf")
                
                # Try each available neighbor
                for point_idx in available_neighbors:
                    # Skip if this is a merged pair and one of its points is already used
                    if point_idx in point_pairs:
                        i, j = point_pairs[point_idx]
                        if i in used_points or j in used_points:
                            continue
                    
                    # Check if this point is a neighbor of all current cluster points
                    valid_point = True
                    current_max_dist = 0
                    
                    for existing_idx in new_cluster:
                        key = (min(existing_idx, point_idx), max(existing_idx, point_idx))
                        
                        # Check if the key exists in the distance matrix
                        if key not in distance_matrix:
                            valid_point = False
                            break
                            
                        dist = distance_matrix[key]
                        current_max_dist = max(current_max_dist, dist)
                    
                    # If this point keeps the cluster within threshold and has the smallest diameter
                    if valid_point and current_max_dist <= threshold and current_max_dist < best_max_dist:
                        best_point = point_idx
                        best_max_dist = current_max_dist
                
                # If we found a point to add, add it and remove from points to try
                if best_point is not None:
                    new_cluster.append(best_point)
                    available_neighbors.remove(best_point)
                    
                    # Update available points to only include common neighbors
                    new_available = []
                    for point in available_neighbors:
                        # The point must be a neighbor of all cluster points
                        valid = True
                        for cluster_point in new_cluster:
                            key = (min(cluster_point, point), max(cluster_point, point))
                            if key not in distance_matrix:
                                valid = False
                                break
                        if valid:
                            new_available.append(point)
                    available_neighbors = new_available
                else:
                    # If no point can be added without exceeding threshold, we're done
                    break
            
            # Only add if the new cluster has at least 2 points
            if len(new_cluster) >= 2:
                updated_candidates.append(new_cluster)
    
    return updated_candidates

def expand_cluster_with_pairs(cluster, point_pairs):
    """
    Expands a cluster by replacing any merged point indices with their constituent points.
    """
    expanded = []
    for idx in cluster:
        if idx in point_pairs:
            # Add the constituent points of this merged pair
            i, j = point_pairs[idx]
            expanded.append(i)
            expanded.append(j)
        else:
            # Add the original point
            expanded.append(idx)
    return expanded

def post_process_clusters(clusters, point_data, point_pairs):
    """
    Ensures all original points are correctly represented in the final clusters.
    """
    # Create a mapping from original points to the clusters they belong to
    point_to_cluster = {}
    
    for cluster_idx, cluster in enumerate(clusters):
        for point_idx in cluster:
            # If this is a real point (not a merged one)
            if point_idx < len(point_data):
                point_to_cluster[point_idx] = cluster_idx
    
    # Check if any points from point_pairs are missing and add them
    for merged_idx, (i, j) in point_pairs.items():
        # If one point is in a cluster but the other isn't, add the missing one
        if i in point_to_cluster and j not in point_to_cluster:
            clusters[point_to_cluster[i]].append(j)
        elif j in point_to_cluster and i not in point_to_cluster:
            clusters[point_to_cluster[j]].append(i)
        # If neither is in a cluster, create a new one
        elif i not in point_to_cluster and j not in point_to_cluster:
            clusters.append([i, j])
    
    return clusters

def optimized_qt_clustering(point_data, distance_matrix, threshold, point_neighbors, point_pairs, max_points_per_cluster=None):
    """
    Optimized QT clustering algorithm that:
    1. Uses a filtered distance matrix with only points within threshold
    2. Keeps track of neighboring points for each point
    3. Pre-clusters closest point pairs to reduce computation
    4. Only considers valid neighbors when building clusters
    5. Has an optional limit on cluster size
    """
    print("Starting optimized QT clustering algorithm...")
    # Generate all possible candidate clusters
    candidate_clusters = generate_all_candidate_clusters(
        point_data, 
        distance_matrix, 
        threshold,
        point_neighbors,
        point_pairs,
        max_points_per_cluster
    )
    
    # Keep track of used points and final clusters
    used_points = set()
    final_clusters = []
    iteration = 0
    
    # Continue until all points are used or no more valid clusters can be formed
    while candidate_clusters and len(used_points) < len(point_data) + len(point_pairs):
        iteration += 1
        iter_start = time.time()
        print(f"\nIteration {iteration}:")
        print(f"- Points used so far: {len(used_points)}/{len(point_data) + len(point_pairs)}")
        print(f"- Candidate clusters remaining: {len(candidate_clusters)}")
        
        # Find the best cluster
        best = find_best_cluster(candidate_clusters, distance_matrix)
        
        if not best:
            print("- No valid cluster found, breaking...")
            break
            
        # Expand the best cluster to include all constituent points of any merged pairs
        expanded_best = expand_cluster_with_pairs(best, point_pairs)
        
        # Add this cluster to final results
        final_clusters.append(expanded_best)
        
        # Print some details about the selected cluster
        if len(expanded_best) <= 10:  # Only print all points for small clusters
            print(f"- Selected best cluster with {len(expanded_best)} points: {[point_data[idx][0] if idx < len(point_data) else f'Pair{idx-len(point_data)}' for idx in expanded_best]}")
        else:
            first_three = [point_data[idx][0] if idx < len(point_data) else f'Pair{idx-len(point_data)}' for idx in expanded_best[:3]]
            print(f"- Selected best cluster with {len(expanded_best)} points: {first_three}... and {len(expanded_best)-3} more")
        
        # Store current used points count
        prev_used_count = len(used_points)
        
        # Update the candidate clusters with available points
        candidate_clusters = update_candidate_clusters(
            candidate_clusters,
            best,  # Use the unexpanded version for updating candidates
            distance_matrix,
            threshold,
            used_points,
            point_neighbors,
            point_pairs
        )
        
        print(f"- Added {len(used_points) - prev_used_count} new points to used points")
        print(f"- Regenerated {len(candidate_clusters)} candidate clusters")
        print(f"- Iteration completed in {time.time() - iter_start:.2f} seconds")
    
    # Add any remaining points as single-point clusters
    remaining_points = [i for i in range(len(point_data)) if i not in used_points and not any(i in pair for pair in point_pairs.values())]
    if remaining_points:
        print(f"\nAdding {len(remaining_points)} remaining points as single-point clusters")
        for i in remaining_points:
            final_clusters.append([i])
    
    # Post-process clusters to ensure all original points are included
    final_clusters = post_process_clusters(final_clusters, point_data, point_pairs)
    
    return final_clusters

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        infile = sys.argv[1]
    else:
        infile = "data/point1000.lst"
    
    # Get max_points_per_cluster from command line if provided
    max_points_per_cluster = None
    if len(sys.argv) >= 3:
        try:
            max_points_per_cluster = int(sys.argv[2])
            print(f"Limiting clusters to maximum of {max_points_per_cluster} points")
        except ValueError:
            pass
    
    print(f"Reading points from: {infile}")
    
    # Load data
    point_data = readpoints(infile)
    print(f"Loaded {len(point_data)} points")
    
    # Calculate all distances to find maximum distance
    print("Finding maximum distance for threshold calculation...")
    max_distance = 0
    
    # For efficiency, only sample a subset of points for max distance calculation
    # if the dataset is large
    sample_size = min(len(point_data), 1000)
    if sample_size < len(point_data):
        print(f"Using a sample of {sample_size} points to estimate maximum distance")
        
    sample_indices = list(range(sample_size))
    
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            dist = euclidean_dist(point_data[i], point_data[j])
            max_distance = max(max_distance, dist)
    
    # Set quality threshold to 30% of maximum distance
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance: {max_distance}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold}")
    
    # Create filtered distance matrix with pre-clustered point pairs
    distance_matrix, point_neighbors, point_pairs = create_filtered_distance_matrix(
        point_data, quality_threshold
    )
    
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    # Run the optimized QT clustering algorithm
    clustering_start_time = time.time()
    clusters = optimized_qt_clustering(
        point_data, 
        distance_matrix, 
        quality_threshold,
        point_neighbors,
        point_pairs,
        max_points_per_cluster
    )
    clustering_time = time.time() - clustering_start_time
    
    # Display results
    print(f"\nFound {len(clusters)} clusters in {clustering_time:.2f} seconds:")
    
    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)
    
    # Count non-singleton clusters
    non_singletons = sum(1 for cluster in clusters if len(cluster) > 1)
    print(f"Non-singleton clusters: {non_singletons}")
    
    # Print cluster information
    for i, cluster in enumerate(clusters):
        # Convert indices to point names
        if len(cluster) <= 10:  # Only show all points for small clusters
            point_names = [point_data[idx][0] for idx in cluster]
            print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")
        else:
            # For large clusters, just show the first few points
            first_points = [point_data[idx][0] for idx in cluster[:5]]
            print(f"Cluster {i+1}: {len(cluster)} points - {first_points}... and {len(cluster)-5} more")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")