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
    the quality threshold. Also identify pairs of mutually closest points.
    
    Returns:
    - distance_matrix: Dictionary with only distances <= quality_threshold
    - point_neighbors: Dictionary mapping each point to its potential neighbors
    - closest_pairs: List of pairs where each point is the closest to the other
    """
    print("Creating filtered distance matrix...")
    start_time = time.time()
    
    distance_matrix = {}
    point_neighbors = {i: set() for i in range(len(point_data))}
    
    # For each point, track its closest neighbor
    closest_neighbor = {}
    closest_distance = {}
    
    for i in range(len(point_data)):
        closest_distance[i] = float("inf")
    
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
                
                # Check if this is the closest neighbor for point i
                if dist < closest_distance[i]:
                    closest_distance[i] = dist
                    closest_neighbor[i] = j
                
                # Check if this is the closest neighbor for point j
                if dist < closest_distance[j]:
                    closest_distance[j] = dist
                    closest_neighbor[j] = i
    
    # Find mutual closest pairs
    closest_pairs = []
    for i in range(len(point_data)):
        if i in closest_neighbor:
            j = closest_neighbor[i]
            # Check if i is also j's closest neighbor
            if j in closest_neighbor and closest_neighbor[j] == i:
                # Only add each pair once (with smaller index first)
                if i < j and (i, j) not in closest_pairs:
                    closest_pairs.append((i, j))
    
    elapsed_time = time.time() - start_time
    print(f"Filtered distance matrix created in {elapsed_time:.2f} seconds")
    print(f"Total pairs considered: {total_pairs}")
    print(f"Pairs included in filtered matrix: {included_pairs} ({included_pairs/total_pairs*100:.2f}%)")
    print(f"Reduction: {100 * (1 - included_pairs / total_pairs):.2f}%")
    print(f"Found {len(closest_pairs)} pairs of mutually closest points")
    
    return distance_matrix, point_neighbors, closest_pairs

def generate_all_candidate_clusters(point_data, distance_matrix, threshold, point_neighbors, closest_pairs, max_points_per_cluster=None):
    print("Generating all candidate clusters...")
    start_time = time.time()
    
    all_candidates = []
    total_centers = len(point_data)
    
    skip_points = set()
    closest_pair_map = {}
    for pair in closest_pairs:
        first, second = pair
        skip_points.add(second)
        closest_pair_map[first] = second
    
    skipped_points = 0
    
    for center_idx in range(total_centers):
        if center_idx in skip_points:
            skipped_points += 1
            continue
        
        if center_idx % 100 == 0 and center_idx > 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / center_idx) * (total_centers - center_idx - len(skip_points))
            print(f"Processing center {center_idx}/{total_centers} ({center_idx/total_centers*100:.1f}%), elapsed: {elapsed:.2f}s, est. remaining: {remaining:.2f}s")
        
        if len(point_neighbors[center_idx]) == 0:
            all_candidates.append([center_idx])
            continue
        
        cluster = [center_idx]
        if center_idx in closest_pair_map:
            paired_point = closest_pair_map[center_idx]
            cluster.append(paired_point)

        diameter_cache = [0.0] * len(point_data)
        last_added = cluster[-1]

        available_points = [p for p in point_neighbors[center_idx] if p not in cluster]

        if max_points_per_cluster and len(cluster) + len(available_points) > max_points_per_cluster:
            available_points = available_points[:max_points_per_cluster - len(cluster)]
        
        while available_points:
            best_point = None
            best_max_dist = float("inf")

            for point_idx in available_points:
                if diameter_cache[point_idx] <= threshold and diameter_cache[point_idx] < best_max_dist:
                    best_point = point_idx
                    best_max_dist = diameter_cache[point_idx]
            
            if best_point is not None:
                cluster.append(best_point)
                available_points.remove(best_point)

                last_added = best_point
                new_available = []

                for point in available_points:
                    key = (min(point, last_added), max(point, last_added))
                    if key not in distance_matrix:
                        continue
                    dist = distance_matrix[key]
                    diameter_cache[point] = max(diameter_cache[point], dist)

                    # Check if this point remains valid with all cluster points
                    valid = True
                    for cp in cluster:
                        if (min(cp, point), max(cp, point)) not in distance_matrix:
                            valid = False
                            break
                    if valid:
                        new_available.append(point)

                available_points = new_available

                if max_points_per_cluster and len(cluster) >= max_points_per_cluster:
                    break
            else:
                break

        all_candidates.append(cluster)
    
    elapsed_time = time.time() - start_time
    cluster_sizes = [len(c) for c in all_candidates]
    avg_cluster_size = sum(cluster_sizes) / len(all_candidates)
    max_cluster_size = max(cluster_sizes)
    multi_point_clusters = sum(1 for c in all_candidates if len(c) > 1)

    print(f"Generated {len(all_candidates)} candidate clusters in {elapsed_time:.2f} seconds")
    print(f"Skipped {skipped_points} centers (second elements in closest pairs)")
    print(f"Clusters with multiple points: {multi_point_clusters} ({multi_point_clusters/len(all_candidates)*100:.2f}%)")
    print(f"Average cluster size: {avg_cluster_size:.2f}, Maximum: {max_cluster_size}")

    return all_candidates

def find_best_cluster(candidate_clusters, distance_matrix):
    """
    Finds the best cluster from the list of candidate clusters.
    The best cluster is the one with the most points.
    In case of a tie, the function will return the cluster with the minimum diameter.
    
    Note: This function still calculates the full diameter for tie-breaking since
    we're comparing complete clusters, not adding points to them.
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
                if key in distance_matrix:  # Should always be true but check anyway
                    dist = distance_matrix[key]
                    cluster_diameter = max(cluster_diameter, dist)
        
        # Update best cluster if this one has a smaller diameter
        if cluster_diameter < min_diameter:
            min_diameter = cluster_diameter
            best_cluster = cluster
    
    return best_cluster

def update_candidate_clusters(candidate_clusters, best_cluster, distance_matrix, threshold, used_points, point_neighbors):
    """
    Updates the list of candidate clusters by:
    1. Removing the best cluster itself
    2. Removing clusters whose center point is in the best cluster
    3. Regenerating clusters for center points that are not in the best cluster

    Now includes diameter cache optimization.
    """
    best_cluster_points = set(best_cluster)
    used_points.update(best_cluster_points)
    updated_candidates = []

    for cluster in candidate_clusters:
        center_point = cluster[0]
        if center_point in best_cluster_points:
            continue
        if center_point not in used_points:
            new_cluster = [center_point]
            available_neighbors = [p for p in point_neighbors[center_point] if p not in used_points]

            diameter_cache = [0.0] * len(distance_matrix)
            last_added = center_point

            while available_neighbors:
                best_point = None
                best_max_dist = float("inf")

                for point_idx in available_neighbors:
                    if diameter_cache[point_idx] <= threshold and diameter_cache[point_idx] < best_max_dist:
                        best_point = point_idx
                        best_max_dist = diameter_cache[point_idx]

                if best_point is not None:
                    new_cluster.append(best_point)
                    available_neighbors.remove(best_point)
                    last_added = best_point

                    new_available = []
                    for point in available_neighbors:
                        key = (min(point, last_added), max(point, last_added))
                        if key in distance_matrix:
                            dist = distance_matrix[key]
                            diameter_cache[point] = max(diameter_cache[point], dist)

                            # Ensure point still connects to all in cluster
                            valid = True
                            for cp in new_cluster:
                                if (min(cp, point), max(cp, point)) not in distance_matrix:
                                    valid = False
                                    break
                            if valid:
                                new_available.append(point)
                    available_neighbors = new_available
                else:
                    break

            if len(new_cluster) >= 2:
                updated_candidates.append(new_cluster)

    return updated_candidates

def optimized_qt_clustering(point_data, distance_matrix, threshold, point_neighbors, closest_pairs, max_points_per_cluster=None):
    """
    Optimized QT clustering algorithm that:
    1. Uses a filtered distance matrix with only points within threshold
    2. Keeps track of neighboring points for each point
    3. Only considers valid neighbors when building clusters
    4. Skips points that are the second element in a closest pair
    5. Has an optional limit on cluster size
    6. Uses optimized diameter calculation for improved performance
    """
    print("Starting optimized QT clustering algorithm...")
    # Generate all possible candidate clusters
    candidate_clusters = generate_all_candidate_clusters(
        point_data, 
        distance_matrix, 
        threshold,
        point_neighbors,
        closest_pairs,
        max_points_per_cluster
    )
    
    # Keep track of used points and final clusters
    used_points = set()
    final_clusters = []
    iteration = 0
    
    # Continue until all points are used or no more valid clusters can be formed
    while candidate_clusters and len(used_points) < len(point_data):
        iteration += 1
        iter_start = time.time()
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
        
        # Print some details about the selected cluster
        if len(best) <= 10:  # Only print all points for small clusters
            print(f"- Selected best cluster with {len(best)} points: {[point_data[idx][0] for idx in best]}")
        else:
            first_three = [point_data[idx][0] for idx in best[:3]]
            print(f"- Selected best cluster with {len(best)} points: {first_three}... and {len(best)-3} more")
        
        # Store current used points count
        prev_used_count = len(used_points)
        
        # Update the candidate clusters with available points
        candidate_clusters = update_candidate_clusters(
            candidate_clusters,
            best,
            distance_matrix,
            threshold,
            used_points,
            point_neighbors
        )
        
        print(f"- Added {len(used_points) - prev_used_count} new points to used points")
        print(f"- Regenerated {len(candidate_clusters)} candidate clusters")
        print(f"- Iteration completed in {time.time() - iter_start:.2f} seconds")
    
    # Add any remaining points as single-point clusters
    remaining_points = [i for i in range(len(point_data)) if i not in used_points]
    if remaining_points:
        print(f"\nAdding {len(remaining_points)} remaining points as single-point clusters")
        for i in remaining_points:
            final_clusters.append([i])
    
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
    
    # Create filtered distance matrix that only contains distances <= threshold
    # Now also returns closest pairs
    distance_matrix, point_neighbors, closest_pairs = create_filtered_distance_matrix(
        point_data, quality_threshold
    )
    
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    # Run the optimized QT clustering algorithm with closest pairs optimization
    clustering_start_time = time.time()
    clusters = optimized_qt_clustering(
        point_data, 
        distance_matrix, 
        quality_threshold,
        point_neighbors,
        closest_pairs,
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