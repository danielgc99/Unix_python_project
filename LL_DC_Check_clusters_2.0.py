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
    - distance_matrix: Full matrix (list of lists) with distances
    - point_neighbors: Dictionary mapping each point to its potential neighbors
    - closest_pairs: List of pairs where each point is the closest to the other
    """
    print("Creating filtered distance matrix...")
    start_time = time.time()
    
    num_points = len(point_data)
    
    # Initialize the distance matrix as a list of lists (full matrix)
    # Initially fill with infinity for distances > threshold
    distance_matrix = [[float("inf") for _ in range(num_points)] for _ in range(num_points)]
    
    # Set diagonal to 0 (distance to self)
    for i in range(num_points):
        distance_matrix[i][i] = 0.0
    
    point_neighbors = {i: set() for i in range(num_points)}
    
    # For each point, track its closest neighbor
    closest_neighbor = {}
    closest_distance = {}
    
    for i in range(num_points):
        closest_distance[i] = float("inf")
    
    total_pairs = num_points * (num_points - 1) // 2
    included_pairs = 0
    
    for i in range(num_points):
        if i % 100 == 0 and i > 0:
            progress = (i * (num_points - 1) - (i * (i - 1) // 2)) / total_pairs * 100
            print(f"Processing point {i}/{num_points} ({progress:.1f}% complete)")
            
        for j in range(i + 1, num_points):
            dist = euclidean_dist(point_data[i], point_data[j])
            
            # Only keep distances that are within the threshold
            if dist <= quality_threshold:
                # Store in both directions for easy lookup
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
                
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
    for i in range(num_points):
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
    
    # ADDED: Print closest pairs information for reference
    print("Closest pairs (showing all):")
    for pair in closest_pairs:
        print(f"  {point_data[pair[0]][0]} and {point_data[pair[1]][0]}")
    
    return distance_matrix, point_neighbors, closest_pairs

def generate_all_candidate_clusters_with_diameter_cache(point_data, distance_matrix, threshold, point_neighbors, closest_pairs, max_points_per_cluster=None):
    """
    Generates all possible candidate clusters for every point as a center.
    Skips points that are the second element in a mutual closest pair.
    Uses a diameter cache to optimize diameter calculations.
    
    Returns:
    - all_candidates: List of clusters where each cluster is a list of point indices
    - cluster_center_map: Dictionary mapping frozenset of cluster points to list of centers that produce them
    - cluster_points_map: Dictionary mapping frozenset of cluster points to the actual points list
    """
    print("Generating all candidate clusters with diameter cache optimization...")
    start_time = time.time()
    
    all_candidates = []
    total_centers = len(point_data)
    
    # Create a set of second elements in closest pairs (to skip)
    skip_points = set()
    closest_pair_map = {}  # Map from first point to second point in pair
    
    for pair in closest_pairs:
        first, second = pair
        skip_points.add(second)  # Skip the second point
        closest_pair_map[first] = second  # Map first to second
    
    skipped_points = 0
    
    # ADDED: Track clusters by set representation for comparison
    cluster_center_map = {}  # Maps frozenset of cluster points to list of centers
    cluster_points_map = {}  # Maps frozenset of cluster points to the actual points list
    
    # For each potential center point
    for center_idx in range(total_centers):
        if center_idx in skip_points:
            skipped_points += 1
            continue  # Skip this point
            
        if center_idx % 100 == 0 and center_idx > 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / center_idx) * (total_centers - center_idx - len(skip_points))
            print(f"Processing center {center_idx}/{total_centers} ({center_idx/total_centers*100:.1f}%), " +
                  f"elapsed: {elapsed:.2f}s, est. remaining: {remaining:.2f}s")
        
        # Skip points with no neighbors
        if len(point_neighbors[center_idx]) == 0:
            all_candidates.append([center_idx])  # Single-point cluster
            
            # ADDED: Track single-point clusters too
            cluster_set = frozenset([center_idx])
            if cluster_set in cluster_center_map:
                cluster_center_map[cluster_set].append(center_idx)
            else:
                cluster_center_map[cluster_set] = [center_idx]
                cluster_points_map[cluster_set] = [center_idx]
                
            continue
        
        # Start with the center point
        cluster = [center_idx]
        
        # If this center has a closest pair, add it first
        if center_idx in closest_pair_map:
            paired_point = closest_pair_map[center_idx]
            cluster.append(paired_point)
        
        # Make a list of all neighbors of the center
        available_points = [p for p in point_neighbors[center_idx] 
                           if p not in cluster]  # Exclude already added points
        
        # Optional limit on cluster size
        if max_points_per_cluster and len(cluster) + len(available_points) > max_points_per_cluster:
            available_points = available_points[:max_points_per_cluster-len(cluster)]
        
        # Initialize the diameter cache for each available point
        # This stores the maximum distance from each unclustered point to any point in the cluster
        diameter_cache = {}
        
        # Initialize with distances to the center point
        latest_added = center_idx
        for point_idx in available_points:
            diameter_cache[point_idx] = distance_matrix[center_idx][point_idx]
        
        # If we added a paired point, update the diameter cache
        if len(cluster) > 1:
            latest_added = cluster[-1]
            for point_idx in available_points:
                diameter_cache[point_idx] = max(
                    diameter_cache[point_idx],
                    distance_matrix[latest_added][point_idx]
                )
        
        # Continue adding points as long as possible
        while available_points:
            best_point = None
            best_diameter = float("inf")
            
            # Try each available point
            for point_idx in available_points:
                # Use the cached diameter value and only check distance to the latest added point
                # This is the key optimization: diam({Cn+1 ∪ ej}) = max{dn_j, dist(ej, en)}
                current_diameter = max(
                    diameter_cache[point_idx],
                    distance_matrix[latest_added][point_idx]
                )
                
                # If the point keeps the cluster within threshold and has the smallest diameter
                if current_diameter <= threshold and current_diameter < best_diameter:
                    best_point = point_idx
                    best_diameter = current_diameter
            
            # If we found a point to add, add it and update cache
            if best_point is not None:
                cluster.append(best_point)
                latest_added = best_point
                available_points.remove(best_point)
                
                # Update the diameter cache for all remaining points
                for point_idx in available_points:
                    diameter_cache[point_idx] = max(
                        diameter_cache[point_idx],
                        distance_matrix[latest_added][point_idx]
                    )
                
                # Check if we've reached the maximum cluster size
                if max_points_per_cluster and len(cluster) >= max_points_per_cluster:
                    break
            else:
                # If no point can be added without exceeding threshold, we're done
                break
                
        # Add this candidate cluster to our list of all candidates
        all_candidates.append(cluster)
        
        # ADDED: Track centers that produce the same cluster
        cluster_set = frozenset(cluster)
        if cluster_set in cluster_center_map:
            cluster_center_map[cluster_set].append(center_idx)
        else:
            cluster_center_map[cluster_set] = [center_idx]
            cluster_points_map[cluster_set] = cluster.copy()  # Store the actual points list
    
    elapsed_time = time.time() - start_time
    
    # Report final statistics
    cluster_sizes = [len(c) for c in all_candidates]
    avg_cluster_size = sum(cluster_sizes) / len(all_candidates)
    max_cluster_size = max(cluster_sizes)
    clusters_with_multiple_points = sum(1 for c in all_candidates if len(c) > 1)
    
    print(f"Generated {len(all_candidates)} candidate clusters in {elapsed_time:.2f} seconds")
    print(f"Skipped {skipped_points} centers (second elements in closest pairs)")
    print(f"Clusters with multiple points: {clusters_with_multiple_points} ({clusters_with_multiple_points/len(all_candidates)*100:.2f}%)")
    print(f"Average cluster size: {avg_cluster_size:.2f}, Maximum: {max_cluster_size}")
    
    # ADDED: Print information about centers producing the same clusters
    print("\nAnalyzing centers producing identical clusters:")
    identical_clusters = {cluster_set: centers for cluster_set, centers in cluster_center_map.items() if len(centers) > 1}
    print(f"Found {len(identical_clusters)} clusters that are produced by multiple centers")
    
    # Print details about ALL identical clusters and check if centers are closest neighbors
    if identical_clusters:
        print("Details of ALL identical clusters:")
        for i, (cluster_set, centers) in enumerate(identical_clusters.items()):
            center_names = [point_data[center][0] for center in centers]
            center_points = [center for center in centers]
            cluster_points = cluster_points_map[cluster_set]
            
            print(f"  Cluster {i+1} with {len(cluster_set)} points is produced by {len(centers)} centers: {center_names}")
            print(f"    Points in this cluster: {[point_data[idx][0] for idx in cluster_points]}")
            
            # Check if any centers are closest neighbors
            closest_neighbor_pairs = []
            for idx, center1 in enumerate(center_points):
                for center2 in center_points[idx+1:]:
                    # Check if they form a closest pair
                    is_closest = False
                    for pair in closest_pairs:
                        if (pair[0] == center1 and pair[1] == center2) or (pair[0] == center2 and pair[1] == center1):
                            is_closest = True
                            break
                    
                    if is_closest:
                        closest_neighbor_pairs.append((point_data[center1][0], point_data[center2][0]))
            
            if closest_neighbor_pairs:
                print(f"    Closest neighbor pairs among these centers: {closest_neighbor_pairs}")
            else:
                print(f"    No closest neighbor pairs among these centers")
    
    # Return the additional data structures alongside the candidates
    return all_candidates, cluster_center_map, cluster_points_map

def find_best_cluster(candidate_clusters, distance_matrix, point_data):
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
    
    # ADDED: Track all clusters with same maximum size
    tied_centers = []
    
    for cluster in largest_clusters:
        # Calculate the maximum distance (diameter) within this cluster
        cluster_diameter = 0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a, b = cluster[i], cluster[j]
                dist = distance_matrix[a][b]
                cluster_diameter = max(cluster_diameter, dist)
        
        # Add to the tied centers list
        tied_centers.append((cluster[0], cluster_diameter))
        
        # Update best cluster if this one has a smaller diameter
        if cluster_diameter < min_diameter:
            min_diameter = cluster_diameter
            best_cluster = cluster
    
    # ADDED: Print information about tied clusters
    if len(largest_clusters) > 1:
        print(f"\nFound {len(largest_clusters)} tied clusters with {max_points} points:")
        for center, diameter in tied_centers:
            print(f"  Center: {point_data[center][0]}, Diameter: {diameter:.4f}")
        print(f"Selected center {point_data[best_cluster[0]][0]} with minimum diameter {min_diameter:.4f}")
    
    return best_cluster

def update_candidate_clusters_with_diameter_cache(candidate_clusters, best_cluster, distance_matrix, threshold, used_points, point_neighbors, point_data):
    """
    Updates the list of candidate clusters using the diameter cache optimization.
    
    1. Removes the best cluster itself
    2. Removing clusters whose center point is in the best cluster
    3. Regenerating clusters for center points that are not in the best cluster
    """
    # Create a set of points that are in the best cluster for faster lookups
    best_cluster_points = set(best_cluster)
    
    # Add best cluster points to the overall used points set
    used_points.update(best_cluster_points)
    
    # Create a list to store updated candidate clusters
    updated_candidates = []
    
    # ADDED: Track centers that were removed and reasons
    removed_centers = []
    regenerated_centers = []
    
    # For each original cluster
    for cluster in candidate_clusters:
        center_point = cluster[0]
        
        # Skip this cluster if its center is in the best cluster
        if center_point in best_cluster_points:
            removed_centers.append(center_point)
            continue
        
        # Check if this center point is still available
        if center_point not in used_points:
            regenerated_centers.append(center_point)
            
            # Start with the center point
            new_cluster = [center_point]
            
            # Get available neighbor points (not used and neighbors of center)
            available_neighbors = [p for p in point_neighbors[center_point] if p not in used_points]
            
            # Initialize diameter cache with distances to center
            diameter_cache = {}
            latest_added = center_point
            
            for point_idx in available_neighbors:
                diameter_cache[point_idx] = distance_matrix[center_point][point_idx]
            
            # Continue adding points as long as possible
            while available_neighbors:
                best_point = None
                best_diameter = float("inf")
                
                # Try each available neighbor
                for point_idx in available_neighbors:
                    # Use the cached diameter and only check against the latest added point
                    current_diameter = max(
                        diameter_cache[point_idx],
                        distance_matrix[latest_added][point_idx]
                    )
                    
                    # If this point keeps the cluster within threshold and has the smallest diameter
                    if current_diameter <= threshold and current_diameter < best_diameter:
                        best_point = point_idx
                        best_diameter = current_diameter
                
                # If we found a point to add, add it and update cache
                if best_point is not None:
                    new_cluster.append(best_point)
                    latest_added = best_point
                    available_neighbors.remove(best_point)
                    
                    # Update the diameter cache for all remaining neighbors
                    for point_idx in available_neighbors:
                        diameter_cache[point_idx] = max(
                            diameter_cache[point_idx],
                            distance_matrix[latest_added][point_idx]
                        )
                else:
                    # If no point can be added without exceeding threshold, we're done
                    break
            
            # Only add if the new cluster has at least 2 points
            if len(new_cluster) >= 2:
                updated_candidates.append(new_cluster)
    
    # ADDED: Print information about updated clusters
    print(f"\nCluster update statistics:")
    print(f"  Removed {len(removed_centers)} centers that were in the best cluster")
    
    if len(removed_centers) > 0:
        display_centers = [point_data[idx][0] for idx in removed_centers]
        print(f"  Removed centers: {display_centers}")
    
    print(f"  Regenerated clusters for {len(regenerated_centers)} centers")
    
    if len(regenerated_centers) > 0:
        display_regen = [point_data[idx][0] for idx in regenerated_centers]
        print(f"  Regenerated centers: {display_regen}")
    
    return updated_candidates

def optimized_qt_clustering_with_diameter_cache(point_data, distance_matrix, threshold, point_neighbors, closest_pairs, max_points_per_cluster=None):
    """
    Optimized QT clustering algorithm with diameter cache that:
    1. Uses a filtered distance matrix with only points within threshold
    2. Keeps track of neighboring points for each point
    3. Uses diameter cache to optimize diameter calculations 
    4. Skips points that are the second element in a closest pair
    5. Has an optional limit on cluster size
    """
    print("Starting optimized QT clustering algorithm with diameter cache...")
    
    # Generate all possible candidate clusters using diameter cache
    candidate_clusters, initial_cluster_center_map, initial_cluster_points_map = generate_all_candidate_clusters_with_diameter_cache(
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
    
    # ADDED: Track centers of the selected clusters
    selected_centers = []
    
    # ADDED NEW: Track all iteration clusters sets to identify duplicates across iterations
    all_iteration_clusters = []
    all_iteration_centers = []
    
    # Continue until all points are used or no more valid clusters can be formed
    while candidate_clusters and len(used_points) < len(point_data):
        iteration += 1
        iter_start = time.time()
        print(f"\nIteration {iteration}:")
        print(f"- Points used so far: {len(used_points)}/{len(point_data)}")
        print(f"- Candidate clusters remaining: {len(candidate_clusters)}")
        
        # ADDED NEW: Store all current cluster sets to compare with previous iterations
        current_iteration_clusters = []
        for cluster in candidate_clusters:
            current_iteration_clusters.append(frozenset(cluster))
        
        all_iteration_clusters.append(current_iteration_clusters)
        
        # ADDED: If not the first iteration, compare current clusters with previous iterations
        if iteration > 1:
            print(f"\n- CROSS-ITERATION CLUSTER ANALYSIS (Iteration {iteration}):")
            
            for prev_iter in range(iteration-1):
                prev_iter_num = prev_iter + 1  # for display (1-indexed)
                
                # Compare current clusters with clusters from the previous iteration
                common_clusters = set(current_iteration_clusters).intersection(set(all_iteration_clusters[prev_iter]))
                
                print(f"  Comparing with Iteration {prev_iter_num}:")
                print(f"  Found {len(common_clusters)} identical clusters between iterations {prev_iter_num} and {iteration}")
                
                if common_clusters:
                    print(f"  Details of clusters appearing in both iterations {prev_iter_num} and {iteration}:")
                    
                    for i, cluster_set in enumerate(common_clusters):
                        # Find centers that produce this cluster in the current iteration
                        current_centers = []
                        for cluster in candidate_clusters:
                            if frozenset(cluster) == cluster_set:
                                current_centers.append(cluster[0])
                        
                        # Convert to point names
                        current_center_names = [point_data[c][0] for c in current_centers]
                        
                        # Get the points in this cluster
                        cluster_points = list(cluster_set)
                        point_names = [point_data[idx][0] for idx in cluster_points]
                        
                        print(f"    Common Cluster {i+1} with {len(cluster_set)} points:")
                        print(f"      Current centers: {current_center_names}")
                        print(f"      Points: {point_names}")
        
        # ADDED: Analyze identical clusters for this iteration
        print(f"\n- Analyzing identical clusters in iteration {iteration}:")
        
        # Create maps for this iteration
        current_cluster_center_map = {}
        current_cluster_points_map = {}
        
        # Build the maps for current iteration
        for cluster in candidate_clusters:
            cluster_set = frozenset(cluster)
            center = cluster[0]
            
            if cluster_set in current_cluster_center_map:
                current_cluster_center_map[cluster_set].append(center)
            else:
                current_cluster_center_map[cluster_set] = [center]
                current_cluster_points_map[cluster_set] = cluster
        
        # Find identical clusters
        identical_clusters = {cluster_set: centers for cluster_set, centers in current_cluster_center_map.items() 
                             if len(centers) > 1}
        
        print(f"  Found {len(identical_clusters)} clusters with multiple centers in this iteration")
        
        if identical_clusters:
            print("  Details of ALL identical clusters in this iteration:")
            for i, (cluster_set, centers) in enumerate(identical_clusters.items()):
                center_names = [point_data[center][0] for center in centers]
                center_points = centers
                cluster_points = current_cluster_points_map[cluster_set]
                
                print(f"    Identical Cluster {i+1} with {len(cluster_set)} points is produced by {len(centers)} centers: {center_names}")
                print(f"      Points in this cluster: {[point_data[idx][0] for idx in cluster_points]}")
                
                # Check if any centers are closest neighbors
                closest_neighbor_pairs = []
                for idx, center1 in enumerate(center_points):
                    for center2 in center_points[idx+1:]:
                        # Check if they form a closest pair
                        is_closest = False
                        for pair in closest_pairs:
                            if (pair[0] == center1 and pair[1] == center2) or (pair[0] == center2 and pair[1] == center1):
                                is_closest = True
                                break
                        
                        if is_closest:
                            closest_neighbor_pairs.append((point_data[center1][0], point_data[center2][0]))
                
                if closest_neighbor_pairs:
                    print(f"      Closest neighbor pairs among these centers: {closest_neighbor_pairs}")
                else:
                    print(f"      No closest neighbor pairs among these centers")
        
        # Find the best cluster
        best = find_best_cluster(candidate_clusters, distance_matrix, point_data)
        
        if not best:
            print("- No valid cluster found, breaking...")
            break
        
        # ADDED: Store the center of the selected cluster
        center_point = best[0]
        selected_centers.append(center_point)
        all_iteration_centers.append(center_point)
            
        # Add this cluster to final results
        final_clusters.append(best)
        
        # Print some details about the selected cluster
        points_in_best = [point_data[idx][0] for idx in best]
        print(f"- Selected best cluster with {len(best)} points: {points_in_best}")
        
        # ADDED: Print the center point explicitly
        print(f"- Center point: {point_data[center_point][0]}")
        
        # ADDED: Check if the selected cluster had alternative centers
        best_set = frozenset(best)
        if best_set in identical_clusters:
            alternative_centers = identical_clusters[best_set]
            print(f"- This selected cluster could also have been produced by these centers: " +
                  f"{[point_data[c][0] for c in alternative_centers if c != center_point]}")
        
        # Store current used points count
        prev_used_count = len(used_points)
        
        # Update the candidate clusters with available points
        candidate_clusters = update_candidate_clusters_with_diameter_cache(
            candidate_clusters,
            best,
            distance_matrix,
            threshold,
            used_points,
            point_neighbors,
            point_data
        )
        
        print(f"- Added {len(used_points) - prev_used_count} new points to used points")
        print(f"- Regenerated {len(candidate_clusters)} candidate clusters")
        print(f"- Iteration completed in {time.time() - iter_start:.2f} seconds")
        
        # ADDED NEW: Check if any of the regenerated clusters are identical to the best cluster
        # from this iteration (by checking if they contain exactly the same points)
        best_cluster_set = frozenset(best)
        duplicated_in_next = []
        
        for cluster in candidate_clusters:
            if frozenset(cluster) == best_cluster_set:
                duplicated_in_next.append(cluster[0])
        
        if duplicated_in_next:
            print(f"\n*** IMPORTANT: The best cluster from iteration {iteration} is duplicated in the next iteration!")
            print(f"*** Duplicated by centers: {[point_data[c][0] for c in duplicated_in_next]}")
            
            # Explain why this happened
            print("*** This means when the best cluster was removed, it was immediately regenerated with a different center.")
            print("*** This happens because the center of the new cluster wasn't used in this iteration's best cluster.")
    
    # Add any remaining points as single-point clusters
    remaining_points = [i for i in range(len(point_data)) if i not in used_points]
    if remaining_points:
        print(f"\nAdding {len(remaining_points)} remaining points as single-point clusters")
        for i in remaining_points:
            final_clusters.append([i])
    
    # ADDED: Summarize all selected centers
    print("\nSelected centers summary:")
    for i, center in enumerate(selected_centers):
        cluster = final_clusters[i]
        print(f"Iteration {i+1}: Center {point_data[center][0]} produced a cluster with {len(cluster)} points")
    
    # ADDED NEW: Final analysis of duplication across all iterations
    print("\nFINAL CROSS-ITERATION CLUSTER ANALYSIS:")
    cluster_point_sets = [frozenset(cluster) for cluster in final_clusters if len(cluster) > 1]
    
    # Check for duplicate clusters across all iterations
    duplicate_count = len(cluster_point_sets) - len(set(cluster_point_sets))
    if duplicate_count > 0:
        print(f"WARNING: Found {duplicate_count} duplicated clusters in the final result!")
        
        # Count occurrences of each cluster
        from collections import Counter
        cluster_counts = Counter(cluster_point_sets)
        
        # Print details of duplicated clusters
        print("Details of duplicated clusters in final results:")
        for cluster_set, count in cluster_counts.items():
            if count > 1:
                # Find iteration indices where this cluster appeared
                iterations = []
                for i, cluster in enumerate(final_clusters):
                    if frozenset(cluster) == cluster_set:
                        iterations.append(i+1)  # +1 for 1-indexing
                
                # Get the points in the cluster
                cluster_points = list(cluster_set)
                point_names = [point_data[idx][0] for idx in cluster_points]
                
                print(f"  Cluster with {len(cluster_set)} points appears {count} times in iterations: {iterations}")
                print(f"  Centers: {[point_data[all_iteration_centers[i-1]][0] for i in iterations]}")
                print(f"  Points: {point_names}")
    
    return final_clusters

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        infile = sys.argv[1]
    else:
        infile = "data/point100.lst"
    
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
    
    # Run the optimized QT clustering algorithm with diameter cache and closest pairs optimization
    clustering_start_time = time.time()
    clusters = optimized_qt_clustering_with_diameter_cache(
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
        point_names = [point_data[idx][0] for idx in cluster]
        print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")