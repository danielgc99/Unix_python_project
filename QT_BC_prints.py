import math
import sys

def readpoints(infile):
    point_list = []
    n = 0
    print(f"\n[readpoints] Reading points from file: {infile}")
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
    print(f"[readpoints] Loaded {len(point_list)} points")
    print(f"[readpoints] First point: {point_list[0]}")
    print(f"[readpoints] Last point: {point_list[-1]}")
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
    print(f"\n[generate_all_candidate_clusters] Generating candidate clusters with threshold {threshold:.4f}")
    all_candidates = []
    
    # For each potential center point
    for center_idx in range(len(point_data)):
        # Start with the center point
        cluster = [center_idx]
        center_name = point_data[center_idx][0]
        print(f"\n[generate_all_candidate_clusters] Building cluster with center: {center_name} (idx: {center_idx})")
        
        # Make a list of all other points
        available_points = [i for i in range(len(point_data)) if i != center_idx]
        
        # Continue adding points as long as possible
        points_added = 0
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
                points_added += 1
                point_name = point_data[best_point][0]
                cluster.append(best_point)
                available_points.remove(best_point)
                if points_added % 5 == 0:  # Only print every 5 points to avoid too much output
                    print(f"[generate_all_candidate_clusters] Added point {point_name} (idx: {best_point}) to cluster. Current size: {len(cluster)}")
            else:
                # If no point can be added without exceeding threshold, we're done
                print(f"[generate_all_candidate_clusters] No more points can be added without exceeding threshold")
                break
                
        # Add this candidate cluster to our list of all candidates
        if len(cluster) >= 2:  # Only include clusters with at least 2 points
            all_candidates.append(cluster)
            print(f"[generate_all_candidate_clusters] Created cluster with center {center_name}: {len(cluster)} points")
        else:
            print(f"[generate_all_candidate_clusters] Skipping cluster with center {center_name} - only has 1 point")
            
    print(f"[generate_all_candidate_clusters] Generated {len(all_candidates)} candidate clusters")
    return all_candidates

def find_best_cluster(candidate_clusters):
    """
    Finds the best cluster from the list of candidate clusters.
    The best cluster is the one with the most points.
    In case of a tie, the function will return the first one found.
    """
    if not candidate_clusters:
        print("[find_best_cluster] No candidate clusters available")
        return []
    
    best_cluster = max(candidate_clusters, key=len)
    print(f"[find_best_cluster] Found best cluster with {len(best_cluster)} points, center idx: {best_cluster[0]}")
    return best_cluster

def update_candidate_clusters(candidate_clusters, best_cluster):
    """
    Updates the list of candidate clusters by:
    1. Removing the best cluster itself
    2. Removing clusters whose center point is in the best cluster
    3. Recalculating and updating clusters that contain any point from the best cluster
    
    Returns the updated list of candidate clusters
    """
    print(f"\n[update_candidate_clusters] Updating candidate clusters after removing a cluster with {len(best_cluster)} points")
    
    # Create a set of points that are in the best cluster for faster lookups
    best_cluster_points = set(best_cluster)
    
    # Create a list to store clusters to keep or update
    updated_candidates = []
    removed = 0
    updated = 0
    kept = 0
    
    for cluster in candidate_clusters:
        center_point = cluster[0]
        
        # Skip this cluster if it's the best cluster itself or its center is in the best cluster
        if center_point in best_cluster_points:
            removed += 1
            continue
            
        # Check if any point in this cluster is in the best cluster
        if any(point in best_cluster_points for point in cluster):
            # This cluster needs to be updated - filter out points in the best cluster
            original_size = len(cluster)
            updated_cluster = [point for point in cluster if point not in best_cluster_points]
            # Only add if the updated cluster has at least 2 points (including center)
            if len(updated_cluster) >= 2:
                updated_candidates.append(updated_cluster)
                updated += 1
                print(f"[update_candidate_clusters] Updated cluster with center {cluster[0]}: {original_size} → {len(updated_cluster)} points")
            else:
                removed += 1
                print(f"[update_candidate_clusters] Removed cluster with center {cluster[0]}: too small after update ({len(updated_cluster)} points)")
        else:
            # This cluster doesn't contain any points from the best cluster, keep it as is
            updated_candidates.append(cluster)
            kept += 1
    
    print(f"[update_candidate_clusters] Kept {kept} clusters unchanged")
    print(f"[update_candidate_clusters] Updated {updated} clusters")
    print(f"[update_candidate_clusters] Removed {removed} clusters")
    print(f"[update_candidate_clusters] Original: {len(candidate_clusters)} clusters → Updated: {len(updated_candidates)} clusters")
    
    return updated_candidates

def optimized_qt_clustering(point_data, distance_matrix, threshold):
    """
    Optimized QT clustering algorithm that avoids recalculating all candidate clusters.
    Pre-computes all candidate clusters and then repeatedly finds and removes the best cluster.
    """
    print(f"\n[optimized_qt_clustering] Starting QT clustering with threshold {threshold:.4f}")
    
    # Generate all possible candidate clusters
    candidate_clusters = generate_all_candidate_clusters(point_data, distance_matrix, threshold)
    
    # Keep track of used points and final clusters
    used_points = set()
    final_clusters = []
    
    iteration = 0
    # Continue until all points are used or no more valid clusters can be formed
    while candidate_clusters and len(used_points) < len(point_data):
        iteration += 1
        print(f"\n[optimized_qt_clustering] Iteration {iteration}")
        print(f"[optimized_qt_clustering] Remaining candidate clusters: {len(candidate_clusters)}")
        print(f"[optimized_qt_clustering] Points assigned to clusters so far: {len(used_points)}/{len(point_data)}")
        
        # Find the best cluster
        best = find_best_cluster(candidate_clusters)
        
        if not best:
            print("[optimized_qt_clustering] No valid cluster found, breaking")
            break
            
        # Add this cluster to final results
        final_clusters.append(best)
        
        # Update the set of used points
        previous_used = len(used_points)
        used_points.update(best)
        print(f"[optimized_qt_clustering] Added {len(used_points) - previous_used} new points to used points")
        
        # Update the candidate clusters
        candidate_clusters = update_candidate_clusters(candidate_clusters, best)
    
    # Add any remaining points as single-point clusters
    remaining_points = []
    for i in range(len(point_data)):
        if i not in used_points:
            final_clusters.append([i])
            remaining_points.append(point_data[i][0])
    
    if remaining_points:
        print(f"\n[optimized_qt_clustering] Added {len(remaining_points)} remaining points as single-point clusters:")
        print(f"[optimized_qt_clustering] Remaining points: {remaining_points}")
    else:
        print(f"\n[optimized_qt_clustering] All points assigned to clusters")
    
    print(f"[optimized_qt_clustering] Finished with {len(final_clusters)} clusters")
    return final_clusters

# Main execution
if __name__ == "__main__":
    print("================ QT CLUSTERING ALGORITHM ================")
    # Parse command line arguments
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = "data/point100.lst"

    print(f"Reading points from: {infile}")
    
    # Load data
    point_data = readpoints(infile)
    
    # Calculate distances between all points
    print("\nCalculating distance matrix...")
    distance_matrix = {}
    for i in range(len(point_data)):
        for j in range(i + 1, len(point_data)):
            dist = euclidean_dist(point_data[i], point_data[j])
            distance_matrix[(i, j)] = dist
    
    print(f"Calculated {len(distance_matrix)} distances between point pairs")
    
    # Find maximum distance (diameter)
    max_distance = max(distance_matrix.values())
    
    # Set quality threshold to 30% of maximum distance
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance (diameter): {max_distance:.4f}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold:.4f}")
    
    # Run the optimized QT clustering algorithm
    clusters = optimized_qt_clustering(point_data, distance_matrix, quality_threshold)
    
    # Display results
    print(f"\n======================== RESULTS ========================")
    print(f"Found {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        # Convert indices to point names
        point_names = [point_data[idx][0] for idx in cluster]
        if len(point_names) > 10:
            # For large clusters, just show the first few points and the count
            display_points = point_names[:3] + ['...'] + point_names[-3:]
            print(f"Cluster {i+1}: {len(cluster)} points - {display_points}")
        else:
            print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")
    
    # Print some cluster statistics
    cluster_sizes = [len(cluster) for cluster in clusters]
    single_point_clusters = sum(1 for size in cluster_sizes if size == 1)
    if cluster_sizes:
        max_cluster_size = max(cluster_sizes)
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        print(f"\nCluster statistics:")
        print(f"  - Total points: {len(point_data)}")
        print(f"  - Total clusters: {len(clusters)}")
        print(f"  - Single-point clusters: {single_point_clusters}")
        print(f"  - Multi-point clusters: {len(clusters) - single_point_clusters}")
        print(f"  - Average cluster size: {avg_cluster_size:.2f}")
        print(f"  - Largest cluster size: {max_cluster_size}")
    print("==========================================================")