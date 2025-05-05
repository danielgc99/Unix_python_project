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
    print("\n==== GENERATING ALL CANDIDATE CLUSTERS ====")
    all_candidates = []
    
    # For each potential center point
    for center_idx in range(len(point_data)):
        print(f"\nGenerating cluster with center point {center_idx} ({point_data[center_idx][0]})")
        # Start with the center point
        cluster = [center_idx]
        
        # Make a list of all other points
        available_points = [i for i in range(len(point_data)) if i != center_idx]
        
        # Continue adding points as long as possible
        iteration = 0
        while available_points:
            iteration += 1
            print(f"  Iteration {iteration}: Current cluster size: {len(cluster)}, Available points: {len(available_points)}")
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
                print(f"    Added point {best_point} ({point_data[best_point][0]}) with max distance: {best_max_dist:.4f}")
                cluster.append(best_point)
                available_points.remove(best_point)
            else:
                # If no point can be added without exceeding threshold, we're done
                print(f"    No more points can be added without exceeding threshold")
                break
                
        # Add this candidate cluster to our list of all candidates
        print(f"  Final cluster with center {center_idx}: {cluster} (size: {len(cluster)})")
        all_candidates.append(cluster)

    print(f"\nGenerated {len(all_candidates)} candidate clusters")
    # Print some statistics about the clusters
    cluster_sizes = [len(c) for c in all_candidates]
    print(f"  Min cluster size: {min(cluster_sizes)}")
    print(f"  Max cluster size: {max(cluster_sizes)}")
    print(f"  Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.2f}")
    
    return all_candidates



def find_best_cluster(candidate_clusters, distance_matrix):
    """
    Finds the best cluster from the list of candidate clusters.
    The best cluster is the one with the most points.
    In case of a tie, the function will return the cluster with the minimum diameter.
    """
    print("\n==== FINDING BEST CLUSTER ====")
    
    if not candidate_clusters:
        print("  No candidate clusters available")
        return []
    
    # Find the maximum number of points in any cluster
    max_points = max(len(cluster) for cluster in candidate_clusters)
    print(f"  Maximum points in any cluster: {max_points}")
    
    # Get all clusters with the maximum number of points
    largest_clusters = [cluster for cluster in candidate_clusters if len(cluster) == max_points]
    print(f"  Number of clusters with {max_points} points: {len(largest_clusters)}")
    
    # If there's only one largest cluster, return it
    if len(largest_clusters) == 1:
        print(f"  Only one largest cluster found: {largest_clusters[0]}")
        return largest_clusters[0]
    
    # Otherwise, find the one with minimum diameter
    best_cluster = None
    min_diameter = float("inf")
    
    print("  Multiple largest clusters found. Finding one with minimum diameter:")
    for i, cluster in enumerate(largest_clusters):
        # Calculate the maximum distance (diameter) within this cluster
        cluster_diameter = 0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a, b = cluster[i], cluster[j]
                key = (a, b) if a < b else (b, a)
                dist = distance_matrix[key]
                cluster_diameter = max(cluster_diameter, dist)
        
        print(f"    Cluster {i+1}: diameter = {cluster_diameter:.4f}")
        
        # Update best cluster if this one has a smaller diameter
        if cluster_diameter < min_diameter:
            min_diameter = cluster_diameter
            best_cluster = cluster
    
    print(f"  Best cluster selected: {best_cluster} with diameter: {min_diameter:.4f}")
    return best_cluster

def update_candidate_clusters(candidate_clusters, best_cluster, distance_matrix, threshold, used_points):
    """
    Updates the list of candidate clusters by:
    1. Removing the best cluster itself
    2. Removing clusters whose center point is in the best cluster
    3. Regenerating clusters for center points that are not in the best cluster
       but whose original clusters contained points from the best cluster
    """
    print("\n==== UPDATING CANDIDATE CLUSTERS ====")
    print(f"  Best cluster: {best_cluster}")
    print(f"  Used points before update: {used_points}")
    
    # Create a set of points that are in the best cluster for faster lookups
    best_cluster_points = set(best_cluster)
    
    # Add best cluster points to the overall used points set
    used_points.update(best_cluster_points)
    print(f"  Used points after update: {used_points}")
    
    # Create a list to store updated candidate clusters
    updated_candidates = []
    
    # Get total number of points in the data
    total_points = max(max(cluster) for cluster in candidate_clusters) + 1
    print(f"  Total number of points: {total_points}")
    
    # Get all available points (points not in any best cluster yet)
    available_points = [i for i in range(total_points) if i not in used_points]
    print(f"  Available points: {len(available_points)}")
    
    clusters_removed = 0
    clusters_regenerated = 0
    
    # For each original cluster
    for cluster_idx, cluster in enumerate(candidate_clusters):
        center_point = cluster[0]
        
        # Skip this cluster if its center is in the best cluster
        if center_point in best_cluster_points:
            print(f"  Removing cluster with center {center_point} (center in best cluster)")
            clusters_removed += 1
            continue
        
        # Check if this center point is still available
        if center_point not in used_points:
            print(f"\n  Regenerating cluster for center {center_point}")
            clusters_regenerated += 1
            
            # Regenerate the cluster for this center using only available points
            regenerated_points = available_points + [center_point]  # Center is always included
            
            # Start with the center point
            new_cluster = [center_point]
            points_to_try = [p for p in regenerated_points if p != center_point]
            
            print(f"    Starting with center {center_point}, {len(points_to_try)} points to try")
            
            # Continue adding points as long as possible
            iteration = 0
            while points_to_try:
                iteration += 1
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
                    print(f"    Iteration {iteration}: Added point {best_point} with max distance: {best_max_dist:.4f}")
                    new_cluster.append(best_point)
                    points_to_try.remove(best_point)
                else:
                    # If no point can be added without exceeding threshold, we're done
                    print(f"    Iteration {iteration}: No more points can be added without exceeding threshold")
                    break
            
            # Only add if the new cluster has at least 2 points
            if len(new_cluster) >= 2:
                print(f"    Final regenerated cluster: {new_cluster} (size: {len(new_cluster)})")
                updated_candidates.append(new_cluster)
            else:
                print(f"    Discarding regenerated cluster - too small (only has {len(new_cluster)} points)")
    
    print(f"\n  Original clusters: {len(candidate_clusters)}")
    print(f"  Clusters removed: {clusters_removed}")
    print(f"  Clusters regenerated: {clusters_regenerated}")
    print(f"  Updated candidates: {len(updated_candidates)}")
    
    return updated_candidates

def optimized_qt_clustering(point_data, distance_matrix, threshold):
    """
    Optimized QT clustering algorithm that avoids recalculating all candidate clusters.
    Pre-computes all possible candidate clusters and then repeatedly finds and removes the best cluster.
    """
    print("\n==== STARTING OPTIMIZED QT CLUSTERING ====")
    print(f"Number of points: {len(point_data)}")
    print(f"Threshold: {threshold:.4f}")
    
    # Generate all possible candidate clusters
    candidate_clusters = generate_all_candidate_clusters(point_data, distance_matrix, threshold)
    
    # Keep track of used points and final clusters
    used_points = set()
    final_clusters = []
    
    iteration = 0
    # Continue until all points are used or no more valid clusters can be formed
    while candidate_clusters and len(used_points) < len(point_data):
        iteration += 1
        print(f"\n---- MAIN ITERATION {iteration} ----")
        print(f"Used points: {len(used_points)}/{len(point_data)}")
        print(f"Candidate clusters: {len(candidate_clusters)}")
        
        # Find the best cluster
        best = find_best_cluster(candidate_clusters, distance_matrix)
        
        if not best:
            print("No valid best cluster found, ending algorithm")
            break
            
        # Add this cluster to final results
        final_clusters.append(best)
        print(f"Added cluster to final results: {best} (size: {len(best)})")
        
        # Update the candidate clusters with available points
        candidate_clusters = update_candidate_clusters(
            candidate_clusters,
            best,
            distance_matrix,
            threshold,
            used_points
        )
    
    # Add any remaining points as single-point clusters
    remaining_singles = 0
    for i in range(len(point_data)):
        if i not in used_points:
            final_clusters.append([i])
            remaining_singles += 1
    
    if remaining_singles > 0:
        print(f"\nAdded {remaining_singles} remaining points as single-point clusters")
    
    # Print final clusters summary
    print("\n==== FINAL CLUSTERING RESULTS ====")
    print(f"Total clusters: {len(final_clusters)}")
    cluster_sizes = [len(c) for c in final_clusters]
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"  Min cluster size: {min(cluster_sizes)}")
    print(f"  Max cluster size: {max(cluster_sizes)}")
    print(f"  Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.2f}")
    
    return final_clusters

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = "data/Point10.lst"

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
    
    print(f"Calculated {len(distance_matrix)} distances between points")
    
    # Find maximum distance (diameter)
    max_distance = max(distance_matrix.values())
    
    # Set quality threshold to 30% of maximum distance
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance: {max_distance}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold}")
    
    # Run the optimized QT clustering algorithm
    clusters = optimized_qt_clustering(point_data, distance_matrix, quality_threshold)
    
    # Display results
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        # Convert indices to point names
        point_names = [point_data[idx][0] for idx in cluster]
        print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")