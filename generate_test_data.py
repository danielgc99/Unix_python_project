"""
Script to generate test data for QT clustering algorithm testing.
Creates several test files with different characteristics in the testdata folder.
"""
import os
import random
import math
import numpy as np

# Create testdata directory if it doesn't exist
if not os.path.exists('testdata'):
    os.makedirs('testdata')

def generate_random_points(num_points, dimensions, min_val=-10, max_val=10, add_header=False):
    """Generate random points within the given range"""
    points = []
    for i in range(num_points):
        point = [f"Point{i+1}"]
        for _ in range(dimensions):
            point.append(random.uniform(min_val, max_val))
        points.append(point)
    
    # Add header if requested
    if add_header:
        header = ["point_id"]
        for i in range(dimensions):
            header.append(f"dim{i+1}")
        points = [header] + points
    
    return points

def generate_clustered_points(num_clusters, points_per_cluster, dimensions, 
                              cluster_radius=2.0, min_val=-10, max_val=10, add_header=False):
    """Generate points that form well-defined clusters"""
    points = []
    count = 1
    
    # Generate cluster centers
    cluster_centers = []
    for _ in range(num_clusters):
        center = []
        for _ in range(dimensions):
            center.append(random.uniform(min_val, max_val))
        cluster_centers.append(center)
    
    # Generate points around each cluster center
    for i, center in enumerate(cluster_centers):
        for _ in range(points_per_cluster):
            point = [f"Point{count}"]
            # Generate a point within the cluster radius
            for dim in range(dimensions):
                # Add random offset within the cluster radius
                offset = random.uniform(-cluster_radius, cluster_radius)
                point.append(center[dim] + offset)
            points.append(point)
            count += 1
    
    # Shuffle points to mix clusters
    random.shuffle(points)
    
    # Add header if requested
    if add_header:
        header = ["point_id"]
        for i in range(dimensions):
            header.append(f"dim{i+1}")
        points = [header] + points
    
    return points

def write_points_to_file(points, filename, include_header=False):
    """Write points to a file in the specified format"""
    with open(os.path.join('testdata', filename), 'w') as f:
        # Skip header if it exists and include_header is False
        start_idx = 1 if (len(points) > 0 and len(points[0]) > 0 and 
                          points[0][0] == "point_id" and not include_header) else 0
        
        for i in range(start_idx, len(points)):
            point = points[i]
            line = ' '.join(str(x) for x in point)
            f.write(line + '\n')

def generate_edge_case_data():
    """Generate some edge case test files"""
    
    # Case 1: Single point
    single_point = [["Point1", 0.0, 0.0, 0.0]]
    write_points_to_file(single_point, "single_point.lst")
    
    # Case 2: Two points at exactly the threshold distance
    # If threshold is 30% of max distance, and max distance is 10
    # Then threshold = 3, so we'll place points at distance 3
    two_points = [
        ["Point1", 0.0, 0.0, 0.0],
        ["Point2", 3.0, 0.0, 0.0]  # Exactly at threshold distance
    ]
    write_points_to_file(two_points, "two_points_threshold.lst")
    
    # Case 3: Points all at the same coordinates
    same_coords = []
    for i in range(10):
        same_coords.append([f"Point{i+1}", 5.0, 5.0, 5.0])
    write_points_to_file(same_coords, "same_coordinates.lst")

# Generate small random dataset (useful for quick tests)
small_random = generate_random_points(20, 3)
write_points_to_file(small_random, "small_random.lst")

# Generate medium random dataset
medium_random = generate_random_points(100, 3)
write_points_to_file(medium_random, "medium_random.lst")

# Generate clustered dataset with well-defined clusters
# 5 clusters with 10 points each in 2D
clustered_2d = generate_clustered_points(5, 10, 2, cluster_radius=1.0)
write_points_to_file(clustered_2d, "clustered_2d.lst")

# 3 clusters with 15 points each in 3D
clustered_3d = generate_clustered_points(3, 15, 3, cluster_radius=1.5)
write_points_to_file(clustered_3d, "clustered_3d.lst")

# Generate dataset with header row (to test both formats)
with_header = generate_random_points(30, 3, add_header=True)
write_points_to_file(with_header, "with_header.lst", include_header=True)

# Generate edge case datasets
generate_edge_case_data()

print("Test data generation complete. Files written to 'testdata' directory:")
for filename in os.listdir('testdata'):
    print(f"- {filename}")