"""
Script to generate validation data with known cluster structure.
This creates a dataset where we know the expected clustering outcome.
"""
import os
import random
import math
import numpy as np
import json

# Create testdata directory if it doesn't exist
if not os.path.exists('testdata'):
    os.makedirs('testdata')

# Create test directory if it doesn't exist
if not os.path.exists('test'):
    os.makedirs('test')

def generate_validation_dataset():
    """
    Generate a dataset with clear clusters that have a known structure.
    Also saves the expected clusters for validation.
    """
    # Create 3 clearly separated clusters in 3D space
    cluster_centers = [
        [0, 0, 0],     # Cluster 1 center
        [10, 10, 10],  # Cluster 2 center
        [5, -5, -5]    # Cluster 3 center
    ]
    
    # Map to track which point belongs to which expected cluster
    expected_clusters = [[], [], []]
    
    # Generate points
    all_points = []
    point_count = 1
    
    # Generate 5 points for cluster 1
    for i in range(5):
        # Small radius to ensure clear separation
        offset = [random.uniform(-0.5, 0.5) for _ in range(3)]
        coords = [cluster_centers[0][j] + offset[j] for j in range(3)]
        all_points.append([f"Point{point_count}", *coords])
        expected_clusters[0].append(point_count - 1)  # 0-based index
        point_count += 1
    
    # Generate 7 points for cluster 2
    for i in range(7):
        offset = [random.uniform(-0.5, 0.5) for _ in range(3)]
        coords = [cluster_centers[1][j] + offset[j] for j in range(3)]
        all_points.append([f"Point{point_count}", *coords])
        expected_clusters[1].append(point_count - 1)
        point_count += 1
    
    # Generate 3 points for cluster 3
    for i in range(3):
        offset = [random.uniform(-0.5, 0.5) for _ in range(3)]
        coords = [cluster_centers[2][j] + offset[j] for j in range(3)]
        all_points.append([f"Point{point_count}", *coords])
        expected_clusters[2].append(point_count - 1)
        point_count += 1
    
    # Add 3 random isolated points (noise)
    for i in range(3):
        # Place these far from any cluster
        coords = [random.uniform(15, 20) for _ in range(3)]
        all_points.append([f"Point{point_count}", *coords])
        # No need to add to expected_clusters as they should be singleton clusters
        point_count += 1
        
    # Shuffle the points to make it more realistic
    random.shuffle(all_points)
    
    # Update expected_clusters based on the new point indices after shuffling
    # First create a mapping from point name to new index
    point_name_to_idx = {point[0]: idx for idx, point in enumerate(all_points)}
    
    # Then update expected_clusters
    new_expected_clusters = [[], [], []]
    for cluster_idx, cluster in enumerate(expected_clusters):
        for old_point_idx in cluster:
            point_name = f"Point{old_point_idx + 1}"
            new_idx = point_name_to_idx[point_name]
            new_expected_clusters[cluster_idx].append(new_idx)
    
    # Sort indices within each cluster
    for cluster in new_expected_clusters:
        cluster.sort()
    
    return all_points, new_expected_clusters

# Generate the validation dataset
all_points, expected_clusters = generate_validation_dataset()

# Write points to file
validation_file = os.path.join('testdata', 'validation_clusters.lst')
with open(validation_file, 'w') as f:
    for point in all_points:
        line = ' '.join(str(x) for x in point)
        f.write(line + '\n')

# Save expected clusters to a JSON file for tests to use
expected_file = os.path.join('test', 'expected_clusters.json')
with open(expected_file, 'w') as f:
    json.dump(expected_clusters, f)

print(f"Validation dataset written to {validation_file}")
print(f"Expected clusters written to {expected_file}")
print(f"Expected clusters: {expected_clusters}")
print(f"Total points: {len(all_points)}")
print(f"Points in known clusters: {sum(len(c) for c in expected_clusters)}")
print(f"Noise points: {len(all_points) - sum(len(c) for c in expected_clusters)}")