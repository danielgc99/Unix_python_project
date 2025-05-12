"""
Unit tests for QT clustering implementation.
"""
import os
import sys
import pytest
import math
from collections import defaultdict

# Add the parent directory to sys.path to import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the main module
from QT_BC_DM_NP_2_0_LL_DC_CC_2_0 import (
    readpoints,
    euclidean_dist,
    create_filtered_distance_matrix,
    find_best_cluster,
    optimized_qt_clustering_with_common_centers
)

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')

class TestQTClustering:
    """Test class for QT clustering algorithm"""
    
    def setup_method(self):
        """Setup method run before each test"""
        # Ensure test data directory exists
        assert os.path.exists(TEST_DATA_DIR), f"Test data directory {TEST_DATA_DIR} does not exist"
        
        # Set up a small test dataset for basic tests
        self.test_points = [
            ("Point1", 0.0, 0.0, 0.0),
            ("Point2", 1.0, 0.0, 0.0),
            ("Point3", 0.0, 1.0, 0.0),
            ("Point4", 0.0, 0.0, 1.0),
            ("Point5", 5.0, 5.0, 5.0),
            ("Point6", 5.1, 5.1, 5.1),
            ("Point7", 10.0, 10.0, 10.0)
        ]
        
        # Create a simple distance matrix for testing
        # Just a 3x3 matrix with known values
        self.test_distance_matrix = [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0]
        ]
        
        # Define point_neighbors for the test distance matrix
        self.test_point_neighbors = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1}
        }
        
        # Define closest pairs for testing
        self.test_closest_pairs = [(0, 1)]
        
        # Test threshold (30% of max distance in test_points)
        self.test_threshold = 3.0
    
    def test_readpoints_basic(self):
        """Test reading points from a file with no header"""
        test_file = os.path.join(TEST_DATA_DIR, 'small_random.lst')
        points = readpoints(test_file)
        
        # Check if points were read correctly
        assert len(points) > 0, "No points were read from the file"
        assert len(points[0]) > 1, "Points don't have coordinates"
        assert points[0][0].startswith("Point"), "Point names not formatted correctly"
    
    def test_readpoints_with_header(self):
        """Test reading points from a file with a header row"""
        test_file = os.path.join(TEST_DATA_DIR, 'with_header.lst')
        
        # File contains a header row, so readpoints should handle it
        points = readpoints(test_file)
        
        # Check if points were read correctly, ignoring the header
        assert len(points) > 0, "No points were read from the file"
        assert len(points[0]) > 1, "Points don't have coordinates"
        assert points[0][0].startswith("Point"), "Point names not formatted correctly"
        
        # Ensure none of the points have header values
        for point in points:
            assert "dim" not in point[0].lower(), "Header row was not properly handled"
    
    def test_euclidean_dist(self):
        """Test the Euclidean distance calculation"""
        # Test known distances with a small tolerance
        point1 = ("Point1", 0, 0, 0)
        point2 = ("Point2", 3, 4, 0)
        point3 = ("Point3", 1, 1, 1)
        
        # Distance between (0,0,0) and (3,4,0) should be 5
        dist1 = euclidean_dist(point1, point2)
        assert abs(dist1 - 5.0) < 0.001, f"Expected distance 5.0, got {dist1}"
        
        # Distance between (0,0,0) and (1,1,1) should be sqrt(3)
        dist2 = euclidean_dist(point1, point3)
        assert abs(dist2 - math.sqrt(3)) < 0.001, f"Expected distance {math.sqrt(3)}, got {dist2}"
        
        # Distance between (3,4,0) and (1,1,1) should be sqrt(13)
        dist3 = euclidean_dist(point2, point3)
        assert abs(dist3 - math.sqrt(13)) < 0.001, f"Expected distance {math.sqrt(13)}, got {dist3}"
        
        # Test with different dimensions
        point4 = ("Point4", 1, 2)
        with pytest.raises(ValueError):
            euclidean_dist(point1, point4)
    
    def test_create_filtered_distance_matrix(self):
        """Test creating the filtered distance matrix"""
        # Use the test points
        distance_matrix, point_neighbors, closest_pairs = create_filtered_distance_matrix(
            self.test_points, self.test_threshold
        )
        
        # Check dimensions of the distance matrix
        assert len(distance_matrix) == len(self.test_points), "Distance matrix has wrong dimensions"
        assert len(distance_matrix[0]) == len(self.test_points), "Distance matrix has wrong dimensions"
        
        # Check some known distances
        # Distance between (0,0,0) and (1,0,0) should be 1.0 and within threshold
        assert math.isclose(distance_matrix[0][1], 1.0), "Incorrect distance in matrix"
        
        # Distance between (0,0,0) and (5,5,5) should be sqrt(75) = 8.66 > threshold
        # So it should be set to infinity in the filtered matrix
        assert distance_matrix[0][4] == float("inf"), "Distance should be filtered out"
        
        # Check point_neighbors
        # Point1 should be neighbors with Point2, Point3, and Point4 (all within threshold)
        assert len(point_neighbors[0]) >= 3, "Point should have at least 3 neighbors"
        
        # Check closest_pairs
        assert isinstance(closest_pairs, list), "closest_pairs should be a list"
        for pair in closest_pairs:
            assert isinstance(pair, tuple), "Each pair should be a tuple"
            assert len(pair) == 2, "Each pair should contain 2 indices"
    
    def test_find_best_cluster(self):
        """Test finding the best cluster"""
        # Create test candidate clusters
        candidate_clusters = [
            [0, 1],      # 2 points
            [2, 3, 4],   # 3 points - should be chosen because more points
            [5, 6]       # 2 points
        ]
        
        # Find the best cluster
        best_cluster = find_best_cluster(candidate_clusters, self.test_distance_matrix)
        
        # The best cluster should be the one with the most points
        assert best_cluster == [2, 3, 4], "Best cluster not correctly identified"
        
        # Test with tied cluster sizes - should choose smallest diameter
        tied_clusters = [
            [0, 1],      # 2 points, diameter = 1.0
            [1, 2],      # 2 points, diameter = 1.5
        ]
        
        best_tied_cluster = find_best_cluster(tied_clusters, self.test_distance_matrix)
        
        # The best cluster should be the one with the smallest diameter
        assert best_tied_cluster == [0, 1], "Best cluster not correctly identified for tie"
        
        # Test with empty list
        assert find_best_cluster([], self.test_distance_matrix) == [], "Empty list should return empty result"
    
    def test_optimized_qt_clustering_small(self):
        """Test optimized QT clustering with a small dataset"""
        # Use a very small test file for quick testing
        test_file = os.path.join(TEST_DATA_DIR, 'small_random.lst')
        points = readpoints(test_file)
        
        # Set threshold to something reasonable for this dataset
        max_distance = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = euclidean_dist(points[i], points[j])
                max_distance = max(max_distance, dist)
        
        # If only one point, max_distance will be 0, handle this case
        if max_distance < 0.001:  # Almost zero
            threshold = 1.0  # Use a default threshold
        else:
            threshold = 0.3 * max_distance
        
        # Create filtered distance matrix
        distance_matrix, point_neighbors, closest_pairs = create_filtered_distance_matrix(
            points, threshold
        )
        
        # Run clustering
        clusters = optimized_qt_clustering_with_common_centers(
            points, distance_matrix, threshold, point_neighbors, closest_pairs
        )
        
        # Check that clusters were created
        assert isinstance(clusters, list), "Clusters should be a list"
        
        # For single point, there should be one cluster with one point
        if len(points) == 1:
            assert len(clusters) == 1, "Should have one cluster for one point"
            assert len(clusters[0]) == 1, "Cluster should contain one point"
    
    def test_optimized_qt_clustering_with_clusters(self):
        """Test optimized QT clustering with a dataset containing well-defined clusters"""
        # Use the clustered 2D dataset
        test_file = os.path.join(TEST_DATA_DIR, 'clustered_2d.lst')
        points = readpoints(test_file)
        
        # Set threshold based on the dataset
        max_distance = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = euclidean_dist(points[i], points[j])
                max_distance = max(max_distance, dist)
        
        threshold = 0.3 * max_distance
        
        # Create filtered distance matrix
        distance_matrix, point_neighbors, closest_pairs = create_filtered_distance_matrix(
            points, threshold
        )
        
        # Run clustering
        clusters = optimized_qt_clustering_with_common_centers(
            points, distance_matrix, threshold, point_neighbors, closest_pairs
        )
        
        # Check that clusters were created
        assert isinstance(clusters, list), "Clusters should be a list"
        assert len(clusters) > 0, "Should have found some clusters"
        
        # Check that clusters contain points
        for cluster in clusters:
            assert len(cluster) > 0, "Clusters should not be empty"
            
        # For well-defined clustered data, we should have found some clusters with multiple points
        has_multi_point_cluster = any(len(cluster) > 1 for cluster in clusters)
        assert has_multi_point_cluster, "Should have found at least one multi-point cluster"
        
        # All points should be included exactly once
        all_points = []
        for cluster in clusters:
            all_points.extend(cluster)
        
        assert len(all_points) == len(points), "All points should be included in clusters"
        assert len(all_points) == len(set(all_points)), "Each point should appear only once"
    
    def test_same_coordinates(self):
        """Test with points at the same coordinates"""
        # Use the same_coordinates test file
        test_file = os.path.join(TEST_DATA_DIR, 'same_coordinates.lst')
        points = readpoints(test_file)
        
        # For points at the same coordinates, all distances should be 0
        # Set a small threshold just to be safe
        threshold = 0.1
        
        # Create filtered distance matrix
        distance_matrix, point_neighbors, closest_pairs = create_filtered_distance_matrix(
            points, threshold
        )
        
        # Run clustering
        clusters = optimized_qt_clustering_with_common_centers(
            points, distance_matrix, threshold, point_neighbors, closest_pairs
        )
        
        # Check that a single cluster was created containing all points
        assert len(clusters) == 1, "Should have one cluster for points at same coordinates"
        assert len(clusters[0]) == len(points), "All points should be in the same cluster"