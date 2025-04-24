#!/usr/bin/env python3
##Libraries
import sys

infile = sys.argv[1]

##Functions
#loading data
def readpoints(infile):
    point_list = []
    with open(infile) as file:
        for line in file:
            line = line.strip().split()
            point_list.append((line[0],[float(n) for n in line[1:]]))
    
    return(point_list)

point_data = readpoints(infile)

import math

def euclidean_dist(pointA, pointB):
    """Takes two points as input and calculates the euclidean distance between them"""

    # Start a square sum
    square_sum = 0

    # Loop through the point values, and measure the square sum of the differences
    for i in range(len(pointA[1])):
        square_sum += (pointA[1][i]- pointB[1][i])**2

    # Get the euclidean distance
    distance = math.sqrt(square_sum)
    return distance

# Calculate the distance between each point
distance_matrix = {}
for i in range(len(point_data)):
    for j in range(i + 1, len(point_data)):
        dist = euclidean_dist(point_data[i], point_data[j])
        distance_matrix[(i, j)] = dist

# Print the distance
print(distance_matrix)

# Calculate the distance between the two points that are ruthest away from each other
max_distance = max(distance_matrix.values())

# Measure QT as 30% of the maximum distance
quality_threshold = 0.3 * max_distance
print(f"Quality Threshold (30% of diameter): {quality_threshold}")

def candidate_cluster(center_idx, point_data, distance_matrix, threshold, used_set):
    """
    Forms a candidate cluster starting from center_idx.
    Only includes points that are not in used_set.
    Keeps adding points while the cluster's diameter remains within threshold.
    """

    # Start with the center point
    cluster = [center_idx]

    # Make a list of potential neighbours with their distance to the center point
    distance_to_center = []
    for i in range(len(point_data)):
        # Try each point that is not the center point or an already used point in another cluster
        if i != center_idx and i not in used_set:
            # Get the distance from centre_idx to i
            key = (center_idx, i) if center_idx < i else (i, center_idx)
            dist = distance_matrix[key]
            distance_to_center.append((i, dist))
    
    # Sort potential neighbor by distance (closest first)
    distance_to_center.sort(key = lambda x: x[1])

    # Try adding each neighbor
    for idx, _ in distance_to_center:
        # Tentatively add the point
        trial_cluster = cluster + [idx]

        # Check all pairwise distances in trial_cluster
        max_dist = 0
        for i in range(len(trial_cluster)):
            for j in range(i + 1, len(trial_cluster)):
                a, b = trial_cluster[i], trial_cluster[j]
                key = (a,b) if a < b else (b, a)
                d = distance_matrix[key]
                if d > max_dist:
                    max_dist = d

        # If the max distance is within the threshold, accept the new point
        if max_dist <= threshold:
            cluster.append(idx)
        #If not, skip it

    return cluster

    

