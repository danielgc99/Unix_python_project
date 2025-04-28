#QT clustering project


##Libraries
import sys
import math

#print(len(sys.argv))
if len(sys.argv) == 2:
    infile = sys.argv[1]
else:
    infile = "data/point100.lst"

##Functions
#loading data
def readpoints(infile):
    point_list = []
    n = 0
    with open(infile) as file:
        for line in file:
            n += 1
            line = line.strip().split()
            #files with index column starting with "point"
            if line[0].lower().startswith("point"):
                point_list.append((line[0].replace("p", "P", 1), *[float(value) for value in line[1:]])) #unpacking opperator
            #for files with no index column
            else:
                point_list.append((f"Point{1+n}",*[float(value) for value in line])) #unpacking opperator       
    return(point_list)

#print(readpoints(infile))
point_data = readpoints(infile)

def euclidean_dist(pointA, pointB):
    """Takes two points as input and calculates the euclidean distance between them"""

    coords1 = pointA[1:]
    coords2 = pointB[1:]
    
    # Verify both points have the same number of dimensions
    if len(coords1) != len(coords2):
        raise ValueError("Points must have the same number of dimensions")
    
    # Start a square sum
    square_sum = 0

    # Loop through the point values, and measure the square sum of the differences
    for i in range(len(coords1)):
        square_sum += (coords2[i] - coords1[i]) ** 2

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
#print(distance_matrix)

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
            ######used_set.add(point_data[i])
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


def best_cluster(point_data, distance_matrix, threshold, used_set):
    """
    Tries to form a cluster from each unused point.
    Returns the best cluster found: the one with the most points (and tightest if tied).
    """

    best_cluster = []
    best_diameter = float("inf")    

    # Try to form a candidate cluster from every unused point
    for i in range(len(point_data)):
        print(i)
        if i in used_set:
            continue # Skip if point has already been used
        
        # Form a candidate cluster starting from point i
        cluster = candidate_cluster(i, point_data, distance_matrix, threshold, used_set)

        # Only consider clusters that have more than 1 point
        if len(cluster) < 2:
            continue

        # Compute the diameter of this cluster 
        max_dist = 0
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                p1, p2 = cluster[a], cluster[b]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                d = distance_matrix[key]
                if d > max_dist:
                    max_dist = d # Track the largest distance in the cluster

        # Compare this cluster with the current best one
        if (len(cluster) > len(best_cluster)) or (len(cluster) == len(best_cluster) and max_dist < best_diameter):
            best_cluster = cluster
            best_diameter = max_dist
        
        return best_cluster
    
#print("Best cluster is:")
#print(best_cluster(point_data, distance_matrix, quality_threshold))


for i in range(10):

    print(f"{i+1} best cluster is formed by points:")
    used_set = set()
    best = best_cluster(point_data, distance_matrix, quality_threshold, used_set)
    used_set.update(best)
    print(best)