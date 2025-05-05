import math
import sys

def readpoints(infile):
    point_list = []
    n = 0
    with open(infile) as file:
        for line in file:
            n += 1
            line = line.strip().split()
            if line[0].lower().startswith("point"):
                point_list.append((line[0].replace("p", "P", 1), *[float(value) for value in line[1:]]))
            else:
                point_list.append((f"Point{1+n}", *[float(value) for value in line]))
    return point_list

def euclidean_dist(pointA, pointB):
    coords1 = pointA[1:]
    coords2 = pointB[1:]
    if len(coords1) != len(coords2):
        raise ValueError("Points must have the same number of dimensions")
    square_sum = sum((coords2[i] - coords1[i]) ** 2 for i in range(len(coords1)))
    return math.sqrt(square_sum)

def candidate_cluster(center_idx, point_data, distance_matrix, threshold, used_set):
    cluster = [center_idx]
    available_points = [i for i in range(len(point_data)) if i != center_idx and i not in used_set]
    
    while available_points:
        best_point = None
        best_diameter = float('inf')
        
        for point_idx in available_points:
            trial_cluster = cluster + [point_idx]
            max_dist = 0
            for i in range(len(trial_cluster)):
                for j in range(i + 1, len(trial_cluster)):
                    p1, p2 = trial_cluster[i], trial_cluster[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    dist = distance_matrix[key]
                    if dist > max_dist:
                        max_dist = dist
            if max_dist <= threshold and max_dist < best_diameter:
                best_point = point_idx
                best_diameter = max_dist
        
        if best_point is not None:
            cluster.append(best_point)
            available_points.remove(best_point)
        else:
            break

    return cluster

def qt_clustering(point_data, distance_matrix, threshold):
    used_points = set()
    clusters = []

    # Store candidate clusters for each point (recompute only when needed)
    candidates = [None] * len(point_data)

    while len(used_points) < len(point_data):
        best_cluster = []
        best_diameter = float('inf')

        for i in range(len(point_data)):
            if i in used_points:
                continue

            # Recompute candidate if it's missing or overlaps with recently used points
            if candidates[i] is None or any(p in used_points for p in candidates[i]):
                candidates[i] = candidate_cluster(i, point_data, distance_matrix, threshold, used_points)

            cluster = candidates[i]
            if len(cluster) < 2:
                continue

            # Measure diameter
            max_dist = 0
            for a in range(len(cluster)):
                for b in range(a + 1, len(cluster)):
                    p1, p2 = cluster[a], cluster[b]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    dist = distance_matrix[key]
                    if dist > max_dist:
                        max_dist = dist

            if (len(cluster) > len(best_cluster)) or (len(cluster) == len(best_cluster) and max_dist < best_diameter):
                best_cluster = cluster
                best_diameter = max_dist

        if len(best_cluster) < 2:
            for i in range(len(point_data)):
                if i not in used_points:
                    clusters.append([i])
                    used_points.add(i)
            break
        else:
            clusters.append(best_cluster)
            for idx in best_cluster:
                used_points.add(idx)

    return clusters

if __name__ == "__main__":
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = "data/point1000.lst"

    print(f"Reading points from: {infile}")
    point_data = readpoints(infile)

    distance_matrix = {}
    for i in range(len(point_data)):
        for j in range(i + 1, len(point_data)):
            dist = euclidean_dist(point_data[i], point_data[j])
            distance_matrix[(i, j)] = dist

    max_distance = max(distance_matrix.values())
    quality_threshold = 0.3 * max_distance
    print(f"Maximum distance: {max_distance}")
    print(f"Quality Threshold (30% of diameter): {quality_threshold}")

    clusters = qt_clustering(point_data, distance_matrix, quality_threshold)

    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        point_names = [point_data[idx][0] for idx in cluster]
        print(f"Cluster {i+1}: {len(cluster)} points - {point_names}")
