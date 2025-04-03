#!/usr/bin/env python3

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


    

