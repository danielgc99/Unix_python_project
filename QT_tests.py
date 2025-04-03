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

for i in range(len(point_data)):
    for x in range(i + 1, len(point_data)):
        dist = euclidean_dist(point_data[i], point_data[x])
        print(f"Distance between {point_data[i][0]} and {point_data[x][0]}: {dist}")
    

