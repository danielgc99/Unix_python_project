#QT clustering project


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

print(readpoints(infile))
