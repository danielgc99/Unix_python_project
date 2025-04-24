#QT clustering project


##Libraries
import sys

infile = sys.argv[1]

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

print(readpoints(infile))
