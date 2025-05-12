import sys

with open(sys.argv[1], "r") as infile:
    lines = infile.readlines()
    

with open(sys.argv[2], "w") as outfile:
    for line in lines:
        line = line.strip().split()
        outfile.write(" ".join(line[0:3]) + "\n")

