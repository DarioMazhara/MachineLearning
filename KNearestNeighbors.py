# Example program for K Nearest Neighbors
# Python3 program to find groups of unknown
# Points using K nearest neighbour algorithm.

import math
# K = number of nearest neighbors to consider
def classifyPoint(points, p, k = 3):
    """Finds classification of p, assumes only
    two groups returns 0 or 1 for group"""
    
    distance = []
    for group in points:
        for feature in points[group]:
            euclidean_distance = math.sqrt((feature[0]-p[0])**2 + (feature[1]-p[1])**2)
            
            distance.append((euclidean_distance, group))
    # sort list in ascending order, selecting first k distances
    distance = sorted(distance)[:k]
    
    freq1 = 0
    freq2 = 0
    
    for d in distance:
        if d[1] == 0:
            freq1 += 1
        elif d[1] == 1:
            freq2 += 1
            
    return 0 if freq1 > freq2 else 1

def main():
    
    points = {0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)],
              1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}
    
    p = (2.5, 7)
    
    k = 3
    
    print("Classified value to p: {}", format(classifyPoint(points, p, k)))

if __name__=='__main__':
    main()
    
    