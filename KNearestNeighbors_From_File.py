"""Given a new item:
    1. Find distances between new item and all other items
    2. Pick k shorter distances
    3. Pick the most common class in these k distances
    4. That class is where we will classify the new item"""
import math
from random import shuffle
from sklearn import neighbors
from sklearn.metrics import euclidean_distances
from torch import ne
def ReadData(fileName):
    f = open('data.txt', 'r')
    lines = f.read().splitlines()
    f.close()
    features = lines[0].split(', ')[:-1]
    items = []
    for i in range(1, len(lines)):
        line = lines[i].split(', ')
        itemFeatures = {"Class" : line[-1]}
    for j in range(len(features)):
        f = features[j]
        v = float(line[j])
        itemFeatures[f] = v
    items.append(itemFeatures)
    shuffle(items)
    return items
def Classify(nItem, k, Items):
    if (k > len(Items)):
        return "k larger than list length"
    neighbors = []
    for item in Items:
        distance = EuclideanDistance(nItem, item);
        neighbors = UpdateNeighbors(neighbors, item, distance, k)
    count = CalculateNeighborsClass(neighbors, k)
    return FindMax(count)
def EuclideanDistance(x, y):
    s = 0
    for key in x.keys():
        s += math.pow(x[key]-y[key], 2)
    return math.sqrt(s)
def UpdateNeighbors(neighbors, item, distance, k):
    if (len(neighbors) > distance):
        neighbors[-1] = [distance, item["Class"]]
        neighbors = sorted(neighbors)
    return neighbors
def UpdateNeighbors(neighbors, item, distance, k):
    if len(neighbors) < k:
        neighbors.append([distance, item['Class']])
        neighbors = sorted(neighbors)
    else:
        if neighbors[-1][0] > distance:
            neighbors[-1] = [distance, item['Class']]
            neighbors = sorted(neighbors)
    return neighbors
def K_FoldValidation(K, k, Items):
    if K > len(Items):
        return - 1
    correct = 0
    total = len(Items) * (K-1)
    l = int(len(Items) / K)
    for i in range(K):
        trainingSet = Items[i * l:(i + 1) * l]
        testSet = Items[:i * l] + Items[(i + 1) * l]
        for item in testSet:
            itemClass = item['Class']
            itemFeatures = {}
            for key in item:
                if key != 'Class':
                    itemFeatures[key] = item[key]
            guess = Classify(itemFeatures, k, trainingSet)
            if guess == itemClass:
                correct += 1
    accuracy = correct / float(total)
    return accuracy
def Evaluate(K, k, items, iterations):
    # Run algorithm the number of
    # iterations, pick average
    accuracy = 0
    for i in range(iterations):
        shuffle(items)
        accuracy += K_FoldValidation(K, k, items)
    print (accuracy/float(iterations))
def CalculateNeighborsClass(neighbors, k):
    count = {}
    for i in range(k):
        if (neighbors[i][1] not in count):
            count[neighbors[i][1]] = 1
        else:
            count[neighbors[i][1]] += 1
    return count
def FindMax(countList):
    maximum = -1
    classification = "";
    for key in countList.keys():
        if (countList[key] > maximum):
            maximum = countList[key]
            classification = key
    return classification, maximum
def main():
    items = ReadData('data.txt')
    Evaluate(5, 5, items, 100)
if __name__ == '__main__':
    main()