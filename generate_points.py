import random
import math
import pylab as pl
import numpy as np
from matplotlib.colors import ListedColormap


# Train data generator
def generateData(numberOfClassEl, numberOfClasses):
    data = []
    for classNum in range(numberOfClasses):
        # Choose random center of 2-dimensional gaussian
        centerX, centerY = random.random() * 5.0, random.random() * 5.0
        # Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(numberOfClassEl):
            data.append([[random.gauss(centerX, 0.4), random.gauss(centerY, 0.4)], classNum])
    return data


def showData(nClasses, nItemsInClass):
    trainData = generateData(nItemsInClass, nClasses)

    with open("kNN_data_3.txt", "w") as file:
        for i in range(len(trainData)):
            file.write(str(trainData[i][0][0]) + " " + str(trainData[i][0][1]) + " " + str(trainData[i][1]) + "\n")

    pl.scatter([trainData[i][0][0] for i in range(len(trainData))],
               [trainData[i][0][1] for i in range(len(trainData))],
               c=[trainData[i][1] for i in range(len(trainData))])
    pl.show()


if __name__ == "__main__":
    showData(5, 100)
