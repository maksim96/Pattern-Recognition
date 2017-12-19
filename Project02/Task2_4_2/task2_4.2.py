import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import euclidean_distances

def knn(k, X, x):
    distances = np.sqrt(((X - x)**2).sum(axis=1))
    return np.argpartition(distances, k)[:k]

data = np.genfromtxt('data2-test.dat')


input = np.genfromtxt('data2-train.dat')
pos = input[input[:,2] == 1][:,:2]
neg = input[input[:,2] == -1][:,:2]

plt.scatter(pos[:,0], pos[:,1], c="red")
plt.scatter(neg[:,0], neg[:,1], c="blue")

plt.savefig("testWithCorrect.png")


for k in [1,3,5]:
    plt.figure()
    prediction = np.empty([input.shape[0]])
    start = time.time()
    for i,x in enumerate(input):
        kNearestNeighbors = knn(k,data[:,:2], x[:2])
        prediction[i] =  int(np.sign(data[kNearestNeighbors,2].sum()))

    end=time.time()
    print(end-start)

    start = time.time()
    allDistances = euclidean_distances(input[:,:2], data[:,:2])
    indices = np.argpartition(allDistances, k, axis=1)[:,:k]
    neighborLabels = data[indices][:,:,2]
    prediction = np.sign(neighborLabels.sum(axis=1).flatten())


    end = time.time()
    print(end - start)


    print("Prediction for k=", k, "is:", 1 - np.abs(prediction - input[:,2]).sum()/input.shape[0])

    posPredRight = input[(prediction == 1) & (input[:,2] == 1)][:, :2]
    posPredFalse = input[(prediction == 1) & (input[:, 2] == -1)][:, :2]
    negPredRight = input[(prediction == -1) & (input[:,2] == -1)][:, :2]
    negPredFalse = input[(prediction == -1) & (input[:, 2] == 1)][:, :2]

    plt.scatter(posPredRight[:, 0], posPredRight[:, 1], color='r', label="+ correct pred")
    plt.scatter(negPredFalse[:, 0], negPredFalse[:, 1], facecolors='none', edgecolors='r', label="+ wrong pred")
    plt.scatter(negPredRight[:, 0], negPredRight[:, 1], color='b', label="- correct pred")
    plt.scatter(posPredFalse[:, 0], posPredFalse[:, 1], facecolors='none', edgecolors='b', label="- wrong pred")
    plt.legend(loc='upper left')

    plt.savefig(str(k) + "NNPrediction.png")

plt.show()