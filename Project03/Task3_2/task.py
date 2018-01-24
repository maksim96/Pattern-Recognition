import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance

data = np.genfromtxt('data-clustering-2.csv', delimiter=',')
beta = 4
S = np.exp(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data.T, 'sqeuclidean') * -beta))
L = np.diag(np.sum(S, axis=1)) - S

for i, x in enumerate(data.T):
    for j, y in enumerate(data.T):
        if (np.exp(-beta * np.sum((x - y) ** 2)) != S[i, j]):
            print('FUUUUUUUUUUUUUUUUUUUUUUU')

eigenValues, eigenVectors = np.linalg.eig(L)

idx = eigenValues.argsort()
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:, idx]

y = np.greater(eigenVectors[:, 1], 0)

c1 = data[:, y]
c2 = data[:, np.logical_not(y)]

print(c1.shape)
print(c2.shape)

plt.scatter(c1[0, :], c1[1, :], c='r')
plt.scatter(c2[0, :], c2[1, :], c='g')

plt.show()
#plt.savefig('task_3_2.png')
