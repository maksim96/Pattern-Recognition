import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from numpy import genfromtxt
from scipy.cluster.vq import kmeans
from scipy import  cluster
from numpy.random import randint
import numpy.linalg as la
from time import time


x, y = genfromtxt('data-clustering-1.csv', delimiter=',')

print x,y



X = np.array([x,y])


print '---------'
print X

for i in range(5):
    __start = time()
    kmeans = KMeans(n_clusters=3, n_init=1, max_iter=500)
    kmeans.fit(X.T)
    __end = time()
    print(kmeans.cluster_centers_)
    print ('runtime_kmeans', __end-__start)

    plt.scatter(X[0,:],X[1,:], label='True Position')
    plt.scatter(X[0,:],X[1,:], c=kmeans.labels_, cmap='rainbow')
    plt.show()



def macQueenClustering(k,data):
    # initialize centroids
    centroids_indices = np.random.randint(0,len(data),size=k)
    centroids = data[centroids_indices]
    nmbrElementsInClusters = np.zeros(shape=k)

    # compute winner clusters for each data point
    for dp in data:
        assignment, cdist = cluster.vq.vq([dp], centroids)
        assignment = assignment[0]
        # update cluster size and centroid
        nmbrElementsInClusters[assignment] += 1
        c_u_i = centroids[assignment]
        c_u_i = c_u_i + 1./nmbrElementsInClusters[assignment] * (dp-c_u_i)
        centroids[assignment] = c_u_i

    return centroids







def hartigan(data, K=3):
    # randomly assign to cluster
    n,m = data.shape
    labels = randint(0, K, n)

    # ---------------------------------
    def E(data, labels):
        """ Cost function
        """
        total_cost = 0
        for k in range(K):
            data_k = data[(labels == k).nonzero()]
            mu_k = np.mean(data_k, 0)
            cost_k = np.sum(la.norm(data_k - mu_k, axis=1))
            total_cost += cost_k
        return total_cost
    # ---------------------------------

    converged = False
    while not converged:
        converged = True

        for j in range(n):
            C_i = labels[j]

            min_cost = E(data, labels)
            C_w = C_i

            for k in range(K):
                #if k == C_i:
                #    continue
                labels[j] = k
                cost = E(data, labels)
                if cost < min_cost:
                    min_cost = cost
                    C_w = k

            if C_w != C_i:
                converged = False

            labels[j] = C_w

    # calculate mean for all clusters
    mu = [np.mean(data[(labels == k).nonzero()], 0) for k in range(K)]

    return mu, labels


for i in range(5):
    __start = time()
    centroidsMacQueen = macQueenClustering(3,X.T)
    __end = time()
    assignment, cdist = cluster.vq.vq(X.T, centroidsMacQueen)
    plt.scatter(X[0, :], X[1, :], c=assignment)
    plt.scatter(centroidsMacQueen[:, 0], centroidsMacQueen[:, 1], c='r')
    print ('runtime_MacQueen', __end-__start)


    plt.show()


for i in range(5):
    __start = time()
    centers,labels = hartigan(X.T, K=3)
    __end = time()
    centers = np.array(centers)
    assignment, cdist = cluster.vq.vq(X.T, centers)
    plt.scatter(X[0, :], X[1, :], c=assignment)
    plt.scatter(centers[:, 0], centers[:, 1], c='r')
    print ('runtime_Hartigan', __end-__start)

    plt.show()
