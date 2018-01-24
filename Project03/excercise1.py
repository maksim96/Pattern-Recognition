import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from numpy import genfromtxt
from scipy import  cluster
from scipy.spatial.distance import cdist
from numpy.random import randint
import numpy.linalg as la
from timeit import timeit

X = genfromtxt('data-clustering-1.csv', delimiter=',').T


def plot_kmeans_result(centroids, labels, k):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='s')
    plt.scatter(centroids[:,0],centroids[:,1], c=range(k), cmap='rainbow', marker='+')


def profile_kmeans_algorithm(algo_fn, k=3, reps=10, profile_reps=1000):

    # run algorithm explicitly to compare results

    centroids = np.zeros((reps,k,2))
    labels = np.zeros((reps,X.shape[0]))

    for i in range(reps):
        centroids[i,:], labels[i] = algo_fn(X, k)

    plot_kmeans_result(centroids[0], labels[0], k)

    # use timeit to profile execution times

    avg_time = timeit(algo_fn.__name__+'(X, 3)','from __main__ import '+algo_fn.__name__+", X", number=profile_reps)
    avg_time /= profile_reps

    print(algo_fn.__name__,"avg time =",avg_time,"per execution")


def lloyds_algorithm(data, k):
    kmeans = KMeans(n_clusters=k, n_init=1, max_iter=500)
    kmeans.fit(data)
    #print(kmeans.cluster_centers_)

    #print(kmeans.labels_)
    #plt.scatter(X[:,0],X[:,1], label='True Position')
    #plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    #plt.show()
    return kmeans.cluster_centers_, kmeans.labels_


# profile_kmeans_algorithm(lloyds_algorithm, 3)


def macQueenClustering(data, k):
    initial_centroid_idx = np.random.randint(0,data.shape[0],size=k)
    centroids = data[initial_centroid_idx]
    cluster_size = np.zeros(k)
    labels = np.zeros(data.shape[0])

    # compute winner clusters for each data point
    for i,p in enumerate(data):
        distances = np.linalg.norm(centroids-p, axis=1)
        closest_idx = np.argmin(distances)
        labels[i] = closest_idx
        cluster_size[closest_idx] += 1
        centroids[closest_idx] +=  1./cluster_size[closest_idx] * (p - centroids[closest_idx])

    return centroids, labels


# profile_kmeans_algorithm(macQueenClustering)


def hartigan_algorithm(data, k):
    # randomly assign to cluster
    labels = randint(0, k, data.shape[0])

    # # ---------------------------------
    # def E(data, labels):
    #     """ Cost function
    #     """
    #     total_cost = 0
    #     for k in range(k):
    #         data_k = data[(labels == k).nonzero()]
    #         mu_k = np.mean(data_k, 0)
    #         cost_k = np.sum(la.norm(data_k - mu_k, axis=1))
    #         total_cost += cost_k
    #     return total_cost
    # # ---------------------------------

    centroids = np.array([np.mean(data[labels == l], axis=0) for l in range(k)])

    converged = False
    while not converged:
        converged = True
        for j, i in enumerate(labels):
            # i is index of current centroid p is assigned to

            # remove x_j from c_i, recompute mean of c_i
            labels[j] = -1
            centroids[i] = np.mean(data[labels == i], axis=0)

            # use online mean calculation here!

            for k in range(k):
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
    # mu = [np.mean(data[(labels == k).nonzero()], 0) for k in range(k)]

    return None, None
    # return mu, labels

profile_kmeans_algorithm(hartigan_algorithm)

# for i in range(5):
#     __start = time()
#     centroidsMacQueen = macQueenClustering(3,X.T)
#     __end = time()
#     assignment, cdist = cluster.vq.vq(X.T, centroidsMacQueen)
#     plt.scatter(X[0, :], X[1, :], c=assignment)
#     plt.scatter(centroidsMacQueen[:, 0], centroidsMacQueen[:, 1], c='r')
#     print ('runtime_MacQueen', __end-__start)
#
#
#     plt.show()
#
#
# for i in range(5):
#     __start = time()
#     centers,labels = hartigan(X.T, K=3)
#     __end = time()
#     centers = np.array(centers)
#     assignment, cdist = cluster.vq.vq(X.T, centers)
#     plt.scatter(X[0, :], X[1, :], c=assignment)
#     plt.scatter(centers[:, 0], centers[:, 1], c='r')
#     print ('runtime_Hartigan', __end-__start)
#
#     plt.show()

plt.show()