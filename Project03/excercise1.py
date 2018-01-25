import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from numpy import genfromtxt
from scipy import cluster
from scipy.spatial.distance import cdist
from numpy.random import randint
import numpy.linalg as la
from timeit import timeit

X = genfromtxt('data-clustering-1.csv', delimiter=',').T


def plot_kmeans_result(centroids, labels, k, ax):
    ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.7, cmap='brg')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='s')
    ax.scatter(centroids[:, 0], centroids[:, 1], c=range(k), marker='+', cmap='brg')


def profile_kmeans_algorithm(algo_fn, k=3, profile_reps=100):
    # run algorithm explicitly to compare results
    name = algo_fn.__name__

    fig, axs = plt.subplots(2, 2, True, True)
    fig.suptitle(name)

    for i in np.ndindex(2, 2):
        centroids, labels = algo_fn(X, k)
        plot_kmeans_result(centroids, labels, k, axs[i])


    # use timeit to profile execution times
    avg_time = timeit(name + '(X, 3)', 'from __main__ import ' + name + ", X", number=profile_reps)
    avg_time /= profile_reps
    print(name, "avg time =", avg_time, "per execution")

    fig.suptitle("{}\navg runtime: {:.4f}secs".format(name,avg_time))
    #plt.savefig('{}_multi.png'.format(name), dpi=200, transparent=True)


def lloyds_algorithm(data, k):
    kmeans = KMeans(n_clusters=k, n_init=1, max_iter=500)
    kmeans.fit(data)
    return kmeans.cluster_centers_, kmeans.labels_


def hartigan_algorithm(data, k):
    # select initial labels randomly
    labels = np.random.randint(0, k, data.shape[0])

    centroids = np.zeros((k, 2))
    class_errors = np.zeros(k)
    assignment_errors = np.zeros(k)

    converged = False
    while not converged:
        converged = True
        for j in range(data.shape[0]):
            i = labels[j]

            for ni in range(k):
                # suppose datapoint j is added to centroid ni
                labels[j] = ni

                # recalculate centers and erros
                for l in range(k):
                    cur_data = data[labels == l]
                    centroids[l] = np.mean(cur_data, axis=0)
                    class_errors[l] = np.linalg.norm(cur_data - centroids[l]) ** 2

                assignment_errors[ni] = np.sum(class_errors)

            w = np.argmin(assignment_errors)  # index of assignment with lowest error

            if i != w:
                converged = False

            labels[j] = w

    # calculate centers again after last assignment
    centroids = np.array([np.mean(data[labels == l], axis=0) for l in range(k)])
    return centroids, labels


def macqueen_algorithm(data, k):
    initial_centroid_idx = np.random.randint(0, data.shape[0], size=k)
    centroids = data[initial_centroid_idx]
    cluster_size = np.zeros(k)
    labels = np.zeros(data.shape[0])

    # compute winner clusters for each data point
    for i, p in enumerate(data):
        distances = np.linalg.norm(centroids - p, axis=1) ** 2
        closest_idx = np.argmin(distances)
        labels[i] = closest_idx
        cluster_size[closest_idx] += 1
        centroids[closest_idx] += 1. / cluster_size[closest_idx] * (p - centroids[closest_idx])

    return centroids, labels


plt.scatter(X[:,0],X[:,1], c='black', alpha=0.7)
plt.suptitle('data points to be clustered')
# plt.savefig('kmeans_data.png', dpi=200, transparent=True)
profile_kmeans_algorithm(lloyds_algorithm, 3)
profile_kmeans_algorithm(hartigan_algorithm, 3)
profile_kmeans_algorithm(macqueen_algorithm, 3)

plt.show()
