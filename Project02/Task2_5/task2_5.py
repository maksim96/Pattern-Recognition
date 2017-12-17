from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import time, sys


class kdnode:
    def __init__(self, axis, split, leftChild, rightChild, point):
        self.axis = axis
        self.split = split
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.point = point


def kdtree_homogenous(points, depth, axisselect):
    axis = 0
    if len(points) == 0:
        return None

    if axisselect == "cycle":
        axis = depth % points.shape[1]
    elif axisselect == "variance":
        axis = np.argmax(np.var(points, axis=0))

    points = points[np.argsort(points[:, axis]), :]
    median = len(points[:, axis]) // 2
    return kdnode(axis,
                  points[median][axis],
                  kdtree_homogenous(points[:median, :], depth + 1, axisselect),
                  kdtree_homogenous(points[median + 1:, :], depth + 1, axisselect),
                  points[median])


def kdtree_inhomogenous(points, depth, axisselect):
    axis = 0
    if len(points) == 1:
        return kdnode(-1, -1, None, None, points[0, :])

    if axisselect == "cycle":
        axis = depth % points.shape[1]
    elif axisselect == "variance":
        axis = np.argmax(np.var(points, axis=0))

    midpoint = (np.max(points[:, axis]) + np.min(points[:, axis])) / 2
    return kdnode(axis, midpoint,
                  kdtree_inhomogenous(points[points[:, axis] <= midpoint], depth + 1, axisselect),
                  kdtree_inhomogenous(points[points[:, axis] > midpoint], depth + 1, axisselect),
                  None)


def depth(root):
    if root is None:
        return 0
    return 1 + np.max([depth(root.leftChild), depth(root.rightChild)])


def plot_kdtree(root, min, max):
    if root.axis == -1:
        return None

    p = np.zeros_like(min)
    p1 = np.zeros_like(min)
    p2 = np.zeros_like(min)
    p[root.axis] = root.split

    for i in range(len(min)):
        if i != root.axis:
            p1[i] = min[i]
            p2[i] = max[i]
        else:
            p1[i] = root.split
            p2[i] = root.split

    color = 'g'
    if root.axis == 0:
        color = 'r'

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=0.5)

    if root.leftChild != None:
        max1 = max.copy()
        max1[root.axis] = root.split
        plot_kdtree(root.leftChild, min, max1)

    if root.rightChild != None:
        min1 = min.copy()
        min1[root.axis] = root.split
        plot_kdtree(root.rightChild, min1, max)


def squareDist(a, b):
    return np.sum((a - b) ** 2)


def nn(point, root):
    if root.leftChild is None and root.rightChild is None:
        return root.point, squareDist(root.point, point)
    current_best = None
    dist = sys.float_info.max
    if point[root.axis] <= root.split and root.leftChild is not None:
        current_best, dist = nn(point, root.leftChild)
    elif root.rightChild is not None:
        current_best, dist = nn(point, root.rightChild)

    #check to see whether the tree is homogenous
    if root.point is not None:
        otherDist = squareDist(point, root.point)
        if otherDist < dist:
            current_best = root.point
            dist = otherDist

    if (current_best[root.axis]-root.split)**2 < dist:
        # there could be points on the other side of the splitting plane
        candidate = None
        candidateDist = 0.0
        if point[root.axis] <= root.split and root.rightChild is not None:
            candidate, candidateDist = nn(point, root.rightChild)
        elif root.leftChild is not None:
            candidate, candidateDist = nn(point, root.leftChild)

        if candidateDist < dist:
            current_best = candidate
            dist = candidateDist

    return current_best, dist


def plot_stuff(title, name, root, min, max, points):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], marker='.')
    plot_kdtree(root, min, max)
    plt.axes().set_aspect('equal', 'datalim')
    plt.title(title)
    plt.savefig(name)


def time_all_nn(root,test_points):
    start = time.time()
    for p in test_points:
        #print(p)
        nearest, dist = nn(p,root)
    end = time.time()
    print(end - start)

    return (end -start)



if __name__ == '__main__':
    points = np.loadtxt('data2-train.dat')
    test_points = np.loadtxt('data2-test.dat')
    test_points = test_points[:,0:2]
    y = points[:, 2]

    points = points[:, 0:2]
    point_list = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])

    #points = point_list

    root1 = kdtree_inhomogenous(points, 0, "cycle")
    root2 = kdtree_inhomogenous(points, 0, "variance")
    root3 = kdtree_homogenous(points, 0, "cycle")
    root4 = kdtree_homogenous(points, 0, "variance")

    #print(nn(np.array([2,4]),root1))

    min = np.ndarray.min(points, axis=0) - 1
    max = np.ndarray.max(points, axis=0) + 1

    plot_stuff("Midpoint, cycle rule, depth=" + str(depth(root1)), "midpoint_cylce.png", root1, min, max, points)
    plot_stuff("Midpoint, max var. rule, depth=" + str(depth(root2)), "midpoint_maxvar.png", root2, min, max, points)
    plot_stuff("Median, cycle rule, depth=" + str(depth(root3)), "median_cylce.png", root3, min, max, points)
    plot_stuff("Median, max var. rule, depth=" + str(depth(root4)), "median_maxvar.png", root4, min, max, points)

    roots = [root1,root2,root3,root4]
    times = []
    nn(np.array([ 137.90146979,  33.54815294]),root3)
    for i,root in enumerate(roots):
        print("root",i)
        times += [time_all_nn(root,test_points)]


