# -*- coding: utf-8 -*-

import numpy as np
import math
import numpy.linalg as la
import scipy.misc as msc
import scipy.ndimage as img
import matplotlib.pyplot as plt


def split_matrix(mat):
    """Splits a square matrix 'mat' into 4 quadrants."""
    m = int(math.floor(mat.shape[0] / 2))
    return [mat[:m, :m], mat[:m, m:], mat[m:, :m], mat[m:, m:]]


def box_counting(mat, lvl, number_pixels):
    """Recursive function for counting the number of squares with at least one foreground pixel"""

    # if at least one foreground pixel (characterized as 'True' in matrix is found
    if True in mat:
        # increase the counter for current lvl by 1
        number_pixels[lvl] += 1

        # if mat can be split further, split it and call fn recursively
        if mat.shape[0] > 1:
            quadrants = split_matrix(mat)
            for submatrix in quadrants:
                box_counting(submatrix, lvl + 1, number_pixels)


def get_regression_matrix(x):
    """Constructs a matrix for a regression on data vectors given in x."""
    return (np.vstack((x, np.ones(len(x))))).T


def linear_regression(X, y):
    """"Computes linear regression by using pseudo inverse of X (as seen in the lecture)"""
    return np.dot(np.dot(la.inv(np.dot(X.T, X)), X.T), y)


def foreground2BinImg(f):
    """Provided function to binarize a given image f."""
    d = img.filters.gaussian_filter(f, sigma=0.50) - \
        img.filters.gaussian_filter(f, sigma=1.00)

    d = np.abs(d)
    m = d.max()
    d[d < 0.1 * m] = 0
    d[d >= 0.1 * m] = 1
    return img.morphology.binary_closing(d)


def calculate_fractal_dimension(impath, divs):
    """Calculates the fractal dimension of an image at 'impath', visualizes the regression in a plot."""

    # get the image and binarize it
    img = msc.imread(impath, flatten=True).astype(np.float)
    binarized = foreground2BinImg(img)

    # prepare counting array
    number_pixels = np.zeros(divs + 1, np.int)
    log_scales = np.arange(0, divs + 1)  # inverse of all scaling factors (needed as x-values)

    # run the box counting algorithm
    box_counting(binarized, 0, number_pixels)

    # create a regression matrix and calculate the linear regression (that is fitting a line to log of data)
    X = get_regression_matrix(log_scales)
    w = linear_regression(X, np.log2(number_pixels))

    # plot log of scales vs log of pixels and linear fit
    plt.plot(log_scales, log_scales * w[0] + w[1], log_scales, np.log2(number_pixels), 'rs')
    plt.show()

    # the fractal dimension is defined as the slope of the linear regression, return it
    return w[0]


if __name__ == "__main__":
    fd1 = calculate_fractal_dimension("data/lightning-3.png", 9)
    fd2 = calculate_fractal_dimension("data/tree-2.png", 9)
    print("lightning:", fd1)
    print("tree:", fd2)
