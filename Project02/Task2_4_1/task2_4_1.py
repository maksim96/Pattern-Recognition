import numpy as np
from itertools import product


def getX(n, set):
    return np.vstack([np.array(i) for j, i in enumerate(product(set, repeat=n))])

def getPhi(x):
    return np.product(np.tile(x, (2 ** len(x), 1)) ** getX(len(x), [0, 1]), axis=1)

def getBigPhi(n):
    return np.vstack([getPhi(np.array(i)) for i in product([1,-1], repeat=n)])

def getFourierCoeff(rule):
    return None
X = getX(3, [1, -1])

rule110 = np.array([-1, 1, 1, 1, -1, 1, 1, -1])
rule126 = np.array([-1, 1, 1, 1, 1, 1, 1, -1])

print('rule110: ',rule110)
print('rule126: ',rule126)

pinv = np.linalg.pinv(X)
coeffs110 = np.dot(pinv, rule110)
coeffs126 = np.dot(pinv, rule126)

print('argmin ||X*w - y ||^2 for rule110:',coeffs110)
print('argmin ||X*w - y ||^2 for rule126:',coeffs126)

print('y^ for rule110: ', np.dot(X, coeffs110))
print('y^ for rule126: ', np.dot(X, coeffs126))

Phi = getBigPhi(3)
pinv2 = np.linalg.pinv(Phi)
coeffs110_ = np.dot(pinv2, rule110)
coeffs126_ = np.dot(pinv2, rule126)

print('argmin ||Phi*w - y ||^2 for rule110:',coeffs110_)
print('argmin ||Phi*w - y ||^2 for rule126:',coeffs126_)

print('y^ for rule110: ', np.dot(Phi, coeffs110_))
print('y^ for rule126: ', np.dot(Phi, coeffs126_))