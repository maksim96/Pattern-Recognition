import numpy as np
from math import pi
import matplotlib.pyplot as plt

# create 'euclidean' unit circle vectors
angles = np.linspace(0, 2 * pi, 181)
vectors = [np.cos(angles), np.sin(angles)]

plt.plot(vectors[0], vectors[1])

# scale them to get unit circle w.r.t. given p-norm
p = 0.5

lengths = np.power(np.sum(np.power(np.abs(vectors), p), 0), 1 / p)
vectors /= lengths

plt.plot(vectors[0], vectors[1])

# show the plots
plt.show()
