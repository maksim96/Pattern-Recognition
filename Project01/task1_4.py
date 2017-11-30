import numpy as np
from math import pi
import matplotlib.pyplot as plt

# create euclidean unit circle vectors
angles = np.linspace(0, 2 * pi, 181)
vectors = [np.cos(angles), np.sin(angles)]

# normalize them to get unit circle w.r.t. given p-norm
p = 0.5

lengths = np.power(np.sum(np.power(np.abs(vectors), p), 0), 1 / p)
norm_vectors = vectors / lengths

# plot unit circles for p=2 and p=0.5
fig = plt.figure()
axs = plt.subplot()
axs.set_aspect('equal')

plt.plot(vectors[0], vectors[1], label="$p=2$")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
axs.plot(norm_vectors[0], norm_vectors[1], label="$p=0.5$")
axs.legend(loc='upper right')

# show the plots
plt.show()

