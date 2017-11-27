import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

arr = np.loadtxt('myspace.csv', delimiter=',', usecols=1)

h = arr[arr > 0]
N = int(np.sum(h))
x = np.arange(1, len(h) + 1, dtype='float')


def get_deriv(h, x, N, kappa, alpha):
    L_k = N / kappa - N * np.log(alpha) + np.sum(h * np.log(x)) - np.sum(h * (x / alpha) ** kappa * np.log(x / alpha))
    L_a = kappa / alpha * (np.sum(h * (x / alpha) ** kappa) - N)

    return np.asscalar(L_a),np.asscalar(L_k)


def step_derivative(y, t0, h, x, N):
    kappa,alpha = y
    return get_deriv(h,x,N,alpha,kappa)


plt.plot(x, h/N)

t = np.linspace(0, 10, 30)

y = odeint(step_derivative, (20., 2.), t, args=(h, x, N))

res_alpha, res_kappa = y[-1, :]

def weib(x, k, a):
    return (k / a) * (x / a) ** (k - 1) * np.exp(-(x / a) ** k)


plt.plot(x, weib(x, res_kappa, res_alpha))
plt.show()
print(res_alpha, res_kappa)
