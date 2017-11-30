import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def get_grad(h, x, N, kappa, alpha):
    L_k = N / kappa - N * np.log(alpha) + np.sum(h * np.log(x)) - np.sum(h * (x / alpha) ** kappa * np.log(x / alpha))
    L_a = kappa / alpha * (np.sum(h * (x / alpha) ** kappa) - N)

    return np.array([[np.asscalar(L_k)], [np.asscalar(L_a)]])


def get_hessian(h, x, N, kappa, alpha):
    L_k2 = -N / (kappa ** 2) - np.sum(h * ((x / alpha) ** kappa) * np.log(x / alpha) ** 2)

    L_a2 = kappa / (alpha ** 2) * (N - (kappa + 1) * np.sum(h * (x / alpha) ** kappa))

    L_ka = (1 / alpha) * np.sum(h * (x / alpha) ** kappa) + kappa / alpha * np.sum(
        h * (x / alpha) ** kappa * np.log(x / alpha)) - N / alpha

    return np.array([[np.asscalar(L_k2), np.asscalar(L_ka)],
                     [np.asscalar(L_ka), np.asscalar(L_a2)]])


def newton(h, x, N, kappa0, alpha0, iterations=20):
    ka = np.array([[kappa0], [alpha0]])

    for i in range(iterations):
        deriv = get_grad(h, x, N, ka[0], ka[1])
        hess = get_hessian(h, x, N, ka[0], ka[1])

        ka = ka - np.linalg.solve(hess, deriv)

    return ka


def weibull(x, k, a):
    return (k / a) * (x / a) ** (k - 1) * np.exp(-(x / a) ** k)


def weibull_plot(x, h, k, a, N):
    """Plots histogram h with values in x and a Weibull distribution with parameters k, a."""
    x1 = np.linspace(0, len(h) + 1, num=1000)
    density = weibull(x1, k, a)*N

    plt.plot(x1, density, label="Weibull fit")
    plt.plot(x, h, label="Google data")
    plt.legend()


def step_derivative(y, t0, h, x, N):
    """callable for np.odeint"""
    alpha,kappa = y
    return get_grad(h,x,N,alpha,kappa).flatten()


if __name__ == '__main__':
    # read data and prepare it

    arr = np.loadtxt('data/myspace.csv', delimiter=',', usecols=1)

    h = arr[arr > 0]
    N = np.sum(h)
    x = np.arange(1, len(h) + 1, dtype='float')

    # calculate using newtons method

    ka = newton(h, x, N, 1.0, 1.0, 20)

    print("newtons method:")
    print('alpha =',np.asscalar(ka[1]))
    print('kappa =',np.asscalar(ka[0]))
    print('')

    weibull_plot(x, h, ka[0], ka[1], N)

    # solve using odeint

    t = np.linspace(1,10,30)
    y = odeint(step_derivative, (5., 100.), t, args=(h, x, N))

    print("odeint:")
    print('alpha =',y[-1,1])
    print('kappa =',y[-1,0])

    plt.show()