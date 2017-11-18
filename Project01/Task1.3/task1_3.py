import numpy as np
import matplotlib.pyplot as plt


def get_deriv(h, x, N, kappa, alpha):
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
        deriv = get_deriv(h, x, N, ka[0], ka[1])
        hess = get_hessian(h, x, N, ka[0], ka[1])

        ka = ka - np.linalg.solve(hess, deriv)

    return ka

def weib(x, k, a):
    return (k / a) * (x / a) ** (k - 1) * np.exp(-(x / a) ** k)


def myplot(x, h, ka1, N):
    x1 = np.linspace(0, len(h) + 1, num=1000)
    density = weib(x1, ka1[0], ka1[1])

    plt.plot(x1, density)
    plt.plot(x, h / N)
    plt.show()


def main():

    arr = np.loadtxt('myspace.csv', delimiter=',', usecols=1)

    h = arr[arr > 0]
    N = np.sum(h)
    x = np.arange(1, len(h) + 1, dtype='float')

    ka = newton(h, x, N, 1.0, 1.0, 20)

    myplot(x, h, ka, N)


if __name__ == '__main__':
    main()
