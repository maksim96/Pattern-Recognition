import numpy as np
import matplotlib.pyplot as plt


def perceptron_out(x, theta, w):
    return 2 * np.exp(-0.5 * np.power(np.dot(w.T, x) - theta, 2)) - 1


def loss_fn(x, y, theta, w):
    return 0.5 * np.sum(np.power(perceptron_out(x, theta, w) - y, 2))


def loss_fn_deriv(x, y, theta, w):
    inner_prod_offset = np.sum(np.inner(w.T, x.T), axis=0) - theta
    y_val = 2 * np.exp(-0.5 * np.power(inner_prod_offset, 2))
    theta_deriv = -np.sum(np.multiply((y_val - 1 - y), np.multiply(y_val, -inner_prod_offset)), axis=1)
    w_deriv = -np.sum(np.multiply((y_val - 1 - y), np.multiply(y_val, np.multiply(inner_prod_offset, x))), axis=1)
    return theta_deriv, w_deriv


def run_gradient_descend(x, y, eps, max_iter, update_params_fn):
    # random start values from standard normal distribution
    theta = np.random.normal()
    w = np.mat(np.random.normal(size=(2, 1)))

    # store start values for plotting
    theta0 = theta
    w0 = w

    error = np.inf
    iter = 0

    error_list = np.zeros(max_iter)

    # run descend
    while error > eps and iter < max_iter:
        error = loss_fn(x, y, theta, w)
        error_list[iter] = error

        theta_deriv, w_deriv = loss_fn_deriv(x, y, theta, w)
        theta, w = update_params_fn(x, y, theta, w, theta_deriv, w_deriv)
        iter += 1

    return theta0, w0, theta, w, error_list[:iter]


# some plotting functions

def show_perceptron_classification(x, y, theta, w, axis):
    xv, yv = np.meshgrid(np.linspace(-1.5, 1.5, 150), np.linspace(-1.5, 1.5, 150))
    map = np.vstack((np.ravel(xv), np.ravel(yv)))
    res = perceptron_out(map, theta, w)
    res.shape = xv.shape
    axis.imshow(res, origin='lower', interpolation='none', cmap='RdYlBu', extent=[-1.5, 1.5, -1.5, 1.5], vmin=-3,
                vmax=2)
    axis.scatter(x[:, y == 1][0], x[:, y == 1][1], c='steelblue', alpha=0.7)
    axis.scatter(x[:, y == -1][0], x[:, y == -1][1], c='orange', alpha=0.7)


def plot_perceptron_and_error(name, x, y, t0, w0, t, w, error_list):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)
    fig.suptitle(name, fontsize=14)
    show_perceptron_classification(x, y, t0, w0, ax[0])
    ax[0].set_title("initialization")
    show_perceptron_classification(x, y, t, w, ax[1])
    ax[1].set_title("result after {} steps".format(error_list.size))
    ax[2].plot(error_list)
    ax[2].set_title("error")


# ----------------------
#     PROGRAM START
# ----------------------

# import data

x = np.loadtxt('data/xor-X.csv', dtype=np.float, delimiter=',')
y = np.loadtxt('data/xor-Y.csv', dtype=np.float, delimiter=None)


# naive gradient descend
# ----------------------

eta_t = 0.005
eta_w = 0.001


def naive_param_update(x, y, theta, w, theta_deriv, w_deriv):
    return theta - eta_t * theta_deriv, w - eta_w * w_deriv


results = run_gradient_descend(x, y, 3, 1000, naive_param_update)
plot_perceptron_and_error("naive gradient descend", x, y, *results)


# try more sophisticated approach using line search
# -------------------------------------------------

eta_t_list = np.array((5, 0.5, 0.05, 0.005))
eta_w_list = np.array((1, 0.1, 0.01, 0.001))


def line_search_param_update(x,y, theta, w, theta_deriv, w_deriv):
    eta_t, eta_w = 0, 0
    best_error = np.inf

    # TODO double for loop = bad
    for nt in eta_t_list:
        for nw in eta_w_list:
            error = loss_fn(x, y, theta - nt * theta_deriv, w - nw * w_deriv)
            if error < best_error:
                best_error = error
                eta_t, eta_w = nt, nw

    return theta - eta_t * theta_deriv, w - eta_w * w_deriv


results = run_gradient_descend(x, y, 3, 1000, line_search_param_update)
plot_perceptron_and_error("gradient descend w/ line search", x, y, *results)
plt.show()
