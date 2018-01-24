import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from time import clock

def tanh(x, w, theta):
    return np.squeeze(np.tanh(np.tensordot(w, x, axes=(len(w.shape) - 1, 0)) - theta[..., np.newaxis]))

#d/dx tanh = 1 - tanh^2(x)
def derrTanhW(X, o, y, w, theta):
    return X*(1-o**2)


def derrTanhTheta(X, o, y, w, theta):
    return -(1-o**2)


def gaussianRBF(x,w,theta):
    # vectorized eval function of perceptron, returns a tensor in the shape of w, where the last dimension
    # contains evaluations for all pairs of parameters.
    return np.squeeze(2*np.exp(-0.5*np.power(np.tensordot(w,x,axes=(len(w.shape)-1,0)) - theta[...,np.newaxis],2))-1)


def loss(out, y):
    return 0.5 * np.sum(np.power(out - y,2),axis=-1)


def derrGaussianW(X, o, y, w, theta):
    return -X*(o+1)*(np.dot(w,X) - theta)


def derrGaussianTheta(X, o, y, w, theta):
    return (o+1)*(np.dot(w,X) - theta)


def EXY(x,y,f,w,theta):
    return f(x*w[0] + y*w[1] - theta)


def applyL2SVMPolyKernel(x, XS, ys, ms, w0, d, b=1.):
    if x.ndim == 1:
        x = x.reshape(len(x), 1)
    k = (b + np.dot(x.T, XS)) ** d
    return np.sign(np.sum(k * ys * ms, axis=1) + w0)

def plot_perceptron(w, theta, f, title=''):
    x1, x2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    map = np.vstack((np.ravel(x1),np.ravel(x2)))
    Z = f(map, w, theta)#EXY(x1, x2, f, w, theta)
    Z.shape = x1.shape

    levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())

    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
	
    plt.contourf(x1, x2, Z, cmap='RdYlBu', vmin=-2.5, vmax=2, levels=levels)
    plt.scatter(X[0, y > 0], X[1, y > 0], color='C0')
    plt.scatter(X[0, y < 0], X[1, y < 0], color='C1')

    plt.title(title)

    plt.savefig(title + '.png')

    plt.figure()


def plot_svm_classification(X, y, m, d, b):
    s = np.where(m>0)[0]
    XS = X[:,s]
    ys = y[s]
    ms = m[s]
    w0 = np.dot(ys,ms)

    x1, x2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    gridMatrix = np.column_stack((x1.flatten(), x2.flatten()))
    Z = applyL2SVMPolyKernel(gridMatrix.T, XS, ys, ms, w0,d,b)
    Z = np.reshape(Z, ((100,100)))

    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])

    plt.contourf(x1, x2, Z, cmap='RdYlBu', title="L2-KernelSVM with d="+str(d), vmin=-3, vmax=3)
    plt.scatter(X[0, y > 0], X[1, y > 0], color='C0')
    plt.scatter(X[0, y < 0], X[1, y < 0], color='C1')

    plt.title('L2 Polynomial Kernel SVM with d = ' + str(d))

    plt.savefig('svm.png')


def trainL2SVMPolyKernel(X, y, d, b=1., C=1., T=1000):
    m, n = X.shape
    I = np.eye(n)
    Y = np.outer(y,y)
    K = (b + np.dot(X.T, X))**d
    M = Y * K + Y + 1./C*I
    mu = np.ones(n) / n
    for t in range(T):
        eta = 2./(t+2)
        grd = 2 * np.dot(M, mu)
        mu += eta * (I[np.argmin(grd)] - mu)
    return mu


def train_perceptron(X, y, f, dfW, dfTheta, use_line_search=False):

    w = np.random.rand(2)*1
    theta = np.random.rand(1)*1

    init_w = w.copy()
    init_theta = theta.copy()

    if use_line_search:
        eta_list = np.logspace(0,5,6,base=0.1)#[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        eta_w_grid, eta_theta_grid = np.meshgrid(eta_list, eta_list)

    i = 0
    E = np.inf
    while i < 1000 and E > 2.76:
        #output vector
        o = f(X,w,theta)

        #square loss
        E = loss(o,y)

        eta_w = 0.005
        eta_theta = 0.001

        gradW = np.sum(dfW(X, o, y ,w, theta)*(o-y), axis=1)
        gradTheta = np.sum(dfTheta(X, o, y, w, theta)*(o-y))

        # line search
        if use_line_search:
            w_grid = w - np.tensordot(eta_w_grid,gradW,axes=0) if use_line_search else w - eta_w_grid * gradW
            theta_grid = theta - eta_theta_grid*gradTheta
            out_grid = f(X,w_grid, theta_grid)
            square_loss = 0.5 * np.sum(np.power(out_grid - y,2),axis=2)
            cmin = np.unravel_index(square_loss.argmin(),square_loss.shape)
            eta_w, eta_theta = eta_w_grid[cmin], eta_theta_grid[cmin]

            # bestE = np.inf

            # for nwTemp in eta_list:
            #     for nthetaTemp in eta_list:
            #         wTemp = w - nwTemp*gradW
            #         thetaTemp = theta - nthetaTemp * gradTheta
            #         # output vector
            #         o = f(X, wTemp, thetaTemp)
            #
            #         # square loss
            #         E = 0.5 * np.dot(o - y, o - y)
            #
            #         if E < bestE:
            #             bestE = E
            #             eta_w = nwTemp
            #             eta_theta = nthetaTemp

        w -= eta_w*gradW
        theta -= eta_theta*gradTheta

        wOrth = np.dot(np.array([[0, -1], [1, 0]]), w)

        lam = -theta/np.dot(w,w)*w[0]/wOrth[0]
        b = theta/np.dot(w,w)*w[1] + lam*wOrth[1]

        i += 1

    return E, w, theta, init_w, init_theta

#train perceptron with activation function f (+ derrivatives), choose best of k random inits. plot results
def perceptron(f, derrFW, derrFTheta, k, use_line_search=False, plot_title='perceptron'):
    errors = np.zeros(k)
    params = list()
    init_params = list()
    for i in range(k):
        e, w, theta, init_w, init_theta = train_perceptron(X, y, f, derrFW, derrFTheta, use_line_search = use_line_search)
        errors[i] = e
        params.append((w, theta))
        init_params.append((init_w, init_theta))

    best_idx = np.argmin(errors)
    plot_perceptron(*init_params[best_idx], f=f, title=plot_title+' (init)')
    plot_perceptron(*params[best_idx], f=f, title=plot_title+' (best of ' + str(k) + ')')

X = np.genfromtxt('data/xor-X.csv', delimiter=',')
y = np.genfromtxt('data/xor-y.csv', delimiter=',')

perceptron(tanh,derrTanhW,derrTanhTheta, 1, use_line_search=True, plot_title='Tanh Activation')
perceptron(gaussianRBF, derrGaussianW, derrGaussianTheta, 10, plot_title='Gaussian Activation')

m = trainL2SVMPolyKernel(X,y,2, b=1)
plot_svm_classification(X, y, m, 2, 1)

plt.show()




