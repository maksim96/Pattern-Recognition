import numpy as np
import matplotlib.pyplot as plt
from pylab import *



#d/dx tanh = 1 - tanh^2(x)
def derrTanhW(X, o, y, w, theta):
    return X*(1-o**2)

def derrTanhTheta(X, o, y, w, theta):
    return 1-o**2

def f(x):
    return 2*np.exp(-0.5*x**2) - 1

def derrFW(X, o, y, w, theta):
    return -X*(o+1)*(np.dot(w,X) - theta)

def derrFTheta(X, o, y, w, theta):
    return (o+1)*(np.dot(w,X) - theta)

def EXY(x,y,f,w,theta):
    return f(x*w[0] + y*w[1] - theta)

def applyL2SVMPolyKernel(x, XS, ys, ms, w0, d, b=1.):
    if x.ndim == 1:
        x = x.reshape(len(x), 1)
    k = (b + np.dot(x.T, XS)) ** d
    return np.sign(np.sum(k * ys * ms, axis=1) + w0)


def plotkernelSVMClassification(X,y,m,d,b):
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

    p = plt.contourf(x1, x2, Z, cmap=cm.RdBu, title="L2-KernelSVM with d="+str(d))
    plt.scatter(X[0, y > 0], X[1, y > 0], color='blue')
    plt.scatter(X[0, y < 0], X[1, y < 0], color='red')

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

def train_perceptron(X, y, f, dfW, dfTheta):
    #random init
    # w = np.array([1.0, 1.0])
    #theta = 0.0

    w = np.random.rand(2)*10
    theta = np.random.rand(1)*10

    x1, x2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    Z = EXY(x1, x2, f, w, theta)

    levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())

    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])

    p = plt.contourf(x1, x2, Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), levels=levels)
    plt.scatter(X[0, y > 0], X[1, y > 0], color='blue')
    plt.scatter(X[0, y < 0], X[1, y < 0], color='red')

    plt.figure()

    for i in range(0,100):
        #output vector
        o = f(np.dot(w, X) - theta)

        #square loss
        E = 0.5*np.dot(o - y, o - y)

        print(E)

        ns = [2,1,0.2,0.1,0.05,0.02,0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]


        nw = 0.005
        ntheta = 0.001

        gradW = np.sum(dfW(X, o, y ,w, theta)*(o-y), axis=1)
        gradTheta = np.sum(dfTheta(X, o, y, w, theta)*(o-y))

        bestE = E

        #linesearch
        for nwTemp in ns:
            for nthetaTemp in ns:
                wTemp = w - nwTemp*gradW
                thetaTemp = theta - nthetaTemp * gradTheta
                # output vector
                o = f(np.dot(wTemp, X) - thetaTemp)

                # square loss
                E = 0.5 * np.dot(o - y, o - y)

                if E < bestE:
                    bestE = E
                    nw = nwTemp
                    ntheta = nthetaTemp

        w -= nw*gradW
        theta -= ntheta*gradTheta



        p0 = theta/np.dot(w,w)*w

        wOrth = np.dot(np.array([[0, -1], [1, 0]]), w)

        lampda = -theta/np.dot(w,w)*w[0]/wOrth[0]
        b = theta/np.dot(w,w)*w[1] + lampda*wOrth[1]

        p1 = np.array([0, b]) - 2*wOrth/wOrth[0]
        p2 = np.array([0, b]) + 2*wOrth/wOrth[0]

        px = np.array([p1[0], p2[0]])
        py = np.array([p1[1], p2[1]])

        axes = plt.gca()
        axes.set_xlim([-2,2])
        axes.set_ylim([-2,2])

        if  i == 99:
            x1,x2 = np.meshgrid(np.linspace(-2,2, 100), np.linspace(-2,2, 100))
            Z = EXY(x1,x2,f,w,theta)

            levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())

            p = plt.contourf(x1,x2, Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), levels=levels)
            plt.scatter(X[0, y > 0], X[1, y > 0], color='blue')
            plt.scatter(X[0, y < 0], X[1, y < 0], color='red')

            plt.figure()





X = np.genfromtxt('data/xor-X.csv', delimiter=',')
y = np.genfromtxt('data/xor-y.csv', delimiter=',')

#train_perceptron(X,y,np.tanh,derrTanhW, derrTanhTheta)

train_perceptron(X,y,f,derrFW,derrFTheta)

m = trainL2SVMPolyKernel(X,y,2, b=1)
plotkernelSVMClassification(X, y, m,2,1)


plt.show()




