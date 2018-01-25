import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
data = np.loadtxt('data/whData.dat', dtype=dt, comments='#', delimiter=None)

wgt = np.array([d[0] for d in data])
hgt = np.array([d[1] for d in data])

hgt = hgt[wgt > 0]
wgt = wgt[wgt > 0]

xmin = hgt.min()-15
xmax = hgt.max()+15
ymin = wgt.min()-15
ymax = wgt.max()+15


plt.plot(hgt,wgt,'ko')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
#plt.savefig('hw_data.png',dpi=200, transparent=True)


def plot_data_and_fit(h, w, x, y):
    plt.figure()
    plt.plot(h, w, 'ko', x, y, 'r-')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    #plt.show()


def trsf(x):
    return x / 100.


def get_error(heights, weights, x, y):
    # fit is evaluated at points in x, find the closest to every h
    error = 0
    for w,h in zip(weights,heights):
        error += (w-y[np.argmin(np.abs(x - h))])**2
    return error


n = 20
x = np.linspace(xmin, xmax, 10000)

# ---------
# method 1:
# regression using ployfit

c = poly.polyfit(hgt, wgt, n)
y = poly.polyval(x, c)
plot_data_and_fit(hgt, wgt, x, y)
print("polyfit     error = ",get_error(hgt,wgt,x,y))

# ---------
# method 2:
# regression using the Vandermonde matrix and pinv

X = poly.polyvander(hgt, n)
c = np.dot(la.pinv(X), wgt)
# print("vander condition: ",la.cond(X))
# print("pinv   condition: ",la.cond(la.pinv(X)))
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y)
print("vander+pinv error = ",get_error(hgt,wgt,x,y), la.cond(X))

# ---------
# method 3:
# regression using the Vandermonde matrix and lstsq

X = poly.polyvander(hgt, n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y)
print("vander+ltsq error = ",get_error(hgt,wgt,x,y), la.cond(X))

# ---------
# method 4:
# regression on transformed data using the Vandermonde matrix and either pinv or lstsq

X = poly.polyvander(trsf(hgt), n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit(hgt, wgt, x, y)
print("transf vand error = ",get_error(hgt,wgt,x,y), la.cond(X))

# ---------
# method 5:
# regression on transformed data using the Vandermonde and Tikhonov regularization

lam = 0.5#0.004765
X = np.vstack((poly.polyvander(trsf(hgt), n), lam*np.identity(n+1)))
t_wgt = np.hstack((wgt, np.zeros((n+1,))))
c = la.lstsq(X,t_wgt)[0]
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit(hgt, wgt, x, y)
plt.suptitle('regularized least squares')
print("trans regul error = ",get_error(hgt,wgt,x,y),la.cond(X))

plt.show()