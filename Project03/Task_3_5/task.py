import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

X1 = data[:, 0:2].astype(np.float)

# ----------------------------------
# filter outliers
X1 = X1[np.all(X1 > 0, axis=1)]
# ----------------------------------

X1 = X1.T
# read gender data into 1D array (i.e. into a vector)
#y = data[:, 2]

# next, let's plot height vs. weight 
# first, copy information rows of X into 1D arrays
wgt = np.copy(X1[0, :])
hgt = np.copy(X1[1, :])

print(hgt,wgt)

xmin = hgt.min()-15
xmax = hgt.max()+15
ymin = wgt.min()-15
ymax = wgt.max()+15


def plot_data_and_fit(h, w, x, y,filename,title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=18)
    plt.plot(h, w, 'ko', x, y, 'r-')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.savefig("mark_plots/"+filename, bbox_inches='tight',transparent=True,pad_inches=0)
    
def trsf(x):
    return x / 100.
    
n = 10
x = np.linspace(xmin, xmax, 100)

# method 1:
# regression using ployfit
c = poly.polyfit(hgt, wgt, n)
y = poly.polyval(x, c)
plot_data_and_fit(hgt, wgt, x, y,"polyfit_mark.png","Polyfit")

# method 2:
# regression using the Vandermonde matrix and pinv
X = poly.polyvander(hgt, n)

c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y,"polyvander_pinv_mark.png","Vandermonde and pinv")

# method 3:
# regression using the Vandermonde matrix and lstsq
X = poly.polyvander(hgt, n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y,"polyvander_lstsq_mark.png","Vandermonde and lstsq")

# method 4:
# regression on transformed data using the Vandermonde
# matrix and either pinv or lstsq
X = poly.polyvander(trsf(hgt), n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit(hgt, wgt, x, y,"polyvander_pinv_transformed_mark.png","Vandermonde and pinv, on transormed Data")
