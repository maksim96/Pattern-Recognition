import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



def plotData2D(X,mean,variance,datalabel, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    xmin = X.min()
    xmax = X.max()
    
    # plot the data 
    x0 = np.linspace(xmin,xmax,num=100)
    y0 = 1/(np.sqrt(2*np.pi*variance))*np.exp(-(x0-mean)**2/2/variance)
    
    axs.plot(x0,y0,label='normal dist')
    axs.plot(X,np.zeros_like(X), 'ro', label=datalabel)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    




if __name__ == "__main__":
    
    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # filter outliers
    data = data[np.all(data[:,0:2].astype(float)>0,axis=1)]
    
    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # transpose data
    X = X.T

    # compute mean and sample variance
    mean = np.sum(X,axis=1)/X.shape[1]
    variance = np.sum((X-mean.reshape(2,1))**2,axis=1)/(X.shape[1]-1)

    #plot
    plotData2D(X[0,:],mean[0],variance[0],'weight','plot0.pdf')
    plotData2D(X[1,:],mean[1],variance[1],'height','plot1.pdf')
    

