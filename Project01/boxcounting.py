# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import numpy.linalg as la
import scipy.misc as msc
import scipy.ndimage as img
import matplotlib.pyplot as plt

    
def split_Matrix(M):                                                      #split matrix into 4 quadrants
    A = np.array([M[:int(M.shape[0]*0.5),:int(M.shape[1]*0.5)],           #upper left submatrix
         M[0:int(M.shape[0]*0.5),int(M.shape[1]*0.5):],                   #upper right submatrix
         M[int(M.shape[0]*0.5):,:int(M.shape[1]*0.5)],                    #lower left
         M[int(M.shape[0]*0.5):,int(M.shape[0]*0.5):]])                   #lower right
         
    return (A)                                                            #returns array of 4 matrices
    
    
def box_counting(M,lvl):            #method for counting the number of squares with at least one foreground pixel in it
    
    global number_pixels            #global array, number of pixels for each scaling factor (1, 1/2, 1/4, et.)
    
    if True in M:                   #foreground pixels are characterized as "True" in matrix
        number_pixels[lvl]+=1       #counter for current lvl (i.e. scaling factor) increased by 1
    
        if M.shape[0]==1:           #if single pixel--> no more splitting 
            return
        else:
             A = split_Matrix(M)    #not a single pixel--> split into four squares             
        
             for submatrix in A:    #for each of these four squares: recursive function call
                                                                                         
                box_counting(submatrix,lvl+1)    
    return                               
         

def get_matrix(x):          #get Matrix X for linear regression
    return(np.vstack((x,np.ones(len(x))))).T
    
def linear_regression(X,y): #linear regression via method shown in lecture
    return(np.dot(np.dot(la.inv(np.dot(X.T,X)),X.T),y))

def foreground2BinImg(f):                                   #binarize img
    d = img.filters.gaussian_filter(f, sigma=0.50) -\
        img.filters.gaussian_filter(f, sigma=1.00)
        
    d = np.abs(d)
    m = d.max()
    d[d<0.1*m] = 0
    d[d>=0.1*m] = 1
    return img.morphology.binary_closing(d)



def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)

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
    
    

imgName1 = 'lightning-3'
imgName2 = 'tree-2'
f = msc.imread(imgName1+'.png', flatten=True).astype(np.float)
g = msc.imread(imgName2+'.png', flatten=True).astype(np.float)
lightning = foreground2BinImg(f) #binarize lightning
tree = foreground2BinImg(g)      #binarize tree

number_pixels = np.zeros(10,np.int)
scales = np.array([1,2,4,8,16,32,64,128,256,512])   #inverse of all scaling factors (needed as x-values)



box_counting(lightning,0)
X = get_matrix(np.log2(scales))     #get matrix X from log of the inverted scales
w = linear_regression(X,np.log2(number_pixels)) #fit line to log of data (w = (slope,intercept))
#plot log of scales vs log of pixels and linear fit
plt.plot(np.log2(scales),np.log2(scales)*w[0]+w[1],np.log2(scales),np.log2(number_pixels),'rs')
plt.show()


#same for image of tree, X is the same
box_counting(tree,0)
w = linear_regression(X,np.log2(number_pixels))
plt.plot(np.log2(scales),np.log2(scales)*w[0]+w[1],np.log2(scales),np.log2(number_pixels),'rs')
plt.show()

