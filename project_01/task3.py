################################################################################
# Task 1.3
# Illumination Compensation
################################################################################

import numpy as np
import numpy.linalg as la
import scipy.misc as misc

#read image
def readImage(filename):
    f = misc.imread(filename, flatten=True).astype("float")
    return f

#write image
def writeImage(data,filename):
    misc.toimage(data,cmin=0,cmax=255).save(filename)


def log_image(img):
    return np.log(img + 1)

def linear_fitting(l):
    rows, cols = l.shape
    row_arr = np.array(range(rows))
    col_arr = np.array(range(cols))
    ones = np.ones(rows*cols)
    x1 = np.repeat(row_arr, cols)
    x2 = np.tile(col_arr, rows)
    X = np.vstack([x1, x2, ones]).T

    z = l.flatten()

    w = lsq_solution_V3(X, z)
    print w

    return np.reshape(np.dot(X, w), (rows, cols))

def bilinear_fitting(l):
    rows, cols = l.shape
    row_arr = np.array(range(rows))
    col_arr = np.array(range(cols))
    ones = np.ones(rows * cols)
    x1 = np.repeat(row_arr, cols)
    x2 = np.tile(col_arr, rows)
    X = np.vstack([x1 * x2, x1, x2, ones]).T

    z = l.flatten()

    w = lsq_solution_V3(X, z)
    print w

    return np.reshape(np.dot(X, w), (rows, cols))

def lsq_solution_V3(X, z):
    w, residual, rank, svalues = la.lstsq(X, z)
    return w


def illumination():
    """Implement ..."""
    img_portrait = readImage("images/portrait.png")

    l = log_image(img_portrait)

    '''
    using linear model
    '''
    i_l = linear_fitting(l)

    print l.max(), l.min(), np.average(l)
    print i_l.max(), i_l.min(), np.average(i_l)

    r_l = l - i_l

    r = np.exp(r_l)

    '''
    using bilinear model
    '''
    i_l2 = bilinear_fitting(l)

illumination()
