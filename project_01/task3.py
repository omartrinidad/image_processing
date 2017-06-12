################################################################################
# Task 1.3
# Illumination Compensation
################################################################################

import numpy as np
import scipy
import numpy.linalg as la
import scipy.misc as misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def readImage(filename):
    #read image
    f = misc.imread(filename, flatten=True).astype("float")
    return f

def writeImage(data,filename):
    #write image
    misc.toimage(data,cmin=0,cmax=255).save(filename)


def plot3D(image, angle=0):
    """ """
    # downscaling has a "smoothing" effect
    image = scipy.misc.imresize(image, 0.15, interp='cubic')

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, image ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    ax.view_init(azim=angle)
    plt.show()


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
    """Implement illumination"""

    img_portrait = readImage("images/portrait.png")
    plot3D(img_portrait)

    l = log_image(img_portrait)
    plot3D(l)

    #using linear model
    i_l = linear_fitting(l)
    plot3D(i_l)

    print l.max(), l.min(), np.average(l)
    print i_l.max(), i_l.min(), np.average(i_l)

    r_l = l - i_l
    plot3D(r_l)

    r = np.exp(r_l)

    #using bilinear model
    i_l2 = bilinear_fitting(l)
    plot3D(i_l2)


illumination()
