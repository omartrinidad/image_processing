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


def plot3D(image, name):
    """ """
    # downscaling has a "smoothing" effect
    image = scipy.misc.imresize(image, 0.15, interp='cubic')

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, image ,rstride=1, cstride=1, cmap=plt.cm.bone, linewidth=0)
    ax.view_init(azim=28, elev=55)
    plt.savefig("resulting_images/" + name + ".png", bbox_inches="tight", transparent=True)
    #plt.show()


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

    z = l.flatten()
    w = lsq_solution_V3(X, z)

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

    return np.reshape(np.dot(X, w), (rows, cols))


def lsq_solution_V3(X, z):
    w, residual, rank, svalues = la.lstsq(X, z)
    return w


def rescale(image, fmax, fmin):
    scaled = ((image - image.max())/(image.max() - image.min())) * (fmax - fmin)
    return scaled


def illumination(img, pixels):
    """Implement illumination"""

    fmax = img.max()
    fmin = img.min()

    plot3D(img, "portrait")
    # get the log of the image
    l = log_image(img)

    #using linear model
    i_l = linear_fitting(l)
    plot3D(i_l, "linear")
    r_l = l - i_l
    r = np.exp(r_l)
    plot3D(r, "linear_plot_{}".format(pixels))
    misc.imsave("resulting_images/linear_image_{}.png".format(pixels), r)

    #using bilinear model
    i_l2 = bilinear_fitting(l)
    plot3D(i_l2, "bilinear")
    r_l = l - i_l2
    r = np.exp(r_l)
    plot3D(r, "bilinear_plot_{}".format(pixels))
    misc.imsave("resulting_images/bilinear_image_{}.png".format(pixels), r)

    #r = rescale(r, fmax, fmin)
    #misc.imshow(r)
    #plot3D(r, "bilinear_plot_rescaled")


# experiment with samples
img_portrait = readImage("images/portrait.png")

for vals in [250, 666, 2500, 25000]:
    rows, cols = img_portrait.shape
    empty = np.zeros((rows, cols)) 

    random_indexes_cols = np.random.randint(cols, size=(vals, 1))
    random_indexes_rows = np.random.randint(rows, size=(vals, 1))
    random_indexes = np.hstack((random_indexes_rows, random_indexes_cols))

    # dirty code
    for i in random_indexes:
        empty[i[0], i[1]] = img_portrait[i[0], i[1]]

    illumination(empty, vals)
