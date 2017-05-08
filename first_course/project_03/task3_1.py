import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.interpolate.rbf import Rbf
from scipy.spatial import distance_matrix
from numpy.linalg import inv
from math import exp, pow


# function taken from scipy
def _euclidean_norm(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum(axis=0))


# function taken from scipy
def _call_norm(x1, x2):
    norm = _euclidean_norm
    if len(x1.shape) == 1:
        x1 = x1[np.newaxis, :]
    if len(x2.shape) == 1:
        x2 = x2[np.newaxis, :]
    x1 = x1[..., :, np.newaxis]
    x2 = x2[..., np.newaxis, :]
    return norm(x1, x2)


def phi_sigma(s, sigma):
    #r = np.multiply(s, s)
    #return np.exp(-(r/(2 * sigma))**2)
    return np.divide(
                np.exp(-np.multiply(s, s)),
                (2.0 * pow(sigma, 2))
            )


def get_ys(xs, x, w, sigma):
    dists = _call_norm(xs,x)
    phs = phi_sigma(dists, sigma)
    return np.dot(phs, w)


def rbf_interpolation(xs, x, y, sigma):
    """
    """
    # get inverse of similarity matrix for x and get weights
    x = x.reshape(x.size, 1)
    y = y.reshape(y.size, 1)
    inv_sim_matrix = inv(phi_sigma(distance_matrix(x, x), sigma))
    w = np.dot(inv_sim_matrix, y)

    return w


n = 20
x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
# we will apply the function to this data
xs = np.linspace(0, n, 100)

# draw plot here
fig = pl.figure(figsize=(20, 5))

draw = fig.add_subplot(111)
draw.set_title("Radial Basis Function Interpolation")
draw.set_ylim([-2, 4])
draw.set_xlim([-0.5, 20.5])
draw.plot(x, y, 'bo', label="data")
draw.plot(x, y, 'r', label="linear")

# Scipy implementation
rbf = Rbf(x, y, function='gaussian', epsilon=0.5)
bim = rbf(xs)
draw.plot(xs, bim, 'orange', linewidth=1.3, label=r"$sigma$=0.5")

rbf = Rbf(x, y, function='gaussian', epsilon=1)
bim = rbf(xs)
draw.plot(xs, bim, 'gray', linewidth=1.3, label=r"$sigma$=1")

rbf = Rbf(x, y, function='gaussian', epsilon=2)
bim = rbf(xs)
draw.plot(xs, bim, 'brown', linewidth=1.3, label=r"$sigma$=2")

rbf = Rbf(x, y, function='gaussian', epsilon=4)
bim = rbf(xs)
draw.plot(xs, bim, 'black', linewidth=1.3, label=r"$sigma$=4")

draw.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.0)
draw.plot()

pl.savefig('slides/images/rbf_interpolation_sci.png', bbox_inches='tight')
pl.show()

# -----------------------------------------------------------------------------
# Own implementation
fig2 = pl.figure(figsize=(20, 5))

draw = fig2.add_subplot(111)
draw.set_title("Radial Basis Function Interpolation")
draw.set_ylim([-2, 4])
draw.set_xlim([-0.5, 20.5])
draw.plot(x, y, 'bo', label="data", markersize=8)
draw.plot(x, y, 'r', label="linear")

# sigma = 0.5
# w = rbf_interpolation(xs, x, y, sigma)
# bimes = get_ys(xs, x, w, sigma)
# draw.plot(xs, bimes, 'orange', linewidth=3, label=r"$sigma$={}, own".format(sigma))

sigma = 0.0
w = rbf_interpolation(xs, x, y, sigma)
bimes = get_ys(xs, x, w, sigma)
draw.plot(xs, bimes, 'blue', linewidth=3, label=r"$\sigma$={}, own".format(sigma))

# sigma = 5
# w = rbf_interpolation(xs, x, y, sigma)
# bimes = get_ys(xs, x, w, sigma)
# draw.plot(xs, bimes, 'red', linewidth=3, label=r"$sigma$={}, own".format(sigma))

#rbf = Rbf(x, y, function='gaussian', epsilon=0.5)
#bim = rbf(xs)
#draw.plot(xs, bim, 'black', linewidth=1.1, label=r"$sigma$={}, Scipy".format(sigma))

draw.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.0)
draw.plot()

pl.savefig('slides/images/rbf_interpolation_own.png', bbox_inches='tight')
pl.show()
