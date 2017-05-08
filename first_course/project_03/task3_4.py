import scipy as sp
import scipy.signal as sg
import numpy as np
import pylab as pyl
import scipy.misc as msc
import scipy.ndimage as ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import *

def transform_xy(Im):
    (sy,sx) = Im.shape
    mu = [sy/2,sx/2]
    Final = np.zeros((sy,sx))
    thr = 2*np.pi
    r_i = np.linspace(0,np.sqrt(mu[1]**2 + mu[0]**2), sy)
    theta_i = np.linspace(0, thr, sy)
    r_grid, theta_grid = np.meshgrid(r_i, theta_i)
    _, gy, gx = createGaussianFilterBySize(size=(21,21))
    xo, yo = pol2cart(r_grid,theta_grid)
    xo = xo + mu[1]
    yo = yo + mu[0]
    C = np.vstack((yo.flatten(),xo.flatten()))

    transformed = ndimage.interpolation.map_coordinates(Im,C).reshape(sy,sx)
    transformed_blur = ndimage.convolve1d(transformed,gx,axis=0)

    mul_fac = sx/np.sqrt(mu[1]**2 + mu[0]**2)
    ys = mu[0] - pyl.arange(sy)
    xs = mu[1] - pyl.arange(sx)
    xq, yq = np.meshgrid(xs,ys)
    r,t = cart2pol(yq,xq)
    inx = abs(sx - t * (sx/thr))%(sx-1)
    iny = abs(sy - r*mul_fac)
    X = np.vstack((iny.flatten(),inx.flatten()))
    final = ndimage.interpolation.map_coordinates(transformed_blur.T[::-1,:],X).reshape(sy,sx).T[:,::-1]

    return transformed, transformed_blur, final

face = msc.imread("images/bauckhage.jpg", flatten=True).astype('float')
clock = msc.imread("images/clock.jpg", flatten=True).astype('float')

result = transform_xy(face)
plt.figure()
plt.imshow(result[0], cmap="Greys_r")
msc.imsave("resulting_images/one_.png", result[0])
plt.show()

plt.imshow(result[1], cmap="Greys_r")
msc.imsave("resulting_images/two_.png", result[1])
plt.show()

plt.imshow(result[2], cmap="Greys_r")
msc.imsave("resulting_images/fin_.png", result[2])
plt.show()
