import scipy as sp
import numpy as np
import pylab as pyl
import scipy.misc as msc
import scipy.ndimage as ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import *

def warping(Im, amp, ph, frq):

    (sy,sx) = Im.shape
    zoom = [1 - 2*amp[1]*1./sx,1 - 2*amp[0]*1./sy]
    Im = ndimage.interpolation.zoom(Im, zoom)
    frqx = frq[0] * 1.0/sx
    frqy = frq[1] * 1.0/sy

    ys = pyl.arange(sy)
    xs = pyl.arange(sx)
    yq, xq = np.meshgrid(ys,xs)
    yq -= amp[1]
    xq -= amp[0]
    xo = amp[0]*np.sin(2*np.pi*yq*frqx + ph)
    inx = (xq + xo)
    yo = amp[1]*np.cos(2*np.pi*inx*frqy + ph)
    iny = (yq + yo)
    X = np.vstack((iny.flatten(), inx.flatten()))
    Final = ndimage.interpolation.map_coordinates(Im, X).reshape(sy,sx)
    return Final

face = msc.imread("images/bauckhage.jpg", flatten=True).astype('float')
clock = msc.imread("images/clock.jpg", flatten=True).astype('float')
trumpas = msc.imread("images/trumpas.jpg", flatten=True).astype('float')

# Generate images for slides
ph = 0
frq = 1.

plt.figure()

# Generate images with different amplitude
for amp in [0, 3, 6, 9, 12, 15]:
    clock_warp = warping(clock, [amp, amp], ph, [3.0, 3.0])
    msc.imsave("resulting_images/warps/clock_amp_{}.png".format(amp, amp), clock_warp)
    #plt.imshow(clock_warp, cmap="Greys_r")
    #plt.show()

# Generate images with different amplitude in x and y
amplitudx = [10, 4, 30]
amplitudy = [0, 30, 4]
for ampx in amplitudx:
    for ampy in amplitudy:
        clock_warp = warping(clock, [ampx,ampy], ph, [frq, frq])
        msc.imsave("resulting_images/warps/clock_amp_{}_{}.png".format(ampx,ampy), clock_warp)
        #plt.imshow(clock_warp, cmap="Greys_r")
        #plt.show()

# Generate images with different phase
amp = 9
phases = [10, 20, 40, 60, 80, 100]
for ph in phases:
    clock_warp = warping(clock, [amp, amp], ph, [frq, frq])
    msc.imsave("resulting_images/warps/clock_ph_{}.png".format(ph), clock_warp)
    #plt.imshow(clock_warp, cmap="Greys_r")
    #plt.show()

# Generate images with different frequency
amp = 9
freqs = [2, 5, 10, 15, 20, 30]
for frq in freqs:
    clock_warp = warping(clock, [amp, amp], ph, [frq, frq])
    msc.imsave("resulting_images/warps/clock_fr_{}.png".format(frq), clock_warp)
    #plt.imshow(clock_warp, cmap="Greys_r")
    #plt.show()


# Combination 1
ampx, ampy, = 10, 15
frx, fry, = 10.5, 1.
ph = 0.5
clock_warp = warping(trumpas.T, [ampx, ampy], ph, [frx, fry])
msc.imsave("resulting_images/warps/combination_1.png".format(ampx, ampy, frx, fry), clock_warp)
#plt.imshow(clock_warp, cmap="Greys_r")
#plt.show()

# Combination 2
ampx, ampy, = 100, 0
frx, fry, = 1., 0.
ph = 4.666
clock_warp = warping(trumpas.T, [ampx, ampy], ph, [frx, fry])
msc.imsave("resulting_images/warps/combination_2.png".format(ampx, ampy, frx, fry), clock_warp)
#plt.imshow(clock_warp, cmap="Greys_r")
#plt.show()
