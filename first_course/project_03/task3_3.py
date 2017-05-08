import scipy as sp
import numpy as np
import pylab as pyl
import scipy.misc as msc
import scipy.ndimage as ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt


def cylinder(Im, r_in):

    dpi = 72
    r_cylinder = int(r_in*dpi);
    (sy,sx) = Im.shape
    Im = np.pad(Im,(0,r_cylinder), mode="constant")[:,:sx]

    # msc.imshow(Im)

    (sy,sx) = Im.shape

    th_range = (pyl.pi)*2
    max_r = sy
    fy = 2*max_r
    fx = 2*max_r
    ys = fy/2 - pyl.arange(fy)
    xs = fx/2 - pyl.arange(fx)
    xq, yq = np.meshgrid(ys,xs)
    rs = pyl.sqrt(yq*yq+xq*xq)
    rsmx = rs < max_r
    ths = pyl.arctan2(yq,xq)
    thsmx = ths < th_range
    inx = abs(sx - ths * (sx/th_range)+1)%(sx-1)
    inx[~thsmx] = sx*2
    iny = abs(sy - rs)
    iny[~rsmx] = sy*2
    ow(iny.flatten())
    X = np.vstack((iny.flatten(),inx.flatten()))
    final = ndimage.interpolation.map_coordinates(Im,X).reshape(fy,fx)[::-1,:]
    return final


face = msc.imread("images/bauckhage.jpg", flatten=True).astype('float')
clock = msc.imread("images/clock.jpg", flatten=True).astype('float')
abbey = msc.imread("images/long.jpg", flatten=True).astype('float')

plt.figure()

msc.imsave("resulting_images/original.png", abbey)
result = cylinder(abbey, 0.5)
#plt.imshow(result, cmap="Greys_r")
#msc.imsave("resulting_images/zero.png", result)
#plt.show()

"""
result = cylinder(abbey, 3.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("resulting_images/donuts_1.png", result)
plt.show()

result = cylinder(abbey, 6.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("resulting_images/donuts_2.png", result)
plt.show()

result = cylinder(clock, 0.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("resulting_images/czero.png", result)
plt.show()

result = cylinder(clock, 3.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("resulting_images/cdonuts_1.png", result)
plt.show()

result = cylinder(clock, 6.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("resulting_images/cdonuts_2.png", result)
plt.show()
"""
