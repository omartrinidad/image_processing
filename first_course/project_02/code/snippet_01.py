import numpy as np
import scipy.misc as msc
import scipy.ndimage as img

sigma = 5.
msize = int(np.ceil(sigma * 2.575) * 2 + 1)
x = np.arange(msize)
g = np.exp(-0.5 * ((x-msize/2) / sigma)**2)
g /= g.sum()
msc.imsave("resulting_images/gauss1D.png", g.reshape(1,msize))

# outer
G = np.outer(g,g)
G /= G.sum()
msc.imsave("resulting_images/gauss2D.png", G)

f = msc.imread("images/bauckhage.jpg").astype("float")
f1 = img.convolve(f, G, mode="constant", cval=0.0)
msc.toimage(f1, cmin=0,cmax=255).save("smooth-1.png")
f2 = img.gaussian_filter(f, sigma, mode="constant", cval=0.0)
msc.toimage(f2, cmin=0,cmax=255).save("smooth-2.png")
h = img.convolve1d(f, g, mode="constant", cval=0.0)
f3 = img.convolve1d(h.T, g, mode="constant", cval=0.0)
f3 = f3.T
msc.toimage(f3, cmin=0, cmax=255).save("resulting_images/smooth-3.png")
