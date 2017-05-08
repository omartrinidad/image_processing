from __future__ import unicode_literals
import numpy as np
import numpy.fft as fft
from mpl_toolkits.mplot3d import axes3d
import scipy.misc as msc
from functions import *
import time as time
import scipy.ndimage as ndimg
import matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt

def task4():
    """
    """
    pass


def task3(image):
    """
    """

    sizes = []
    sigmas = [((size-1)/2)/2.575 for size in sizes]

    sigma_66 = np.ndarray(len(sigmas))
    sigma_266 = np.ndarray(len(sigmas))
    sigma_366 = np.ndarray(len(sigmas))
    sigma_566 = np.ndarray(len(sigmas))
    times = np.ndarray(len(sigmas))

    sigmas = [0.6, 1.6, 2.6, 3.6]
    for s in sigmas:
        response = recursive_filter(image, sigma=s)
        final = recursive_filter(response.T, sigma=s)
        s = str(s).replace('.','')
        msc.imsave("task3_{}_.png".format(s), final.T)

    # sigma 0.666
    """
    for i in range(len(sigmas)):
        tick = time.time()
        recursive_filter(image, sigma=0.666)
        toc = time.time()
        sigma_66[i] = toc-tick

    # sigma 2.666
    for i in range(len(sigmas)):
        tick = time.time()
        recursive_filter(image, sigma=2.666)
        toc = time.time()
        sigma_266[i] = toc-tick

    # sigma 3.666
    for i in range(len(sigmas)):
        tick = time.time()
        recursive_filter(image, sigma=3.666)
        toc = time.time()
        sigma_366[i] = toc-tick

    # sigma 5.666
    for i in range(len(sigmas)):
        tick = time.time()
        recursive_filter(image, sigma=5.666)
        toc = time.time()
        sigma_566[i] = toc-tick

    fig = plt.figure()

    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot(sizes, np.log(sigma_66*1000), 'k--', lw=2, label='sig 0.666')
    ax.plot(sizes, np.log(sigma_266*1000), 'k:', lw=2, label='sig 2.666')
    ax.plot(sizes, np.log(sigma_366*1000), 'r--', lw=2, label='sig 3.666')
    ax.plot(sizes, np.log(sigma_566*1000), 'r:', lw=2, label='sig 5.666')

    ax.set_xlabel('Sigma', size=22)
    ax.set_ylabel('Log(time*1000)', size=22)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("resulting_images/task3plot.png", bbox_inches='tight')
    plt.show()
    """


def task2( img, filter2d, name="image", use_sys=False, show=False ):
    """
    """

    s1 = filter2d.shape[0]
    s2 = filter2d.shape[1]

    if(use_sys):
        dx, dy = np.gradient(filter2d)
        new_imgx = ndimg.convolve(img, dx)
        new_imgy = ndimg.convolve(img, dy)
        fig_name = "resulting_images/sys_task2_{}_{}x{}.png".format(name, filter2d.shape[0], filter2d.shape[1])
    else:
        new_fil = discreteDerivative(filter2d)
        dx = new_fil[:,:,0]
        dx = dx / dx.sum()
        dy = new_fil[:,:,1]
        dy = dy / dy.sum()

        new_imgx = naiveConvolve2dOptimized(img, dx)
        new_imgy = naiveConvolve2dOptimized(img, dy)


        fig_name = "resulting_images/task2_{}_{}x{}.png".format(name, filter2d.shape[0], filter2d.shape[1])

    msc.imsave("{}x{}_{}_x.png".format(s1, s2, name), np.abs(new_imgx))
    msc.imsave("{}x{}_{}_y.png".format(s1, s2, name), np.abs(new_imgy))

    fig = plt.figure()

    fig.add_subplot(2,2,1).set_title("Original")
    plt.axis("off")
    plt.imshow(img, cmap="Greys_r")

    fig.add_subplot(2,2,2).set_title("$|\Delta x|$")
    plt.axis("off")
    plt.imshow(np.abs(new_imgx), cmap="Greys_r")

    fig.add_subplot(2,2,3).set_title("$|\Delta y|$")
    plt.axis("off")
    plt.imshow(np.abs(new_imgy), cmap="Greys_r")

    new_img = np.sqrt(np.add(np.square(new_imgx), np.square(new_imgx)))
    msc.imsave("{}x{}_{}_xy.png".format(s1, s2, name) , np.abs(new_img))

    fig.add_subplot(2,2,4).set_title("$||f \cdot \Delta (g)||$")
    plt.axis("off")
    plt.imshow(new_img, cmap="Greys_r")

    # fig.add_subplot(2,2,1).set_title("original")
    # plt.imshow(img, cmap="Greys_r")
    plt.savefig(fig_name, bbox_inches='tight')
    if(show):
        plt.show()
    return new_img


def task1(img):
    """
    """
    sizes = [3, 5, 21]
    #sizes = range(3,22,2)
    sigmas = [((size-1)/2)/2.575 for size in sizes]
    nc2 = np.ndarray(len(sizes))
    nc2o = np.ndarray(len(sizes))
    nc1 = np.ndarray(len(sizes))
    nc1o = np.ndarray(len(sizes))
    cfft = np.ndarray(len(sizes))
    sysc = np.ndarray(len(sizes))
    for i in range(len(sizes)):
        # G, gx, gy = createGaussianFilter(sigma=sigmas[len(sigmas) - 1])
        figure = plt.figure("{}x{}, sig={:.3f}".format(sizes[i],sizes[i],sigmas[i]))
        #print("{}x{}, sig={:.3f}".format(sizes[i],sizes[i],sigmas[i]))
        figure.add_subplot(2,3,1).set_title("Original")
        plt.axis("off")
        plt.imshow(img, cmap="Greys_r")
        #print("-----------------------------------------")
        G, gx, gy = createGaussianFilter(sigma=sigmas[i])
        tick = time.time()
        new_img = convolve1d(img.T, gx)
        new_img = convolve1d(new_img.T, gy)
        toc = time.time()
        nc1[i] = toc-tick
        figure.add_subplot(2,3,2).set_title("Naive Con 1D")
        plt.axis('off')
        plt.imshow(new_img, cmap="Greys_r")
        #print("Time: ", nc1[i])
        tick = time.time()
        new_img = convolve1dOptimized(img.T, gx)
        new_img = convolve1dOptimized(new_img.T, gy)
        toc = time.time()
        nc1o[i] = toc-tick
        #print("Time: ", nc1o[i])
        tick = time.time()
        new_img = naiveConvolve2d(img, G)
        toc = time.time()
        nc2[i] = toc-tick
        figure.add_subplot(2,3,3).set_title("Naive Con 2D")
        plt.axis('off')
        plt.imshow(new_img, cmap="Greys_r")
        #print("Time: ", nc2[i])
        tick = time.time()
        new_img = naiveConvolve2dOptimized(img, G)
        toc = time.time()
        nc2o[i] = toc-tick
        figure.add_subplot(2,3,4).set_title("Naive Con 2D Opt")
        plt.axis('off')
        plt.imshow(new_img, cmap="Greys_r")
        #print("Time: ", nc2o[i])
        tick = time.time()
        new_img = convolve2dFFT(img, (sizes[i],sizes[i]))
        toc = time.time()
        cfft[i] = toc-tick
        figure.add_subplot(2,3,5).set_title("ConFFT")
        plt.axis('off')
        plt.imshow(new_img, cmap="Greys_r")
        #print("Time: ", cfft[i])
        tick = time.time()
        new_img = ndimg.convolve(img, G, mode="constant", cval=0.0)
        toc = time.time()
        sysc[i] = toc-tick
        figure.add_subplot(2,3,6).set_title("SciPy Con")
        plt.axis('off')
        plt.imshow(new_img, cmap="Greys_r")
        #print("Time: ", sysc[i])
        #print("-----------------------------------------")
        plt.savefig("resulting_images/{}x{}, sig={:.3f}.png".format(sizes[i],sizes[i],sigmas[i]), bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot(sizes, np.log(nc2*1000), 'r--', lw=2, label=r'\textbf{Naive convolution 2D}')
    ax.plot(sizes, np.log(nc1*1000), 'k--', lw=2, label=r'\textbf{Naive convolution 1D}')
    ax.plot(sizes, np.log(nc2o*1000), 'r:', lw=2, label=r'\textbf{Naive convolution 2D (optimized)}')
    ax.plot(sizes, np.log(nc1o*1000), 'k:', lw=2, label=r'\textbf{Naive convolution 1D (optimized)}')
    ax.plot(sizes, np.log(sysc*1000), 'g', lw=2,  label=r'\textbf{Scipy implementation}')
    ax.plot(sizes, np.log(cfft*1000), 'b', lw=2,  label=r'\textbf{Convolution with FFT}')
    ax.set_xlabel(r'\textbf{Filter size}', size=14)
    ax.set_ylabel(r'$log(time \cdot 1000)$', size=14)

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0., fontsize=12)
    #draw.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.0)

    plt.savefig("resulting_images/task1plots.png".format(sizes[i],sizes[i],sigmas[i]), bbox_inches='tight')
    plt.show()

def main():

    # Load images
    face = readIntensityImage("images/bauckhage.jpg")
    clock = readIntensityImage("images/clock.jpg")

    # Run Task 1
    # task1(face)

    # Run Task 2
    """
    sizes = range(3,22,2)
    images = ["bauckhage.jpg", "clock.jpg"]
    for image in images:
         for size in sizes:
             task2(
                 readIntensityImage("images/{}".format(image)),
                 createGaussianFilterBySize( size=(size,size) )[0],
                 name=image.split(".")[0],
                 use_sys=True,
                 show=True
                 )
    """
    # Run Task 3
    task3(face)
    """
    task3(clock)
    """


main()
