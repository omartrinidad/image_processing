
import numpy as np
import scipy.misc as msc
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.spatial.distance import euclidean
from cmath import sqrt


def readIntensityImage(filename):
    f = msc.imread(filename, flatten=True).astype("float")
    return f


def writeIntensityImage(f, filename):
    msc.toimage(f, cmin=0, cmax=255).save(filename)


def draw_donut(image, r_min, r_max):
    """
    Update all points whose distance is >= r_mix but <= r_max
    """

    w, h = image.shape
    center_coordinate = [w/2, h/2]
    new_image = np.copy(image)

    for i in xrange(w):
        for j in xrange(h):
            dis = euclidean([i, j], center_coordinate)
            if r_min <= dis and dis <= r_max:
                new_image[i, j] = 0.0001

    return new_image


def fourier_transform(
        plt, offset = 1., amplitude = 5.,
        frequency = 600., phase = np.pi):
    """
    Task 1.2, see and understand the value of the parameters of a function
    in the result of Fourier transform
    """

    n = 512
    x = np.linspace(0, 3*np.pi, n)
    f = np.sin(x)

    # plt.plot(x, f, 'k-')
    F = fft.fft(f)

    w = fft.fftfreq(n)
    # plt.plot(w, np.abs(F), 'k-'), warning

    f = offset + amplitude * np.sin(frequency*x + phase)
    F = fft.fft(f)
    w = fft.fftfreq(n)
    plt.plot(w, np.abs(F), 'k-')


def merge_images(g, h):
    """
    Task 1.4, merge amplitude and phase from two images
    """

    # fourier transform
    g_ft = fft.fft2(g)
    h_ft = fft.fft2(h)

    g_fftshift = fft.fftshift(g_ft)
    h_fftshift = fft.fftshift(h_ft)

    # the absolute value of a complex number is the amplitude
    amplitude = np.abs(g_fftshift)
    phase = np.angle(h_fftshift)

    combination = amplitude * np.exp(phase * sqrt(-1) )

    # inverse of combination
    combination = fft.ifftshift(combination)
    combination = fft.ifft2(combination)

    return np.abs(combination)
