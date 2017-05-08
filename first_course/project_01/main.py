
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.misc as msc
from functions import *


def task4():

    board = readIntensityImage("images/bauckhage.jpg")
    texture = readIntensityImage("images/cat.png")

    # combination of face and face gill give exactly the face image!
    k = merge_images(board, board)
    msc.imshow(k)

    k = merge_images(board, texture)
    msc.imshow(k)

    k = merge_images(texture, board)
    msc.imshow(k)

    k = merge_images(texture, texture)
    msc.imshow(k)


def task3():

    image = readIntensityImage("images/bauckhage.jpg")
    image = readIntensityImage("images/clock.jpg")

    # fourier transform
    image_ft = fft.fft2(image)
    fftshift = fft.fftshift(image_ft)
    result = np.abs(fftshift)
    # misc.imshow(np.log(result))

    fig = plt.figure()

    # test for different values r_min and r_max
    i = 0
    for a, b in zip([(0, 50)], [(170, 200)]):

        i += 1
        # supression of frequencies
        temp = draw_donut(fftshift, a[0], a[1])
        temp = draw_donut(temp, b[0], b[1])

        e = fig.add_subplot(1, 2, i)
        e.set_xticklabels([])
        e.set_yticklabels([])
        e.imshow(np.log(np.abs(temp)), cmap='gray')

        # inverse fourier transform
        inverse_fftshift = fft.ifftshift(temp)
        again = fft.ifft2(inverse_fftshift)
        again = np.abs(again)

        i += 1
        # misc.imshow(again)
        e = fig.add_subplot(1, 2, i)
        e.set_xticklabels([])
        e.set_yticklabels([])
        e.imshow(again, cmap='gray')

    plt.show()


def task2():

    # fig = plt.figure()
    # fourier_transform(plt)
    # plt.show()

    # plot different offset values
    fig = plt.figure()
    for i, offset in enumerate([2., 4., 8., ], 1):
        e = fig.add_subplot(1, 3, i)
        e.set_title( "Offset {}".format(offset))
        fourier_transform(plt, offset = offset)
        e.plot()
    plt.show()

    # plot different amplitude values
    fig = plt.figure()
    for i, amp in enumerate([-3, -2, -1, 1, 2, 3], 1):
        e = fig.add_subplot(1, 6, i)
        e.set_title( "Amplitude {}".format(amp))
        fourier_transform(plt, amplitude = amp)
        e.plot()
    plt.show()

    # plot different frequency values
    fig = plt.figure()
    for i, freq in enumerate([1, 2, 4, 8, 16, 32, 64], 1):
        e = fig.add_subplot(7, 1, i)
        e.set_title( "Frequency {}".format(freq))
        fourier_transform(plt, frequency = freq)
        e.plot()
    plt.show()

    # plot different phase values
    fig = plt.figure()
    for i, phase in enumerate([0, np.pi, np.pi*2, np.pi*3], 1):
        e = fig.add_subplot(4, 1, i)
        e.set_title( "Phase {}".format(phase))
        fourier_transform(plt, phase = phase)
        e.plot()
    plt.show()


def task1():

    image = readIntensityImage("images/cat.png")
    msc.imshow(image)

    # create a copy of the original image
    new_image = np.copy(image)
    new_image = draw_donut(new_image, 80, 200)
    msc.imshow(new_image)


task1()
#task2()
task3()
task4()
