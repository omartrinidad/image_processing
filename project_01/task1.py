################################################################################
# Task 1.1
# Lloyd Max algorithm for gray value quantization
################################################################################

import numpy as np
import scipy.misc as misc
from scipy import ndimage
from auxiliar import load_images, show_histo

# load a list with the next images: clock, cat, portrait, face, asterixGrey,
# bauckhage, bauckhage-gamma-1, and bauckhage-gamma-2
images = load_images()


def lloyd_max(image, levels=8, stop_condition='max_iterations', max_iterations=200):
    """Implement the Lloyd Max quantization algorithm"""
    # get the intensity histogram
    intensities = np.histogram(image, bins=np.arange(255))[0]
    # get density function
    # ToDo
    # Initialize boundaries
    av = np.arange(levels+1) * 256/levels
    # Initialize quantization points
    bv = (np.arange(levels+1) * 256/levels) + 256/(2*levels)
    # iterate the equations
    # ToDo
    # implement stop condition
    # ToDo


# apply the program to a bright and to a dark image
lloyd_max(images['portrait'])
