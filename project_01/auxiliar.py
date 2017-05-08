################################################################################
# Auxiliar functions
################################################################################

import matplotlib.pyplot as plt
import scipy.misc as misc
import os
import re


def show_histo(image):
    """Given an image show the corresponding histogram"""
    plt.hist(image.ravel(), bins=100,  normed=1, facecolor='blue', alpha=0.5)
    plt.hist(intensities, bins=100, histtype='step')
    plt.title("Histogram")
    plt.xlabel("Pixel values")
    plt.ylabel("Frequencies")
    plt.show()


def load_images():
    """Load all the testing images in a dictionary"""
    images = {}
    for root, _, files in os.walk('images'):
        for filename in files:
            if re.match('([-\w]+\.(?:jpg|gif|png))', filename):
                print(filename)
                path = os.path.join(root, filename) 
                val = misc.imread(path, flatten=True).astype("float")
                key = re.search('^[-\w]+', filename).group(0)
                images[key] = val
    return images
