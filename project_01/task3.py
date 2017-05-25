################################################################################
# Task 1.3
# Illumination Compensation
################################################################################

import numpy as np
import scipy.misc as misc
from auxiliar import load_images
import matplotlib.pyplot as plt


# load the images in the global scope
images = load_images()


def illumination():
    """Implement ..."""
    bauckhage = images['bauckhage']
    # misc.imshow(bauckhage)
    plt.imshow(bauckhage)
    plt.show()

illumination()
