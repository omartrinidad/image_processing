################################################################################
# Task 1.2
# Illumination Compensation
################################################################################

import numpy as np
import scipy.misc as misc
from auxiliar import load_images

# load the images in the global scope
images = load_images()


def illumination():
    """Implement ..."""
    bauckhage = images['bauckhage']
    misc.imshow(bauckhage)


illumination()
