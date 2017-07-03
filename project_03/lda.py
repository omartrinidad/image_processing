#!/usr/bin/env/python
# encoding: utf8
"""
Task 3.1
Linear Discriminant Analysis
Python 3
"""

import tarfile
import numpy as np
import os
from scipy import misc


def readImage(filename):
    """
    read image
    """
    f = misc.imread(filename, flatten=True).astype("float")
    return f


#with tarfile.open("uiuc/uiucTest.tgz", "r:gz") as tar:
#    for entry in tar:
#        img = tar.extractfile(entry).read()
#        img = np.fromiter(img, dtype=np.float64)

dataset = np.empty(shape=(2511,))
labels = np.ones(2511)
for root, _, files in os.walk('uiuc/train'):
    for filename in files:
        path = os.path.join(root, filename) 
        image = misc.imread(path, flatten=True).astype("float")
        dataset = np.vstack((image.ravel(), dataset))

# tricky code
dataset = np.delete(dataset, (-1), axis=0)
label = np.char.array(files).rfind('Pos')
label[label != -1] = 1

# sw = s1 - s2
# solve sw ^ -1 (mu1 - mu2) = , check slide 12
