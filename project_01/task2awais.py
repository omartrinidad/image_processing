################################################################################
# Task 1.2
# histogram transformations
################################################################################

import scipy.misc as misc
import numpy as np

#read image
def readImage(filename):
    f = misc.imread(filename, flatten=True).astype("float")
    return f

#write image
def writeImage(data,filename):
    misc.toimage(data,cmin=0,cmax=255).save(filename)


def LloydAlgorithm(imgData, levels=2):
    """Implement the Lloyd Max quantization algorithm"""
    totalIterations = 100
    threshold = 0.01

    #first calculate the intensity histogram
    histogram = np.histogram(imgData, bins=np.arange(257))[0]

    # calculate PDF
    probs = histogram/histogram.sum()

    #initialize the boundaries and quantization points
    boundaries = np.arange(levels+1) * 256/levels
    pointsBv = (np.arange(levels) * 256/levels) + 256/(2*levels)

    #iterate the equations
    error = 0.0
    for i in xrange(levels):
        currentBoundary = boundaries[i]
        nextBoundary = boundaries[i+1]
        for j in xrange(int(currentBoundary),int(nextBoundary)):
            error += (j-pointsBv[i])*(j-pointsBv[i])*probs[j]

    iterationCount = 0
    while(iterationCount<totalIterations and error>threshold):
        iterationCount+=1
        #update boundaries
        for i in range(levels):
            boundaries[i] = (pointsBv[i]+pointsBv[i-1])/2.0

        #update points
        for i in range(levels):
            numerator = 0.0
            denominator = 0.0
            currentBoundary = boundaries[i]
            nextBoundary = boundaries[i+1]

            for j in xrange(int(currentBoundary),int(nextBoundary)):
                if(j>probs.size):
                    break
                numerator += j*probs[j]
                denominator += probs[j]
            if(numerator==0 or denominator==0):
                pointsBv[i] = 0.0
            else:
                pointsBv[i] = float(numerator)/float(denominator)


    rows, cols = imgData.shape
    newImgData = np.ndarray(shape=(rows, cols))
    for i in xrange(rows):
        for j in xrange(cols):
            for k in range(levels):
                if(imgData[i,j]>=boundaries[k] and imgData[i,j]<boundaries[k+1]):
                    break
            newImgData[i,j] = pointsBv[k]

    return newImgData



image = readImage('images/bauckhage-gamma-2.png')
result = LloydAlgorithm(image, levels=4)
misc.imshow(result)
writeImage(result,'result.png')
