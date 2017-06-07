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


#LloydMax Algorithm
def LloydAlgorithm(imgData):
    totalIterations = 100
    threshold = 0.01

    #first calculate the intensity histogram
    histogram = np.histogram(imgData, bins=np.arange(257))[0]

    #calculate PDF
    totalSum = histogram.sum()
    probs = np.ndarray(shape=(256,))
    for i in range(len(histogram)):
        probs[i] = histogram[i]/float(totalSum)

    #choose number L of quantization levels
    level = 256

    #initialize the boundaries
    boundaries = np.ndarray(shape=(level+1,))
    for i in range(level+1):
        boundaries[i] = float(i)*256.0/float(level)

    #initialize the quantization points bv
    pointsBv = np.ndarray(shape=(level,))
    for i in range(level):
        pointsBv[i] = float(i)*(256.0/float(level))+(256.0/2.0*float(level))

    #iterate the equations
    error = 0.0
    for i in xrange(level):
        currentBoundary = boundaries[i]
        nextBoundary = boundaries[i+1]
        for j in xrange(int(currentBoundary),int(nextBoundary)):
            error += (j-pointsBv[i])*(j-pointsBv[i])*probs[j]

    iterationCount = 0
    while(iterationCount<totalIterations and error>threshold):
        iterationCount+=1
        #update boundaries
        for i in range(level):
            boundaries[i] = (pointsBv[i]+pointsBv[i-1])/2.0

        #update points
        for i in range(level):
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


    rows,cols = imgData.shape
    newImgData = np.ndarray(shape=(256,256))
    for i in xrange(rows):
        for j in xrange(cols):
            for k in range(level):
                if(imgData[i,j]>=boundaries[k] and imgData[i,j]<boundaries[k+1]):
                    break
            newImgData[i,j] = pointsBv[k]

    writeImage(newImgData,'result.png')




LloydAlgorithm(readImage('../../images/bauckhage-gamma-2.png'))
