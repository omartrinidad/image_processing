
import tarfile
import numpy as np
import os
from scipy import misc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random

IMG_SIZE = 2511
label_dict = {0: 'Background', 1: 'Car'}
TEST_DIR = 'uiuc/test'
TRAIN_DIR = 'uiuc/train1'

def readData(dirName):
    for root, _, files in os.walk(dirName):
        data = np.zeros(shape=(len(files),81, 31))
        counter = 0
        for filename in files:
            path = os.path.join(root, filename)
            image = misc.imresize(misc.imread(path, flatten=True).astype("float"),size=(81,31))
            image = preprocessing.normalize(image)
            data[counter] = image
            counter+=1

    label = np.char.array(files).rfind('Pos').astype(np.float32)
    posClass = abs(1/float(len(files)))
    negClass = abs(1/float(len(files)))*-1
    label[np.where(label > 0)] = posClass
    label[np.where(label == -1)] =  negClass
    return data,label

def saveImage(imgData,filename):
    misc.imsave(filename,imgData)

def readTrainingData():
    return readData(TRAIN_DIR)
def readTestData():
    return readData(TEST_DIR)


def Gram_Schmidt(vecs, row_wise_storage=True, tol=1E-10):
    vecs = np.asarray(vecs)  # transform to array if list of vectors
    if row_wise_storage:
        A = np.transpose(vecs).copy()
    else:
        A = vecs.copy()

    m, n = A.shape
    V = np.zeros((m, n))

    for j in xrange(n):
        v0 = A[:, j]
        v = v0.copy()
        for i in xrange(j):
            vi = V[:, i]

            if (abs(vi) > tol).any():
                v -= (np.vdot(v0, vi) / np.vdot(vi, vi)) * vi
        V[:, j] = v

    return np.transpose(V) if row_wise_storage else V



def evaluateClassifier(classifier,newData,W,labels,projectedLabels):
    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(newData.shape[0]):
        img = newData[i]
        pp = img*W
        px = (np.mean(pp) - np.var(pp))*1000.0
        print px
        if(px<classifier):
            projectedLabels[i] = abs(1/float(len(labels)))*-1
        else:
            projectedLabels[i] = abs(1/float(len(labels)))

    print projectedLabels

    for i in range(len(labels)):
        if(projectedLabels[i]==1 and labels[i]==1):
            tp+=1

        if(projectedLabels[i]==1 and labels[i]==0):
            fp+=1

        if(projectedLabels[i]==0 and labels[i]==0):
            tn+=1
        if(projectedLabels[i]==0 and labels[i]==1):
            fn+=1

    precision = tp/float(tp+fp) if (tp+fp)>0 else 0
    recall = tp/float(tp+fn) if (tp+fn)>0 else 0

    divisor = float(tp+tn+fp+fn) if float(tp+tn+fp+fn)>0 else 1
    return (tp+tn)/divisor,precision,recall






if __name__ == '__main__':
    dataset, labels = readTrainingData()


    us = np.zeros(shape=(dataset.shape[0], dataset.shape[1]))
    vs = np.zeros(shape=(dataset.shape[0], dataset.shape[2]))
    for r in range(dataset.shape[0]):

        u = np.random.rand(dataset.shape[1])
        v = np.random.rand(dataset.shape[2])
        t = 1
        tmax = 79
        epsilon = 0.001

        while True:
            # compute u contraction
            contractionU = np.empty((dataset.shape[0], 31))
            for l in range(dataset.shape[0]):
                img = dataset[l]
                contractionU[l] = np.dot(u, img)

            v1 = np.linalg.inv(np.dot(contractionU.T, contractionU))
            v2 = np.dot(contractionU.T, labels)
            v_temp = np.dot(v1, v2)
            t_st_v = np.vstack((v, v_temp))
            v_gs = Gram_Schmidt(t_st_v)
            v_temp = v_gs[-1]
            v = preprocessing.normalize(v_temp)[0]

            # compute v contraction
            contractionV = np.empty((dataset.shape[0], 81))
            for l in range(dataset.shape[0]):
                img = dataset[l]
                contractionV[l] = np.dot(img, v)
            u1 = np.linalg.inv(np.dot(contractionV.T, contractionV))
            u2 = np.dot(contractionV.T, labels)
            u_temp = np.dot(u1, u2)
            t_st_u = np.vstack((u, u_temp))
            u_gs = Gram_Schmidt(t_st_u)
            u_temp = u_gs[-1]
            u = preprocessing.normalize(u_temp)[0]

            t = t + 1
            check = abs(u[t] - u[t - 1])
            print('check = ', check, ': t=', t)
            if (check < epsilon) or (t > tmax):
                break
        us[r] = u
        vs[r] = v

    W = np.empty(shape=(dataset.shape[1], dataset.shape[2]))
    for r in range(dataset.shape[0]):
        W += np.outer(us[r],vs[r])

    misc.imsave('tensorWOut.jpg',W)

    # read new data
    newData, labels = readTrainingData()
    projectedLabels = np.zeros(newData.shape[0])


    # create k = 1,...,10 different classifier
    classifiers = np.random.uniform(-10,10,10)
    bestThreshold = classifiers[0]
    bestPerformance, precision, recall = evaluateClassifier(bestThreshold,newData,W, labels, projectedLabels)
    precisions = []
    recalls = []
    for threshold in classifiers:
        print("Threshold = ",threshold)
        performance,precision,recall = evaluateClassifier(threshold,newData,W,labels,projectedLabels)
        precisions.append(precision)
        recalls.append(recall)
        print("Performance = ",performance)
        if(performance>bestPerformance):
            bestPerformance = performance
            bestThreshold = threshold

    print('Best Threshold = ',bestThreshold)
    print("Best Performance = ",bestPerformance)
    plt.clf()
    plt.plot(recalls, precisions, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.grid()
    plt.tight_layout()
    plt.show()



    print W
    # plot W
    plt.plot(W)
    plt.grid()
    plt.tight_layout()
    plt.show()

