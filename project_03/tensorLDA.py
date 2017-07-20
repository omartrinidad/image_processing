
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
TEST_DIR = 'uiuc/test1'
TRAIN_DIR = 'uiuc/train1'


def draw(vector, shape, name):
    plt.figure()
    plt.imshow(np.reshape(vector, shape), 'gray')
    plt.axis('off')
    plt.savefig(name)
    plt.show()


def readData(dirName):
    for root, _, files in os.walk(dirName):
        data = np.zeros(shape=(len(files),81, 31))
        counter = 0
        for filename in files:
            if (filename.find("DS_Store") == 1):
                continue
            path = os.path.join(root, filename)
            image = misc.imresize(misc.imread(path, flatten=True).astype("float"),size=(81,31))
            data[counter] = preprocessing.normalize(image)
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

    for j in range(n):
        v0 = A[:, j]
        v = v0.copy()
        for i in range(j):
            vi = V[:, i]

            if (abs(vi) > tol).any():
                v -= (np.vdot(v0, vi) / np.vdot(vi, vi)) * vi
        V[:, j] = v

    return np.transpose(V) if row_wise_storage else V



def evaluateClassifier(classifier,data,W,labels):
    tp=0
    fp=0
    tn=0
    fn=0
    projectedLabels = []

    for item in data[labels==labels.max()]: #check for pos
        if np.dot(W.ravel().T,item.ravel()) >= classifier:
            prediction = 1
        else:
            prediction = 0

        if(prediction!=1):
            fp+=1
        else:
            tp+=1
        projectedLabels.append(prediction)

    for item in data[labels==labels.min()]: #check for neg
        if np.dot(W.ravel().T,item.ravel()) >= classifier:
            prediction = 1
        else:
            prediction = 0

        if (prediction != 0):
            fn += 1
        else:
            tn += 1
        projectedLabels.append(prediction)

    precision = tp/float(tp+fp) if (tp+fp)>0 else 0
    recall = tp/float(tp+fn) if (tp+fn)>0 else 0

    return (tp+tn)/float(tp+tn+fp+fn),precision,recall,projectedLabels







if __name__ == '__main__':
    dataset, labels = readTrainingData()

    # calculate means for two classes
    means = []
    means.append(np.mean(dataset[labels==labels.min()],axis=0));
    means.append(np.mean(dataset[labels==labels.max()],axis=0));

    # calculate overall mean
    overall_mean = np.mean(dataset, axis=0)

    p = 100
    us = np.zeros(shape=(p, dataset.shape[1]))
    vs = np.zeros(shape=(p, dataset.shape[2]))
    
    for r in range(p):

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

    draw(W, (81, 31))
    #misc.imsave('tensorWOut.jpg',W)

    mu = []
    mu.append(np.dot(W.ravel().T, means[0].ravel()))
    mu.append(np.dot(W.ravel().T, means[1].ravel()))
    print(mu)

    # create k = 1,...,10 different classifier
    cl_total = 10000
    classifiers = []
    for index in range(cl_total):
        theta = mu[0] + abs(float(mu[1] - mu[0])) / (cl_total + 1) * (index + 1)
        classifiers.append(theta)
    bestThreshold = classifiers[0]

    bestPerformance, precision, recall,projectedLabels = evaluateClassifier(bestThreshold,dataset,W, labels)
    precisions = np.zeros((len(classifiers),))
    recalls = np.zeros((len(classifiers)))
    for i,threshold in enumerate(classifiers):
        #print("Threshold = ",threshold)
        performance,precision,recall,projectedLabels = evaluateClassifier(threshold,dataset,W,labels)
        precisions[i] = precision
        recalls[i] = recall
        #print("Performance = ", performance)
        #print("Precision = ", precision)
        #print("Recall = ", recall)
        if(performance>bestPerformance):
            bestPerformance = performance
            bestThreshold = threshold

    print('Best Threshold = ', bestThreshold)
    print("Best Performance = ", bestPerformance)
    print("Projected labels: ")
    print(projectedLabels)
    print("Recalls:")
    print(recalls)
    print("Precisions: ")
    print(precisions)

    plt.clf()
    plt.plot(recalls, precisions, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.xlim(precisions.min() - 0.5, recalls.max() + 0.5)
    plt.ylim(precisions.min() - 0.5, recalls.max() + 0.5)
    plt.grid()
    plt.ti1ght_layout()
    plt.savefig("tensor_precision_recall.png")
    plt.show()


    print(W)
    # plot W
    plt.plot(W)
    plt.grid()
    plt.tight_layout()
    plt.savefig("tensor_W.png")
    plt.show()

    # check on test data
    newData, labels = readTestData()
    projectedLabels = []
    for item in newData:  # check for pos
        if np.dot(W.ravel().T,item.ravel()) >= bestThreshold:
            prediction = 1
        else:
            prediction = 0
        projectedLabels.append(prediction)

    print(projectedLabels)
