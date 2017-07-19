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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random

IMG_SIZE = 2511
label_dict = {0: 'Background', 1: 'Car'}
TEST_DIR = 'uiuc/test'
TRAIN_DIR = 'uiuc/train1'



def draw(vector, shape):
    plt.figure()
    plt.imshow(np.reshape(vector, shape), 'gray')
    plt.axis('off')
    plt.show()

def plotData(X_lda,label):
    plt.clf()
    ax = plt.subplot(222)
    for lab, marker, color in zip(range(2), ('*', '^'), ('blue', 'red')):
        plt.scatter(x=X_lda[label == lab],
                    y=X_lda[label == lab],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[lab]
                    )
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA')
    plt.xlim(X_lda[0].min() - 0.01, X_lda[0].max() + 0.01)
    plt.ylim(X_lda[0].min() - 0.01, X_lda[0].max() + 0.01)
    plt.grid()
    #plt.tight_layout()
    plt.show()

def readData(dirName):
    for root, _, files in os.walk(dirName):
        data = np.zeros(shape=(len(files),IMG_SIZE))
        counter =0
        for filename in files:
            path = os.path.join(root, filename)
            image = misc.imresize(misc.imread(path, flatten=True).astype("float"),size=(81,31)).ravel()
            data[counter] = image
            counter+=1
    data = preprocessing.normalize(data)
    label = np.char.array(files).rfind('Pos')
    label[np.where(label == -1)] = 0  # neg class
    label[np.where(label > 0)] = 1  # pos class
    return data,label

def saveImage(imgData,filename):
    misc.imsave(filename,imgData)

def readTrainingData():
    return readData(TRAIN_DIR)
def readTestData():
    return readData(TEST_DIR)


def getSBMatrix(mean_vec,overall_mean,dataset):
    S_B = np.zeros((dataset.shape[1], dataset.shape[1]))
    for i, mean_vec in enumerate(means):
        n = dataset[np.where(labels == i)].shape[0]
        mean_vec = mean_vec.reshape(IMG_SIZE, 1)  # make column vector
        overall_mean = overall_mean.reshape(IMG_SIZE, 1)  # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B

def getSWMatrix(means,dataset):
    S_W = np.zeros((dataset.shape[1],dataset.shape[1]))
    for i, mean_vec in enumerate(means):
        class_scatter = np.zeros(S_W.shape)
        for row in dataset[np.where(labels == i)]:
            row, mean_vec = row.reshape(IMG_SIZE, 1), mean_vec.reshape(IMG_SIZE, 1)  # make column vectors
            class_scatter += (row - mean_vec).dot((row - mean_vec).T)
        S_W += class_scatter  # sum class scatter matrices
    return S_W

def getProjector(S_B,S_W):
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    #W = np.hstack((eig_pairs[0][1].reshape(IMG_SIZE, 1), eig_pairs[1][1].reshape(IMG_SIZE, 1)))
    W = np.hstack((eig_pairs[0][1].reshape(IMG_SIZE, 1)))
    return W.real


def evaluateClassifier(classifier,data,W,labels):
    tp=0
    fp=0
    tn=0
    fn=0
    projectedLabels = []

    for item in data[labels==1]: #check for pos
        if np.dot(W, item) >= classifier:
            prediction = 1
        else:
            prediction = 0

        if(prediction!=1):
            fp+=1
        else:
            tp+=1
        projectedLabels.append(prediction)

    for item in data[labels==0]: #check for neg
        if np.dot(W, item) >= classifier:
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
    dataset,labels = readTrainingData()

    #calculate means for two classes
    means = []
    means.append(np.mean(dataset[np.where(labels==0)],axis=0));
    means.append(np.mean(dataset[np.where(labels==1)],axis=0));

    #calculate overall mean
    overall_mean = np.mean(dataset, axis=0)

    # calculate within class covariance matrix
    S_W = getSWMatrix(means, dataset)
    draw(S_W,S_W.shape)

    #calculate between class covariance matrix
    S_B = getSBMatrix(means,overall_mean,dataset)
    draw(S_B,S_B.shape)


    #calculate projector matrix
    W = getProjector(S_B,S_W)
    draw(W,(81,31))


    #read new data
    newData,labels = readTrainingData()
    projectedLabels = []

    #X_lda = newData.dot(W)
    X_lda = np.empty(shape=(newData.shape[0],))
    for i,row in enumerate(newData):
        X_lda[i] = row.dot(W)
    print X_lda

    mu =[]
    mu.append(np.dot(W.T, means[0]))
    mu.append(np.dot(W.T, means[1]))
    print mu

    #create k = 1,...,10 different classifier
    cl_total = 10000
    classifiers = []
    for index in range(cl_total):
        theta = mu[0] + abs(float(mu[1] - mu[0])) / (cl_total + 1) * (index + 1)
        classifiers.append(theta)

    bestThreshold = classifiers[0]
    bestPerformance,precision,recall,projectedLabels = evaluateClassifier(bestThreshold,newData,W,labels)
    precisions = np.zeros((len(classifiers),))
    recalls = np.zeros((len(classifiers)))
    for i,threshold in enumerate(classifiers):
        print("Threshold = ",threshold)
        performance,precision,recall,projectedLabels = evaluateClassifier(threshold,newData,W,labels)
        precisions[i] = precision
        recalls[i] = recall
        print("Performance = ",performance)
        print("Precision = ", precision)
        print("Recall = ", recall)

        if(performance>bestPerformance):
            bestPerformance = performance
            bestThreshold = threshold

    print('Best Threshold = ',bestThreshold)
    print("Best Performance = ",bestPerformance)
    print("Projected labels: ")
    print projectedLabels

    #plot output
    plotData(X_lda, labels)

    # plot W
    plt.plot(W)
    plt.grid()
    plt.show()

    plt.clf()
    plt.plot(recalls, precisions, lw=2, color='navy',
         label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.xlim(precisions.min()-0.5,recalls.max()+0.5)
    plt.ylim(precisions.min()-0.5,recalls.max()+0.5)
    plt.grid()
    plt.show()


