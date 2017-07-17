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
import random

IMG_SIZE = 2511
label_dict = {0: 'Background', 1: 'Car'}
TEST_DIR = 'uiuc/test1'
TRAIN_DIR = 'uiuc/train1'

def plotData(X_lda,label):
    ax = plt.subplot(222)
    for lab, marker, color in zip(range(2), ('*', '^'), ('blue', 'red')):
        plt.scatter(x=X_lda[:, 0].real[label == lab],
                    y=X_lda[:, 1].real[label == lab],
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

    plt.grid()
    plt.tight_layout()
    plt.show()

def readData(dirName):
    data = np.empty(shape=(IMG_SIZE,))
    for root, _, files in os.walk(dirName):
        for filename in files:
            path = os.path.join(root, filename)
            image = misc.imread(path, flatten=True).astype("float")
            data = np.vstack((image.ravel(), data))

    label = np.char.array(files).rfind('Pos')
    label[np.where(label == -1)] = 0  # neg class
    label[np.where(label > 0)] = 1  # pos class
    return data,label

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
    S_W = np.zeros(S_B.shape)
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
    W = np.hstack((eig_pairs[0][1].reshape(IMG_SIZE, 1), eig_pairs[1][1].reshape(IMG_SIZE, 1)))
    return W


def evaluateClassifier(classifier,data,labels):
    tp=0
    fp=0
    tn=0
    fn=0
    for row in enumerate(X_lda):
        if(row[1][0]>=classifier):
            tp+=1
        else:
            fn+=1

        if(row[1][1]>=classifier):
            tn+=1
        else:
            fp+=1
    precision = tp/float(tp+fp) if (tp+fp)>0 else 0
    recall = tp/float(tp+fn) if (tp+fn)>0 else 0

    return (tp+tn)/float(tp+tn+fp+fn),precision,recall













if __name__ == '__main__':
    dataset,labels = readTrainingData()
    dataset = np.delete(dataset, (-1), axis=0)

    #calculate means for two classes
    means = []
    means.append(np.mean(dataset[np.where(labels==0)],axis=0));
    means.append(np.mean(dataset[np.where(labels==1)],axis=0));

    #calculate overall mean
    overall_mean = np.mean(dataset, axis=0)


    #calculate between class covariance matrix
    S_B = getSBMatrix(means,overall_mean,dataset)

    #calculate within class covariance matrix
    S_W = getSWMatrix(means,dataset)

    #calculate projector matrix
    W = getProjector(S_B,S_W)
    #plot W
    plt.plot(W)
    plt.grid()
    plt.tight_layout()
    plt.show()


    #read new data
    newData,labels = readTrainingData()
    newData = np.delete(newData, (-1), axis=0)
    projectedLabels = []

    #X_lda = newData.dot(W)
    X_lda = np.empty(shape=(newData.shape[0],W.shape[1]))
    for i,row in enumerate(newData):
        X_lda[i][0] = row.dot(W[:, 0].real)
        X_lda[i][1] = row.dot(W[:, 1].real)

    plotData(X_lda, labels)

    #create k = 1,...,10 different classifier
    classifiers = [0.0,-0.1,-0.015,-0.02,0.5,-0.4,-0.5,0.66,-0.7,-0.73]
    bestThreshold = classifiers[0]
    bestPerformance = evaluateClassifier(bestThreshold,X_lda,labels)
    precisions = []
    recalls = []
    for threshold in classifiers:
        print("Threshold = ",threshold)
        performance,precision,recall = evaluateClassifier(threshold,X_lda,labels)
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


