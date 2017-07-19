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
TRAIN_DIR = 'uiuc/train'


def plotData(X_lda, label):
    """
    Plot result from LDA
    """
    
    plt.clf()
    for lab, marker, color in zip(range(2), ('*', '^'), ('blue', 'red')):
        plt.scatter(x=X_lda[0].real[label == lab],
                    y=X_lda[1].real[label == lab],
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
    plt.savefig("lda.png")
    plt.show()


def get_sample(dataset, labels, size):
    """
    """
    neg = dataset[np.where(labels==0)][0:size]
    pos = dataset[np.where(labels==1)][0:size]
    ds = np.vstack((pos, neg))
    labels = np.hstack((np.zeros(size), np.ones(size)))

    return ds, labels


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
    W = np.hstack((eig_pairs[0][1].reshape(IMG_SIZE, 1), eig_pairs[1][1].reshape(IMG_SIZE, 1)))
    return W.real


def evaluateClassifier(classifier,data,labels):
    tp=0
    fp=0
    tn=0
    fn=0
    projectedLabels = []
    for i in range(len(data[0])):
        currentValue = (data[0][i]+data[1][i])/2.0
        if(currentValue>=classifier):
            projectedLabels.append(1) #pos class
        else:
            projectedLabels.append(0)   #neg class

    for i in range(len(data[0])):
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

    return (tp+tn)/float(tp+tn+fp+fn),precision,recall,projectedLabels


if __name__ == '__main__':
    dataset, labels = readTrainingData()
    dataset, labels = get_sample(dataset, labels, 10)

    #calculate means for two classes
    means = []
    means.append(np.mean(dataset[np.where(labels==0)],axis=0));
    means.append(np.mean(dataset[np.where(labels==1)],axis=0));

    #calculate overall mean
    overall_mean = np.mean(dataset, axis=0)

    # calculate within class covariance matrix
    S_W = getSWMatrix(means, dataset)
    #calculate between class covariance matrix
    S_B = getSBMatrix(means,overall_mean,dataset)

    #calculate projector matrix
    W = getProjector(S_B,S_W)

    imgData = np.empty(shape=(31,81))
    WCounter = 0
    for i in range(31):
        for j in range(81):
            imgData[i][j] = W[WCounter][0].real
            WCounter+=1
    saveImage(imgData, 'out1.jpg')

    imgData = np.empty(shape=(31, 81))
    WCounter = 0
    for i in range(31):
        for j in range(81):
            imgData[i][j] = W[WCounter][1].real
            WCounter += 1
    saveImage(imgData, 'out2.jpg')

    # plot W
    plt.clf()
    plt.plot(W)
    plt.grid()
    plt.tight_layout()
    plt.savefig("W.png")
    plt.show()

    #read new data
    newData, labels = readTrainingData()
    newData, labels = get_sample(newData, labels, 10)
    projectedLabels = []

    #X_lda = newData.dot(W)
    X_lda1 = newData.dot(W[:, 0])
    X_lda2 = newData.dot(W[:, 1])
    X_lda = np.empty(shape=(W.shape[1],newData.shape[0]))
    X_lda[0] = X_lda1
    X_lda[1] = X_lda2
    print(X_lda)

    #create k = 1,...,10 different classifier
    classifiers = np.random.uniform(0.01,means[1].mean(),100)
    bestThreshold = classifiers[0]
    bestPerformance,precision,recall,projectedLabels = evaluateClassifier(bestThreshold,X_lda,labels)
    precisions = np.zeros(classifiers.shape)
    recalls = np.zeros(classifiers.shape)
    for i,threshold in enumerate(classifiers):
        print("Threshold = ",threshold)
        performance,precision,recall,projectedLabels = evaluateClassifier(threshold,X_lda,labels)
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
    print(projectedLabels)

    #plot output
    plotData(X_lda, labels)

    # Plot Precision Recall
    plt.clf()
    plt.plot(recalls, precisions, lw=2, color='navy',
         label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.xlim(precisions.min()-0.5,recalls.max()+0.5)
    plt.ylim(precisions.min()-0.5,recalls.max()+0.5)
    plt.grid()
    plt.savefig("precision_recall.png")
    plt.show()
