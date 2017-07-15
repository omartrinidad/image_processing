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


def readImage(filename):
    """
    read image
    """
    f = misc.imread(filename, flatten=True).astype("float")
    return f

def plot_data(self, data):
    classes = ['background','car']
    colors = cm.rainbow(np.linspace(0, 1, len(classes)))
    plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

    for i, row in data.iterrows():
        proj = np.dot(self.w, row[:self.labels])
        plt.scatter(proj, np.random.normal(0,1,1)+0, color =
                    plotlabels[row[self.labelcol]])
    plt.show()

#with tarfile.open("uiuc/uiucTest.tgz", "r:gz") as tar:
#    for entry in tar:
#        img = tar.extractfile(entry).read()
#        img = np.fromiter(img, dtype=np.float64)

dataset = np.empty(shape=(2511,))
for root, _, files in os.walk('uiuc/train1'):
    for filename in files:
        path = os.path.join(root, filename) 
        image = misc.imread(path, flatten=True).astype("float")
        dataset = np.vstack((image.ravel(), dataset))

# tricky code
dataset = np.delete(dataset, (-1), axis=0)
label = np.char.array(files).rfind('Pos')
label[np.where(label==-1)] = 0 #neg class
label[np.where(label>0)] = 1    #pos class


#calculate class means
means = []
means.append(np.mean(dataset[np.where(label==0)],axis=0));
means.append(np.mean(dataset[np.where(label==1)],axis=0));

#calculate overall mean
overall_mean = np.mean(dataset, axis=0)


#calculate between class covariance matrix
S_B = np.zeros((dataset.shape[1],dataset.shape[1]))
for i,mean_vec in enumerate(means):
    n = dataset[np.where(label == i)].shape[0]
    mean_vec = mean_vec.reshape(2511,1) # make column vector
    overall_mean = overall_mean.reshape(2511,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

#calculate within class covariance matrix
S_W = np.zeros(S_B.shape)
for i,mean_vec in enumerate(means):
    class_scatter = np.zeros(S_W.shape)
    for row in dataset[np.where(label==i)]:
        row, mean_vec = row.reshape(2511,1), mean_vec.reshape(2511,1) # make column vectors
        class_scatter += (row-mean_vec).dot((row-mean_vec).T)
    S_W += class_scatter                             # sum class scatter matrices

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

W = np.hstack((eig_pairs[0][1].reshape(2511,1), eig_pairs[1][1].reshape(2511,1)))

# sw = s1 - s2
# solve sw ^ -1 (mu1 - mu2) = , check slide 12
