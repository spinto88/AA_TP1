from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from time import time

# Cargo el data Frame guardado como un archivo cPickle
df = pk.load(file('DataFrame.pk'))

# X es una matriz de documentos x features
X = df.ix[:, 2:].values
y = df['class']

del(df)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
Crange = [0.1, 1.00, 10.00, 100.00]

for kernel in kernels:

    for C in Crange:

        ti = int(time())

        clf = SVC(kernel = kernel, C = C)

        scores = cross_val_score(clf, X, y, cv = 10)

        tf = int(time() - ti)

        mean = np.mean(scores)
        std = np.std(scores)

        fp = open('SVM.dat','a')
        fp.write(kernel + '\t' + str(C) + '\t' + str(mean) + '\t' + str(std) + '\t' + str(time) + '\n')
        fp.close()

