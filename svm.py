import numpy as np
import cPickle as pk
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from time import time

"""
Calculo del score para SVM con distintos kernels
y valores de C.
Para el calculo de SVM la matrix de X con que tomamos
solo contenia 10000 mails (mitad ham, mitad spam)
"""

# X es una matriz de documentos x features
X = np.load('X.npy')
y = np.load('y.npy')

for C in [0.1, 1.00, 10.0, 100.0]:

    for kernel in ['linear','poly','rbf','sigmoid']:

        ti = int(time())

        clf = SVC(kernel = kernel, C = C, cache_size = 2048)

        scores = cross_val_score(clf, X, y, cv = 10)

        fp = open('Svm.txt','a')
        fp.write(str(C) + '\t' + kernel + '\t' + str(np.mean(scores)) + '\t' + str(np.std(scores)) + '\t' + str(tf) + '\n')
        fp.close()
    
