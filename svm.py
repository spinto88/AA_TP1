import numpy as np
import cPickle as pk
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from time import time

# X es una matriz de documentos x features
X = np.load('X_poca_data.npy')
y = np.load('y_poca_data.npy')

for kernel in ['rbf', 'poly', 'sigmoid', 'linear']:
    
    for C in [1.00, 0.1, 100.0, 10.00]:

        ti = int(time())

        print kernel, C

        clf = SVC(C = C, kernel = kernel, cache_size = 2048)
        scores = cross_val_score(clf, X, y, cv = 10)

        tf = int(time()-ti)

        fp = open('Svm.txt','a')
        fp.write(str(C) + '\t' + kernel + '\t' + str(np.mean(scores)) + '\t' + str(np.std(scores)) + '\t' + str(tf) + '\n')
        fp.close()
    
