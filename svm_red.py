import numpy as np
import cPickle as pk
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from time import time

y = np.load('y_acotada.npy')

C = 1.00
kernel = 'poly'

for number_of_factors in [5, 20, 40, 'original']:

    ti = int(time())

    try:
        X = np.load('X_red' + str(number_of_factors) + '.npy')
    except:
        X = np.load('X_acotada.npy')

    clf = SVC(kernel = kernel, C = C)
    scores = cross_val_score(clf, X, y, cv = 10)

    tf = int(time()-ti)

    print np.mean(scores), tf

    fp = open('Svm_red_2.txt','a')
    fp.write(str(number_of_factors) + '\t' + str(np.mean(scores)) + '\t' + str(np.std(scores)) + '\t' + str(tf) + '\n')
    fp.close()
    

