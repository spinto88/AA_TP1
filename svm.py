import numpy as np
import cPickle as pk
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from time import time


# Cargo el data Frame guardado como un archivo cPickle
#df = pk.load(file('DataFrame.pk'))

# X es una matriz de documentos x features
X = np.load('X.npy')
y = np.load('y.npy')

C = 1.00
kernel = 'poly'

ti = int(time())

clf = SVC(kernel = kernel, C = C, cache_size = 2048)

scores = cross_val_score(clf, X, y, cv = 10)

fp = open('Svm.txt','a')
fp.write(str(C) + '\t' + kernel + '\t' + str(np.mean(scores)) + '\t' + str(np.std(scores)) + '\t' + str(tf) + '\n')
fp.close()
    
