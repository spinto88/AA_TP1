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

for C in [1.00, 0.1, 100.0, 10.00]:

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:

        ti = int(time())

        clf = SVC(kernel = kernel, C = C)
        scores = cross_val_score(clf, X, y, cv = 10)

        tf = int(time()-ti)

        fp = open('Svm.txt','a')
        fp.write(str(C) + '\t' + kernel + '\t' + str(np.mean(scores)) + '\t' + str(np.std(scores)) + '\t' + str(tf) + '\n')
        fp.close()
    

