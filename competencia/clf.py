from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
import csv
#import matplotlib.pyplot as plt

# Cargo el data Frame guardado como un archivo cPickle
#df = pk.load(file('DataFrame.pk'))
# X es una matriz de documentos x features
X = np.load('X.npy')
y = np.load('y.npy')

clf = RFC(n_estimators=64, n_jobs=4)

clf.fit(X,y)

pk.dump(clf, file('clf.pk','w'))
