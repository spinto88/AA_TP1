from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

"""
Clasificador de Naive Bayes que toma las matrices reducidas
y la original e imprime el valor del cross validation
"""

y = np.load('y.npy')

for factors in [5, 10, 20, 120, 'original']:

    try:
        X = np.load('X_red' + str(factors) + '.npy')
    except:
        X = np.load('X.npy')

    clf = GaussianNB()
    res = cross_val_score(clf, X, y, cv=10)

    print factors, np.mean(res), np.std(res)
