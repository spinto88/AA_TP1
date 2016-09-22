from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.load('X_red.npy')
y = np.load('y.npy')

clf = GaussianNB()
res = cross_val_score(clf, X, y, cv=10)
print "{:.3f}".format(np.mean(res)), "{:.3f}".format(np.std(res))
