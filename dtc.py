from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np

y = np.load('y.npy')

for fact in [5,10,20,30,40,120,'original']:

    try:
        X = np.load('X_red' + str(fact) + '.npy')
    except:
        X = np.load('X.npy')

    clf = DTC()
    res = cross_val_score(clf, X, y, cv=10)
    print 'Factors: ' + str(fact)
    print "{:.3f}".format(np.mean(res)), "{:.3f}".format(np.std(res))
