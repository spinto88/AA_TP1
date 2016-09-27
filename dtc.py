from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
import matplotlib.pyplot as plt

y = np.load('y.npy')

perf_PCA = []
for fact in range(1,128): #[5,10,20,30,40,120,'original']:
    print 'Factors: ' + str(fact)
    try:
        X = np.load('X_red' + str(fact) + '.npy')
    except:
        X = np.load('X.npy')

# PARA INSPECCIONAR HIPER-PARAMETROS (http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
    for crit in "gini", "entropy":
        for spl in "best", "random":
            perf_dep = []
            for m_dep in range(1, X.shape[1]):
                clf = DTC(criterion=crit, splitter=spl, max_depth=m_dep)
                res = cross_val_score(clf, X, y, cv=10)
                perf_dep.append(np.mean(res))
            plt.plot(perf_dep, lw=2)
            plt.xlabel('Profundidad maxima')
            plt.ylabel('Desempeno (sobre datos entrenamiento)')

# PARA PCA
    clf = DTC(criterion='gini', splitter='best', max_depth=15)
    res = cross_val_score(clf, X, y, cv=10)
    perf_PCA.append(np.mean(res))
    
plt.plot(perf_PCA, lw=2)
plt.xlabel('Numero de componentes')
plt.ylabel('Desempeno (sobre datos entrenamiento)')
