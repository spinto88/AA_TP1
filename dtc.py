from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
import matplotlib.pyplot as plt

y = np.load('y.npy')

for fact in 'original',: #[5,10,20,30,40,120,'original']:
    print 'Factors: ' + str(fact)

    try:
        X = np.load('X_red' + str(fact) + '.npy')
    except:
        X = np.load('X.npy')

    #clf = DTC()
    #res = cross_val_score(clf, X, y, cv=10)
    #print "{:.3f}".format(np.mean(res)), "{:.3f}".format(np.std(res))


# PARA INSPECCIONAR PARAMETROS
# Elijo mi clasificador.
#    for crit in "gini", "entropy":
#        for spl in "best", "random":
#            for m_dep in range(1, X.shape[1], X.shape[1]/10):
    perf = []
    for crit in "gini",:
        for spl in "best",:
            for m_dep in range(1, X.shape[1]):

# Ejecuto el clasificador entrenando con un esquema de cross validation de 10 folds.
            # Params. Trees: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
                clf = DTC(criterion=crit, splitter=spl, max_depth=m_dep)
                res = cross_val_score(clf, X, y, cv=10)
                print "{:.3f}".format(np.mean(res)), "{:.3f}".format(np.std(res)), "[crit: {}; spl: {}; m_dep: {}]".format(crit, spl, m_dep)

                perf.append(np.mean(res))
                
    plt.plot(perf, lw=2)
    plt.xlabel('Profundidad maxima')
    plt.ylabel('Desempeno (sobre datos entrenamiento)')
