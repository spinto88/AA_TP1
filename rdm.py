from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

# Cargo el data Frame guardado como un archivo cPickle
df = pk.load(file('DataFrame.pk'))

clf = DTC()

# X es una matriz de documentos x features
X = df.ix[:, 2:].values
y = df['class']

scores = cross_val_score(clf, X, y, cv = 10)
print 'Sin reduccion de dim: score = ', np.mean(scores), np.std(scores)

number_of_doc = X.shape[0]

# Matriz de covarianza de features x features
cov = np.cov(np.transpose(X))

# Hago una descomposicion SVD
U, S, Vt = linalg.svd(cov)

# Cantidad de autovalores con los que me quedo 
# (< dim(features))

for eigs in [5, 10, 20, 40]:

    number_of_eig = eigs
    U_reduce = U[:number_of_eig]
    S_reduce = S[:number_of_eig]

    # Matrix reducida
    X_reduce = X.dot(np.transpose(U_reduce))
  
    scores = cross_val_score(clf, X_reduce, y, cv = 10)
    print 'Eig: ', number_of_eig, ' - score: ', np.mean(scores), np.std(scores)
    
