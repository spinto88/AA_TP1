from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

# Cargo el data Frame guardado como un archivo cPickle
df = pk.load(file('DataFrame.pk'))

clf = KNN()

# X es una matriz de documentos x features
X = df.ix[:, 2:].values
y = df['class']
nn = []
#for n in [1,5,10,20,40,80,160,320,640,1280]:
#	clf = KNN(n_neighbors=n)
#	scores = cross_val_score(clf, X, y, cv = 10)
#	print  (np.mean(scores), np.std(scores))
#	nn.append( (np.mean(scores), np.std(scores)) )
#print nn
#n_best = np.argmin([i[0] for i in nn])+3
n_best = 1

clf = KNN(n_neighbors=n_best)


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
    
