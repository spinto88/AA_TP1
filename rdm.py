from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

# Cargo el data Frame guardado como un archivo cPickle
df = pk.load(file('DataFrame.pk'))

# X es una matriz de documentos x features
X = df.ix[:, 2:].values
y = df['class']

np.save('X.npy', X)
np.save('y.npy', y)

exit()

number_of_doc = X.shape[0]

# Matriz de covarianza de features x features
cov = np.cov(np.transpose(X))

# Hago una descomposicion SVD
U, S, Vt = linalg.svd(cov)

S = np.log10(S)

plt.plot(S, '.-', markersize = 10)
plt.axis([0, len(S), min(S) - 1, max(S) + 1])
plt.xlabel('Number of eigenvalue')
plt.ylabel('log10(Eigenvalue)')
plt.grid('on')
plt.show()

