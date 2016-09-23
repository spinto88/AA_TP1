from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

#
# Dimension del espacio reducido
number_of_factors = 5


# X es una matriz de documentos x features
X = np.load('X.npy')

number_of_doc = X.shape[0]

# Matriz de covarianza de features x features
cov = np.cov(np.transpose(X))

# Hago una descomposicion SVD
U, S, Vt = linalg.svd(cov)

U_red = U[:number_of_factors]

X_red = X.dot(np.transpose(U_red))

np.save('U_red' + str(number_of_factors) + '.npy', U_red)
np.save('X_red' + str(number_of_factors) + '.npy', X_red)
