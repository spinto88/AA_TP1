from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

"""
Genero matrices X con los atributos reducidos
"""

# X es una matriz de documentos x features
X = np.load('X.npy')

# Matriz de covarianza de features x features
cov = np.cov(np.transpose(X))

# Hago una descomposicion SVD
U, S, Vt = linalg.svd(cov)

for number_of_factors in [5, 10, 20, 40, 80, 120]:

    # Genero una matriz cambio de base con los
    # con los factores que me quedo
    U_red = U[:number_of_factors]

    # Genero la matrix X reducida
    X_red = X.dot(np.transpose(U_red))

    np.save('U_red' + str(number_of_factors) + '.npy', U_red)
    np.save('X_red' + str(number_of_factors) + '.npy', X_red)
