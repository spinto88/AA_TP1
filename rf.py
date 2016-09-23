from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
import csv
#import matplotlib.pyplot as plt

# Cargo el data Frame guardado como un archivo cPickle
df = pk.load(file('DataFrame.pk'))

clf = RFC()

# X es una matriz de documentos x features
X = df.ix[:, 2:].values
y = df['class']

n_estimators=160


number_of_doc = X.shape[0]

# Matriz de covarianza de features x features
cov = np.cov(np.transpose(X))

# Hago una descomposicion SVD
U, S, Vt = linalg.svd(cov)

# Cantidad de autovalores con los que me quedo 
# (< dim(features))
res_file = open('resultados/rf.csv', 'wb') 
res_writer = csv.writer(res_file)
res_writer.writerow(['n_variables_svd','mean','std'])
for eigs in [5, 10, 20, 40]:

	number_of_eig = eigs
	U_reduce = U[:number_of_eig]
	S_reduce = S[:number_of_eig]
	# Matrix reducida
	X_reduce = X.dot(np.transpose(U_reduce))
	  
	scores = cross_val_score(clf, X_reduce, y, cv = 10)
	res_writer.writerow([eigs, np.mean(scores), np.std(scores)] )
					
	print 'Eig: ', eigs, ' - score: ', np.mean(scores), np.std(scores)
	    
