from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import cross_val_score
#import matplotlib.pyplot as plt
import csv
# Cargo el data Frame guardado como un archivo cPickle
df = pk.load(file('DataFrame.pk'))

clf = KNN()

# X es una matriz de documentos x features
X = df.ix[:, 2:].values
y = df['class']

res_file = open('resultados/knn-n_neighbors.csv', 'wb') 
res_writer = csv.writer(res_file)
res_writer.writerow(['n_neighbors','mean','std'])
for n in [1,5,10,20,40,80,160,320,640]:
	clf = KNN(n_neighbors=n)
	scores = cross_val_score(clf, X, y, cv = 10)
	res_writer.writerow([n, np.mean(scores), np.std(scores)] )
	print  (np.mean(scores), np.std(scores))


