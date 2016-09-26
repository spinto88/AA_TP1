from scipy import linalg
import numpy as np
import cPickle as pk
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
import csv
#import matplotlib.pyplot as plt

# Cargo el data Frame guardado como un archivo cPickle
df = pk.load(file('DataFrame.pk'))

#clf = RFC()

# X es una matriz de documentos x features
X = df.ix[:, 2:].values
y = df['class']

res_file = open('resultados/rf-n_estimators.csv', 'wb') 
res_writer = csv.writer(res_file)
res_writer.writerow(['n_estimators','mean','std'])
for i in [1,10,20,40,80,160,320,640]:
	clf = RFC(n_estimators=i)
	scores = cross_val_score(clf, X, y, cv = 10)
	res_writer.writerow([i, np.mean(scores), np.std(scores)] ) 	
	print 'n_estimator = ', i ,'. Score = ', np.mean(scores), np.std(scores)

