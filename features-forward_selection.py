
import numpy as np
#import pylab
#import random
import cPickle as pk
#import os
#os.chdir('/home/landfried/gaming/materias/aprendizaje_automatico/AA_TP1')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
import csv

df = pk.load(file('DataFrame.pk'))

clf = RFC(n_estimators=10)

S = []
F = list(df.columns[2:])

res_file = open('resultados/features-forward_selection.csv', 'wb') 
res_writer = csv.writer(res_file)
res_writer.writerow(['S','mean','std'])

y = df.ix[:,'class']
old_perf=0
new_perf=0
while new_perf >= old_perf and len(F)>0:
	old_perf = new_perf 	
	f_ranking = pd.DataFrame([],columns=['f','score'])
	f_ranking.set_index('f')
	for f in F:
		#f=F[1]
		scores = cross_val_score(clf, df.ix[:,S+[f]], y, cv = 5)
		f_ranking.loc[f,'scores'] =  np.mean(scores)
	f_ranking = f_ranking.sort(columns=['scores'], ascending=False) 		
	new_perf = f_ranking.ix[0,'scores']
	f_best = f_ranking.index[0]		
	print new_perf, f_best
	if new_perf >= old_perf:
		
		S.append(f_best)
		F.remove(f_best)
		res_writer.writerow([S, np.mean(scores), np.std(scores)] ) 