
import numpy as np
#import pylab
import random
import cPickle as pk
#import os
#os.chdir('/home/landfried/gaming/materias/aprendizaje_automatico/AA_TP1')
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.cross_validation import cross_val_score
import csv
"""
file_hill = open('resultados/features-hill.csv', 'wb') 
write = csv.writer(file_hill)
write.writerow(['S','mean','std'])
file_hill.close()
"""

def write_optimo_local(S, mean, std):
	print "Optimo local!", mean, std
	file_hill = open('resultados/features-hill.csv', 'a') 
	write = csv.writer(file_hill)
	write.writerow([mean,std,S])
	file_hill.close()
	#return 0

df = pk.load(file('DataFrame.pk'))

clf = DTC()
y = df.ix[:,'class']
for t in range(8):
	#t = 1
	print '------------ Nueva buesqueda -------------- '
	size = random.choice(range(len(df.columns[2:])))
	S = list(np.random.choice(df.columns[2:],size=size, replace=False))
	F = list(set(df.columns[2:])-set(S))
	
	scores = cross_val_score(clf, df.ix[:,S], y, cv = 2)
	old_perf =  np.mean(scores)
	old_std = np.std(scores)

	new_perf=old_perf
	new_std = old_std
	while new_perf >= old_perf:
		old_perf = new_perf 	
		old_std = new_std
		f_ranking = pd.DataFrame([],columns=['f','score'])
		f_ranking.set_index('f')
		s_ranking = pd.DataFrame([],columns=['s','score'])
		s_ranking.set_index('s')					
		for f in F:
			P=list(S)			
			P.append(f)
			scores = cross_val_score(clf, df.ix[:,P], y, cv = 2)
			f_ranking.loc[f,'score'] =  np.mean(scores)
			f_ranking.loc[f,'std'] =  np.std(scores)
		for s in S:
			#s=S[1]
			P = list(S)
			P.remove(s)
			scores = cross_val_score(clf, df.ix[:,P], y, cv = 2)
			s_ranking.loc[s,'score'] =  np.mean(scores)
			s_ranking.loc[s,'std'] =  np.std(scores)
		f_ranking = f_ranking.sort(columns=['score'], ascending=False) 		
		s_ranking = s_ranking.sort(columns=['score'], ascending=False) 		
		if len(F)==0: new_perf_f = 0
		else: 
			new_perf_f = f_ranking.ix[0,'score']
			new_std = f_ranking.ix[0,'std']
		if len(S)==0: new_perf_s = 0
		else: 
			new_perf_s = s_ranking.ix[0,'score']	
			new_std =  s_ranking.ix[0,'std']	
		if new_perf_f>=new_perf_s:	
			new_perf = new_perf_f
			aumento =True
		else: 
			aumento =False
			new_perf = new_perf_s
			
		if new_perf >= old_perf:
			if aumento: 
				f_best = f_ranking.index[0]	
				S.append(f_best)
				F.remove(f_best)
			else: 
				s_best = s_ranking.index[0]	
				S.remove(s_best)
				F.append(s_best)
			print new_perf, len(S) 
	
	write_optimo_local(S, old_perf, old_std)
