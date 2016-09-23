
import numpy as np
#import pylab
import cPickle as pk
import os
os.chdir('/home/landfried/gaming/materias/aprendizaje_automatico/AA_TP1')
#from decisionTree import DecisionTree
import csv
import pandas as pd
df = pk.load(file('DataFrame.pk'))


def log2(v): return np.log(v)/np.log(2)

def entropy(y): 
    set_y = set()
    for yi in y: set_y.add(yi)
    H = 0
    for obs in set_y:
        pi = sum(y == obs)/float(len(y))
        H += - pi * log2(pi)        
    return H
	
def information_gain(column,y):
    H_S = entropy(y)
    G = H_S    
    set_column = set()
    for ci in column: set_column.add(ci)
    for obs in set_column:        
        H_s = entropy(y[column==obs])
        G += - (len(column[column==obs])/float(len(column)))*H_s
    return G

ranking = pd.DataFrame([],columns=['feature','information_gain'])
ranking.set_index('feature')

for i in df.columns[2:]:
	#i = df.columns[3]
	ranking.loc[i,'information_gain'] = information_gain(df.ix[:,i],df.ix[:,'class'])

ranking = ranking.sort(columns=['information_gain'],ascending=False)

res_file = open('resultados/features-ranking.csv', 'wb') 
res_writer = csv.writer(res_file)
res_writer.writerow(['feature','information_gain'])
for i in ranking.index:
	res_writer.writerow([i,ranking.ix[i,'information_gain']] )	
	
	

