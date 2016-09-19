# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
import cPickle as pk

# carga las palabras (word features) que mejor discriminan segun ("On Feature Extraction for Spam E-Mail Detection". Serkan Gunal et. al.)
words_feat1 = ['make', 'actor', 'money', 'address', 'offer', 'monkey', 'adult', 'order', 'all', 'original', 'blue', 'bank', 'our', 'best', 'over', 'chamber', 'business', 'paper', 'punch', 'call', 'people', 'revenue', 'click', 'pm', 'conference', 'price', 'credit', 'cs', 'promotion', 'tape', 'data', 'quality', 'thread', 'dear', 're', 'egg', 'direct', 'receive', 'edu', 'regards', 'email', 'remove', 'upgrade', 'fast', 'report', 'font', 'fault', 'free', 'spam', 'george', 'table', 'hello', 'take', 'germ', 'here', 'technology', 'gun', 'hi', 'telnet', 'hammer', 'how', 'thank', 'hp', 'think', 'wrap', 'internet', 'indigo', 'investment', 'we']
# carga las palabras que aparecen frecuentemente (y no contenidas ya en words_feat1) en spam segun ("saving private E-mails". Steven Vaughan-Nichols)
words_feat2 = ['madam', 'republic', 'enter', 'valuable']

# Leo los mails (poner los paths correctos).
ham_txt = json.load(open('dataset_dev/ham_dev.json'))
spam_txt = json.load(open('dataset_dev/spam_dev.json'))

# Armo un dataset de Pandas
# http://pandas.pydata.org/

df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# Extraigo dos atributos simples:
# 1) Longitud del mail.
df['len'] = map(len, df.text)

# 2) Cantidad de espacios en el mail.
def count_spaces(txt): return txt.count(" ")
df['count_spaces'] = map(count_spaces, df.text)

# Extraigo otros atributos
# 3) Cantidad de apariciones de cada una de las words_feature
for word in words_feat1 + words_feat2:
    print word,
    df["count_" + word] = [txt.lower().count(word) for txt in df.text]



# Guardo el data frame en un archivo
# Se carga con cPickle.load(file('DataFrame.pk'))
fp = open('DataFrame.pk','w')
pk.dump(df,fp)
fp.close()


"""
X e y son las matrices que hay que pasarles a los clasificadores
# Preparo data para clasificar
X = df.ix[:,2:].values  # ix sirve para indexar las columnas con enteros
y = df['class']
"""


