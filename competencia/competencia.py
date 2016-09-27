from feature_extractor import feature_extractor
import json 
import cPickle as pk

data_competencia = json.load(open('...')

clf = pk.load(file('clf.pk'))

j = 0
for text in data_competencia:

    mail = feature_extractor(text)

    print j, clf.predict(mail)

    j += 1
    

