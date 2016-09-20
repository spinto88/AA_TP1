import codecs
import os

fp = codecs.open('Words_ham.txt','r','utf-8')
ham_text = fp.read()
fp.close()

fp = codecs.open('Words_spam_b.txt','r','utf-8')
spam_text = fp.read()
fp.close()

ham_text = ham_text.split('\n')
spam_text = spam_text.split('\n')

ham = [i.split('\t') for i in ham_text]
spam = [i.split('\t') for i in spam_text]

words_ham = [i[0] for i in ham]
words_spam = [i[0] for i in spam]


print len(words_ham), len(words_spam)
intersection = list(set(words_ham) & set(words_spam))


try:
    os.remove('Exclusive_ham.txt')
except:
    pass

try:
    os.remove('Exclusive_spam.txt')
except:
    pass

fp = codecs.open('Exclusive_ham.txt','a','utf8')
for word in words_ham:
    if word not in intersection:
        fp.write(unicode(word) + '\n')
fp.close()

fp = codecs.open('Exclusive_spam.txt','a','utf8')
for word in words_spam:
    if word not in intersection:
        fp.write(word + '\n')
fp.close()





