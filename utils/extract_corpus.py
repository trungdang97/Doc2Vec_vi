import os

def LoadCorpus(path):
    f = open(path, "r", encoding='utf-8')
    return f.readlines()

corpus = LoadCorpus('E:/_TEXT_DATA/corpus-title.tok.txt')
amount = 200000

f = open('datasets/vi/corpus-title-{0}.tok.txt'.format(amount),'w+',encoding='utf-8')

for i in range(amount):
    f.write(corpus[i])
f.close()