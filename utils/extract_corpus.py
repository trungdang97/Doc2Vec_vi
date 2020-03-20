import os


def LoadCorpus(path, amount):
    f = open(path, "r", encoding='utf-8')
    return [next(f) for x in range(amount)]


fileName = 'TheThao'
amount = 1000000
corpus = LoadCorpus('E:/_TEXT_DATA/' + fileName + '.txt', amount)
corpus = [x for x in corpus if len(x.split()) >= 10]

f = open(
    'E:/_TEXT_DATA/{0}-{1}.txt'.format(fileName, len(corpus)), 'w+', encoding='utf-8')

for i in range(len(corpus)):
    f.write(corpus[i])
f.close()
