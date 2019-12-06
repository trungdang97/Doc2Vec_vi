import os
import numpy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from nltk.tokenize import word_tokenize
from scipy import spatial
from sklearn import metrics

def ReadDataFromFile(path):
    f = open(path, "r")
    sentences = f.readlines()
    for i, s in enumerate(sentences):
        # TaggedDocument(tokens,tags) both params are lists
        sentences[i] = TaggedDocument(word_tokenize(s.lower()), [i])
    return sentences

sem2015_model = 'doc2vec_en/models/sem2016.model'
sem2016_model = 'doc2vec_en/models/sem2016.model'

sem2015_test = ReadDataFromFile('datasets/en/sem_eval/semeval2015_no_polarity/test.txt')
sem2016_test = ReadDataFromFile('datasets/en/sem_eval/semeval2016_no_label/test.txt')

model = Doc2Vec.load(sem2015_model)
vectors = []
f = open('datasets/en/sem_eval/semeval2015_no_polarity/test.vector.txt','w+')
for s in sem2015_test:
    vector = model.infer_vector(s.words,alpha=0.025,steps=100)
    for d in vector:
        f.write("%s\t" % d)
    f.write("\n")
f.close()

model = Doc2Vec.load(sem2016_model)
vectors = []
f = open('datasets/en/sem_eval/semeval2016_no_label/test.vector.txt','w+')
for s in sem2016_test:
    vector = model.infer_vector(s.words,alpha=0.025,steps=100)
    for d in vector:
        f.write("%s\t" % d)
    f.write("\n")
f.close()