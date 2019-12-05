import os
import numpy
import time
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from scipy import spatial

model_path = "doc2vec_vi/models/corpus-title.model"
corpus_path = "datasets/vi/corpus-title-small.tok.txt"

# load corpus into program


def LoadCorpus(path):
    f = open(path, "r", encoding='utf-8')
    return f.readlines()

# get n random sentences from corpus


def PrepareBatch():
    sentences = corpus.copy()
    for i, s in enumerate(sentences):
        # TaggedDocument(tokens,tags) both params are lists
        # Split by spaces then remove the LMU concatenation symbol '_'
        sentences[i] = sentences[i].split()
        sentences[i] = [w.replace('_', ' ').lower() for w in sentences[i]]
    return sentences

corpus = LoadCorpus(corpus_path)
batch = PrepareBatch()
model = Doc2Vec.load(model_path)

sentence = ['nghiên cứu','tiện ích']

vector = model.infer_vector([w.lower() for w in sentence], steps=100, alpha=0.05)

evaluation = []

class eval:
    sentence = ''
    similarity = 0
    def __init__(self, sentence, similarity):
        self.sentence = sentence
        self.similarity = similarity

for i, s in enumerate(batch):
    v = model.infer_vector([w.lower() for w in s], steps=100, alpha=0.05)
    cs = 1 - spatial.distance.cosine(vector,v)
    evaluation.append(eval(corpus[i], cs))
    #print(corpus[i],cs)

evaluation = sorted(evaluation, key=lambda x: x.similarity, reverse=True)
for item in evaluation:
    print('{0:.2f} {1}'.format(item.similarity,item.sentence))