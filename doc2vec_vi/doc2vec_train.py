import os
import numpy
import time
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from scipy import spatial
from sklearn import metrics

corpus = []

batch_number = 1000

model = Doc2Vec()
vector_size = 50
min_count = 2
epochs = 100
dm = 1 # use PV-DM
alpha = 0.025

model_path = "doc2vec_vi/models/corpus-title.model"
corpus_path = "datasets/vi/corpus-title.tok.txt"

# load corpus into program


def LoadCorpus(path):
    f = open(path, "r", encoding='utf-8')
    return f.readlines()

# get n random sentences from corpus


def PrepareBatch():
    sentences = random.sample(corpus, batch_number)
    for i, s in enumerate(sentences):
        # TaggedDocument(tokens,tags) both params are lists
        # Split by spaces then remove the LMU concatenation symbol '_'
        sentences[i] = sentences[i].split()
        sentences[i] = [w.replace('_', ' ').lower() for w in sentences[i]]

        sentences[i] = TaggedDocument(sentences[i], [i])
    return sentences

# train function


def Train():
    try:
        model = Doc2Vec.load(model_path)
    except IOError:       
        model = Doc2Vec(vector_size=vector_size,
                    min_count=min_count, dm=dm)
    finally:
        iter = 1
        while(True):
            
            batch_start = time.time()
            batch = PrepareBatch()
            
            #print('Done preparing batch {0:.2f}'.format(batch_end-batch_start) + 's')

            model.build_vocab(batch,update=True)
            for epoch in range(epochs):
                #print("Iteration {0}".format(epoch))
                model.train(batch,total_examples=model.corpus_count,epochs=model.epochs)
            
            model.save(model_path)
            batch_end = time.time()
            print('Finish batch {0}'.format(iter) + 'in {0:.2f}s'.format(batch_end-batch_start))
            iter = iter + 1
    return


#
load_start = time.time()
corpus = LoadCorpus(corpus_path)
load_end = time.time()
print('Done loading corpus {0:.2f}'.format(load_end-load_start) + 's')

Train()