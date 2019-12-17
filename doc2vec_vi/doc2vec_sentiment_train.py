import os
import numpy
import time
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from scipy import spatial
from sklearn import metrics

corpus = []
corpus_range = 0  # take range from 0-999, 1000-1999
max_range = 0  # equal to corpus_size / batch_number
# number of sentences take from corpus (SHOULD BE MODIFIED)
batch_number = 1000

# HYPERPARAMS
model = Doc2Vec()
vector_size = 50  # word vector dimension
min_count = 2  # word count
epochs = 100  # update times for a batch
dm = 1  # use PV-DM, 0 then use DBOW
alpha = 0.025  # should be small

model_path = "doc2vec_vi/models/corpus-title.nosign-trained.model"  # path to trained model
#corpus_path = ""  # path to corpus file
uts_model_path = "doc2vec_vi/models/uts_comments.model" # path to model bonus trained with comments
vlsp_model_path = "doc2vec_vi/models/vlsp_comments.model"
# sentiment corpus
uts_corpus_path = "datasets/vi/uts2017_bank_no_label_cleaned/no_label_cleaned/train.tok.txt"
vlsp_corpus_path = "datasets/vi/vlsp2018_sa_hot_no_label_cleaned/no_label_cleaned/train.tok.txt"

# load corpus into program


def LoadCorpus(path):
    f = open(path, "r", encoding='utf-8')
    return f.readlines()

# get n random sentences from corpus (SHOULD BE MODIFIED)


def PrepareBatch():
    start = corpus_range * batch_number
    end = start + batch_number
    sentences = corpus[start:end]
    for i, s in enumerate(sentences):
        # Split by spaces then remove the LMU (lingustically meaningful unit) concatenation symbol '_'
        sentences[i] = sentences[i].split()
        sentences[i] = [w.replace('_', ' ').lower() for w in sentences[i]]

        # TaggedDocument(tokens,tags) both params are lists
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
        global corpus_range
        while(corpus_range < max_range):

            batch_start = time.time()
            batch = PrepareBatch()  # load batch

            #print('Done preparing batch {0:.2f}'.format(batch_end-batch_start) + 's')
            # build vocab for one-hot and output, MUST HAVE update=True when ADD UNSEEN documents to corpus
            model.build_vocab(batch, update=True)

            for epoch in range(epochs):
                #print("Iteration {0}".format(epoch))
                model.train(batch, total_examples=model.corpus_count,
                            epochs=model.epochs)  # train model

            model.save(vlsp_model_path)  # save model
            batch_end = time.time()
            print('Finish batch {0} / {1}'.format(corpus_range+1, max_range) +
                  ' in {0:.2f}s'.format(batch_end-batch_start) + '')
            corpus_range = corpus_range + 1
    return


#
load_start = time.time()
# load all corpus to MEMORY, too many browser tabs can cause memory error
corpus = LoadCorpus(vlsp_corpus_path)
load_end = time.time()
max_range = len(corpus) / batch_number
print('Done loading corpus {0:.2f}'.format(load_end-load_start) + 's')

Train()  # call train function
