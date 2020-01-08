import os
import numpy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from scipy import spatial
from sklearn import metrics

train_path = "datasets/en/lee_background.cor"
test_path = "datasets/en/lee.cor"

news1M_path = "E:/_TEXT_DATA/eng_news_2016_1M/eng_news_2016_1M-sentences.txt"

def ReadDataFromFile(path):
    f = open(path, "r", encoding="utf-8")
    sentences = f.readlines()
    for i, s in enumerate(sentences):
        # TaggedDocument(tokens,tags) both params are lists
        sentences[i] = TaggedDocument(word_tokenize(s.lower()), [i])
    return sentences


sem2015 = ReadDataFromFile('datasets/en/sem_eval/semeval2015_no_polarity/train.txt')
sem2016 = ReadDataFromFile('datasets/en/sem_eval/semeval2016_no_label/train.txt')

model = "doc2vec_en/models/kaggle_news1M.model"

model1 = Doc2Vec.load("doc2vec_en/models/lee.model")
model1.build_vocab(sem2015, update=True)
model1.train(documents=sem2015, total_examples=model1.corpus_count, epochs=model1.epochs)
model1.save("doc2vec_en/models/kaggle_n_sem2015.model")

model2 = Doc2Vec.load("doc2vec_en/models/lee.model")
model2.build_vocab(sem2016, update=True)
model2.train(documents=sem2016, total_examples=model1.corpus_count, epochs=model1.epochs)
model2.save("doc2vec_en/models/kaggle_n_sem2016.model")

print("Done")