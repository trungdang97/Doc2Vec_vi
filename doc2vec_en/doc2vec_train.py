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
        sentence = sentences[i].split('\t')
        sentences[i] = TaggedDocument(word_tokenize(sentence[1].lower()), [i])
    return sentences


def ReadDataFromText8():
    dataset = api.load('text8')
    data = [d for d in dataset]
    for i, s in enumerate(data):
        # TaggedDocument(tokens,tags) both params are lists
        data[i] = TaggedDocument(s, [i])
    return data

#train = ReadDataFromText8()
train = ReadDataFromFile(news1M_path)
#test = ReadDataFromFile(test_path)

#sem2015 = ReadDataFromFile('datasets/en/sem_eval/semeval2015_no_polarity/train.txt')
#sem2016 = ReadDataFromFile('datasets/en/sem_eval/semeval2016_no_label/train.txt')

# save model after training

model = Doc2Vec(vector_size=50, min_count=2, epochs=100, dm=1, negative=0)
model.build_vocab(documents=train)
model.train(documents=train, total_examples=model.corpus_count, epochs=model.epochs)
model.save("doc2vec_en/models/english1M.model")

# load model
# model = Doc2Vec.load("lee.model")
# vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires']) # infer_vector is a method that convert a list of tokens into a vector
# we can say doc2vec (paragraph vector) is just word2vec, however the words' vectors are modified to fit the multiple context that it was used in

# print(vector)
#vector1 = model.infer_vector(test[0].words, alpha=0.5,steps=100)
#vector2 = model.infer_vector(test[0].words, alpha=0.5,steps=100)
# print(vector1)
# print(vector2)

# model = Doc2Vec.load("doc2vec_en/models/lee.model")
# model.build_vocab(sem2015, update=True)
# model.train(documents=sem2015, total_examples=model.corpus_count, epochs=model.epochs)
# model.save("doc2vec_en/models/sem2015.model")

# model = Doc2Vec.load("doc2vec_en/models/lee.model")
# model.build_vocab(sem2016, update=True)
# model.train(documents=sem2016, total_examples=model.corpus_count, epochs=model.epochs)
# model.save("doc2vec_en/models/sem2016.model")

print("Done")

# vector1 = model.infer_vector(test[3].words, alpha=0.5, steps=100)
# vector2 = model.infer_vector(test[20].words, alpha=0.5, steps=100)
# cs = metrics.pairwise.cosine_similarity(vector1.reshape(1,-1),vector2.reshape(1,-1))
# print(cs)
# for doc in test:
#     vector2 = model.infer_vector(doc.words, alpha=0.5, steps=100)
#     cs = 1-spatial.distance.cosine(vector1, vector2)
#     print(cs)

#cs = spatial.distance.cosine(vector1,vector2)
# print(cs)
# print(1-cs)
# load model, convert inputs into vectors and save it to database and calculate the cosine similarity
# since each input and the output of the network is one-hot, the vocabulary must be big enough
# the weights' matrix are just a vector table of the vocabulary

#word_vectors = model.wv
# print(len(word_vectors.vocab))
# for word in word_vectors.vocab:
#    print(word)
