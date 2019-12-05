import os
import numpy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from nltk.tokenize import word_tokenize
from scipy import spatial
from sklearn import metrics

train_path = "datasets/en/lee_background.cor"
test_path = "datasets/en/lee.cor"

def ReadDataFromFile(path):
    f = open(path, "r")
    sentences = f.readlines()
    for i, s in enumerate(sentences):
        # TaggedDocument(tokens,tags) both params are lists
        sentences[i] = TaggedDocument(word_tokenize(s.lower()), [i])
    return sentences


train = ReadDataFromFile(train_path)
test = ReadDataFromFile(test_path)

# save model after training

# model = Doc2Vec(vector_size=50, min_count=2, epochs=100, dm=1, negative=0)
# model.build_vocab(documents=train)
# model.train(documents=train, total_examples=model.corpus_count, epochs=model.epochs)
# model.save("lee.model")

# load model
model = Doc2Vec.load("lee.model")
# vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires']) # infer_vector is a method that convert a list of tokens into a vector
# we can say doc2vec (paragraph vector) is just word2vec, however the words' vectors are modified to fit the multiple context that it was used in

# print(vector)
#vector1 = model.infer_vector(test[0].words, alpha=0.5,steps=100)
#vector2 = model.infer_vector(test[0].words, alpha=0.5,steps=100)
# print(vector1)
# print(vector2)

vector1 = model.infer_vector(test[3].words, alpha=0.5, steps=100)
vector2 = model.infer_vector(test[20].words, alpha=0.5, steps=100)
cs = metrics.pairwise.cosine_similarity(vector1.reshape(1,-1),vector2.reshape(1,-1))
print(cs)
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
