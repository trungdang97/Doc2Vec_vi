from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from scipy import spatial

epochs = 100
model = Doc2Vec.load('doc2vec_en/models/kaggle_n_sem2015.model')

sent1 = "Australia fires: Almost 2,000 homes destroyed in marathon crisis".split(' ')
sent2 = "Nearly 2,000 houses destroyed in Australia's fire crisis".split(' ')
sent3 = "Australia fires: Almost 2,000 homes destroyed".split(' ')

v1 = model.infer_vector(sent1, epochs=epochs)
v2 = model.infer_vector(sent2, epochs=epochs)
v3 = model.infer_vector(sent3, epochs=epochs)

print(str(1-spatial.distance.cosine(v1,v2)))