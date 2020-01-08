from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from nltk.tokenize import word_tokenize

news1M_kaggle_path = "E:/_TEXT_DATA/abcnews-date-text.csv"

def ReadDataFromFile(path):
    f = open(path, "r", encoding="utf-8")
    sentences = f.readlines()
    for i, s in enumerate(sentences):
        if(i > 0):
            # TaggedDocument(tokens,tags) both params are lists
            sentence = sentences[i].split(',')
            sentences[i] = TaggedDocument(word_tokenize(sentence[1].lower()), [i])
    sentences.pop(0)
    return sentences

train = ReadDataFromFile(news1M_kaggle_path)

model = Doc2Vec(vector_size=50,min_count=2,epochs=100,dm=1)
model.build_vocab(documents=train)
model.train(documents=train, total_examples=model.corpus_count, epochs=model.epochs)
model.save('doc2vec_en/models/kaggle_news1M.model')

print('Done')