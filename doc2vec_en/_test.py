from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from scipy import spatial
from nltk import word_tokenize
import csv

epochs = 100
model = Doc2Vec.load('doc2vec_en/models/kaggle_n_sem2015.model')


class Result:
    def __init__(self, similarity, is_duplicate):
        self.similarity = similarity
        self.is_duplicate = is_duplicate


class QuestionPair:
    def __init__(self, list):
        self.id = list[0].replace('"', '')
        self.qid1 = list[1].replace('"', '')
        self.qid2 = list[2].replace('"', '')
        self.question1 = list[3].replace('"', '')
        self.question2 = list[4].replace('"', '')
        self.is_duplicate = bool(
            int(list[5].replace('"', '').replace('\n', '')))
        self.similarity = -1000


def ReadQuestionsFromFile():
    # questions_list = []
    # f = open("E:/_TEXT_DATA/questions.csv","r",encoding="utf-8")
    # pairs = f.readlines()
    # pairs.pop(0)
    # for pair in pairs:
    #     item = QuestionPair(pair.split(','))
    #     questions_list.append(item)
    # return questions_list
    questions_list = []
    f = open("E:/_TEXT_DATA/questions.csv", "r", encoding="utf-8")
    csv_reader = csv.reader(f, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            questions_list.append(QuestionPair(row))
        line_count += 1
    return questions_list


questions_pairs = ReadQuestionsFromFile()

# results = []

f = open("quora_results", "w+", encoding="utf-8")

csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
for pair in questions_pairs:
    v1 = model.infer_vector(word_tokenize(
        pair.question1.lower()), epochs=epochs)
    v2 = model.infer_vector(word_tokenize(
        pair.question2.lower()), epochs=epochs)
    pair.similarity = 1 - spatial.distance.cosine(v1, v2)
    # results.append(Result(similarity, pair.is_duplicate))
    csv_writer.writerow([pair.id,pair.qid1,pair.qid2,pair.question1,pair.question2,pair.is_duplicate,pair.similarity])

f.close()