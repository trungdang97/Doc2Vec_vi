import os
import numpy as np
import time
import random
import math

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors
from scipy import spatial


def TS(v1_norm, v2_norm, sin):
    return (v1_norm * v2_norm * sin) / 2


def SS(r, degrees):
    return math.pi * math.pow(r, 2) * (degrees / 360)

def MD(v1, v2):
    return abs(math.sqrt(sum(pow(d, 2) for d in v1)) - math.sqrt(sum(pow(d, 2) for d in v2)))

def ED(v1_norm,v2_norm,cos):
    return math.sqrt(pow(v1_norm,2)+pow(v2_norm,2)-2*v1_norm*v2_norm*cos)

def TSSS(v1, v2):
    cos = 1-spatial.distance.cosine(v1, v2)
    sin = math.sqrt(1-math.pow(cos, 2))
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    r = pow(ED(v1_norm,v2_norm,cos) - MD(v1,v2),2)
    rad = math.acos(v1_norm/v2_norm)
    degrees = rad*180/math.pi
    ts = TS(v1_norm,v2_norm,sin)
    ss = SS(r,degrees)
    return ts*ss

epochs = 10000

model = Doc2Vec.load('doc2vec_vi/models/corpus-title-1135.model')

sent1 = ['Thực hiện','các', 'biện pháp', 'phòng',',', 'chống','bệnh', 'viêm phổi cấp', 'do', 'virus', 'Corona']
sent2 = ['Các', 'biện pháp', 'chống', 'dịch', 'nCoV', 'tại', 'Việt Nam']

sent3 = ['Người dân', 'Hà Nội', 'Sài Gòn', 'đón', 'Giáng sinh']
sent4 = ['Người dân', 'khắp', 'thế giới', 'đón', 'Giáng sinh']

v1 = model.infer_vector([x.lower() for x in sent1], epochs=epochs)
v2 = model.infer_vector([x.lower() for x in sent2], epochs=epochs)

print(str(1-spatial.distance.cosine(v1,v2)))

# print(TSSS(v1, v2))