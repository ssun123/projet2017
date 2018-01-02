import time
from pprint import pprint

from konlpy.tag import Twitter

pos_tagger = Twitter()

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split(',') for line in f.read().splitlines()]
        data=data[1:]
    return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def tokenize3(doc):
    # 조사,구두점 제외
    list=[]
    pos = pos_tagger.pos(doc)
    for item in pos:
        word,tag = item
        if tag in ['Josa','Punctuation','Number','Eomi','PreEomi','Foreign','Suffix']:
            pass
        else:
            list.append('/'.join(item))
#    print(list)
    return list

#train_= read_data('pre_proc_gprm.txt') #지마켓 프리미엄 리뷰 데이터 의미 유무 전처리 결과 1~6 8200->5000개
#train_= read_data('pre_proc_ggen.txt') #지마켓 일반리뷰 데이터 의미 유무 전처리 결과 1~4 2만천개-> 4천개
#train_= read_data('pre_proc_gtotal_1212.txt') #전처리후 리뷰 토탈  4만2천 -> 1만6천
#train_= read_data('pre_proc_gtotal_1213.txt') #전처리후 리뷰 토탈  8만2천 -> 2만6천
#train_= read_data('pre_proc_gtotal_1218.txt') #전처리후 리뷰 토탈  17만 -> 4만 8천
train_= read_data('pre_proc_gtotal_1221.txt') #전처리후 리뷰 토탈  29만 -> 7만 7천

#food# train_= read_data('pre_proc_food_1218.txt') #롯데마트 식품 리뷰



train = [tokenize3(row[1]) for row in train_ if row[2]==' 0']
#train = [tokenize3(row[1]) for row in train_ ]   #전체셋

print(len(train_))
print(len(train))

tokens = [t for d in train for t in d]                                                                   
print(tokens[0],tokens[2])
print(len(tokens))

import nltk
text = nltk.Text(tokens, name='f')
#print(text)
pprint(text.vocab().most_common(50))    # returns frequency distribution

#단어가 쓰인 문맥
###text.concordance('배송/Noun', lines=10)
##text.concordance('가격/Noun', lines=10)
#text.concordance('품질/Noun', lines=10)


import multiprocessing
cores = multiprocessing.cpu_count()
print(cores)

#w2v_model_1208 리뷰 전
#w2v_model_1211 조사 구두점등 제외
from gensim.models import Word2Vec
model = Word2Vec(train, size=100, window = 3, min_count=100, workers=4, iter=200, sg=1)
#model.save('w2v_model_food_1218')
model.save('w2v_model_1221_notnor')
#model = Word2Vec.load('w2v_model_food_1218')
model = Word2Vec.load('w2v_model_1221_notnor')
model.wv.save_word2vec_format('w2v_model_1221_notnor.txt')
#model.wv.save_word2vec_format('w2v_model_food_1218.txt')

#f_w2v = open('w2v_model_1208.txt')
#print(f_w2v.readline())
print(model.wv.most_similar('가격/Noun'))
print(model.wv.most_similar('배송/Noun'))
print(model.wv.most_similar('품질/Noun'))

print(model.wv.most_similar('사이즈/Noun'))
print(model.wv.most_similar('색상/Noun'))
print(model.wv.most_similar('핏/Noun'))
print(model.wv.most_similar('신축/Noun'))
print(model.wv.most_similar('재질/Noun'))
print(model.wv.most_similar('디자인/Noun'))



                       
import numpy as np
from scipy.spatial import distance

# loading
f_w2v = open('w2v_model_1221_notnor.txt')
#f_w2v = open('w2v_model_food_1218.txt')
print(f_w2v.readline())
data = [line.split(' ') for line in f_w2v.read().splitlines()]
#data=data[1:]
arr = np.array(data)
arr2 = arr[:,1:]
#arr2 = np.genfromtxt('w2v_model_1208.txt', delimiter=' ', skip_header=1)
#print(arr[0])
print(arr[0][0],arr[0][1])
print(arr[1][0],arr[1][1])

#print(arr2[0])
print(arr2[0][0],arr2[0][1])
print(arr2[1][0],arr2[1][1])


#문장별 스코어(Term-Document Matrix)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#vec = CountVectorizer()
#X = vec.fit_transform(train)
i,j = 0,0
tdm = []
tdm_f = []
#for doc in train[:100]:
for doc in train:
    for kwd in data:
        if(kwd[0] in doc):
            tdm.append('1')
        else:
            tdm.append('0')
        #j= j+1
    #i= i+1
    tdm_f.append(tdm)
    tdm=[]

#가중치 행렬 
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

dist = squareform(pdist(arr2, 'seuclidean'))
dist2= 1 - cosine_similarity(arr2,arr2)
#print (dist)
#dist_exp = np.exp(-dist^2/100)
#print(dist_exp)
list =[]
kwd = ['가격/Noun','배송/Noun','사이즈/Noun','색상/Noun','핏/Noun','재질/Noun']
for t in kwd:
    for i in range(0,len(arr)):
        if(t == arr[i][0]):
            #list.append([t,arr2[i]])        #이렇게 하면 변환 안됨 np array에 네이밍 하는게 좋은데 아직 모르겠
            list.append(dist[i])
#print(list)
arr3 = np.array(list,dtype='float64')

from sklearn.preprocessing import normalize
n_arr3 = normalize(arr3)      #정규화, 기능별 가중치 매트릭스
#n_arr3 = arr3   #정규화 안하면?
#print(n_arr3)
#print(n_arr3.T)           #embedding_word_vector수 X kwd
trans_n_arr3 = n_arr3.T

#kwd 별 스코어
tdm_arr = np.array(tdm_f,dtype='float64')
res = np.dot(tdm_arr,n_arr3.T)
print(res)
len_list=[]
#문장별 길이 체크
for line in train:
    if(len(line)>4):
        len_list.append(len(line))
    else:
        len_list.append(0)
#문장길이로 res값 변경
for i in range(0,len(len_list)):
    if(len_list[i] == 0):
        res[i] = 0
    else:
        res[i] /= len_list[i]
res_wr = open('fashion_review_6kwd.txt','w')

#결과print
for i in range(0, len(res)):
    res_wr.write(','.join(map(str,res[i])))
    res_wr.write(',,')
    res_wr.write(','.join(train[i]))
    res_wr.write('\n')

res_list=[]
for rev in res.T:
   p = rev.max()
   idx = rev.argmax()
   print(p,train[idx])













