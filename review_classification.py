import pre_proc_reviews as pp
from pprint import pprint


train_data = pp.read_data('test_all4_1.txt') #test_all4.txt 과일,야채,고기 포함 50:50 7만
test_data = pp.read_data('crawl_banana2_uid.txt') #banana 테스트 #test_set4 포도 상품평 일부 50:50 600개

#res = open('pre_proc2.txt','w')
#pp.res
from konlpy.tag import Twitter
pos_tagger = Twitter()
#품질이 정말 좋아요(좋네요) 배송이 빨라서 좋아요 자주 구매하는 상품입니다
#default = ['품질이 정말 좋아요','배송이 빨라서 좋아요','자주 구매하는 상품입니다','늘 구매하는 상품 좋아요', '맘에듭니다','맛있어요','늘 구매하는 제품입니다','맘에듭니다맛있어요','배송 빠르고 편해요']
#pos_default = [pos_tagger.pos(i) for i in default]

train_docs = [(pp.tokenize3(row[5],row[0])) for row in train_data]  #리턴값 (pos,score) score:0 의미있음 score1: 의미없음
test_docs = [(pp.tokenize3(row[5],row[0])) for row in test_data]

tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))

import nltk
#text = nltk.Text(tokens, name='pork')
#print(text)

import multiprocessing
cores = multiprocessing.cpu_count()
print(cores)


from gensim.models import doc2vec
from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
# 여기서는 15만개 training documents 전부 사용함
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]


# 사전 구축
doc_vectorizer = doc2vec.Doc2Vec(
    dm=0,            # PV-DBOW / default 1
    dbow_words=1,    # w2v simultaneous with DBOW d2v / default 0
    window=6,        # distance between the predicted word and context words
    size=300,        # vector size
    alpha=0.025,     # learning-rate
    seed=1234,
    min_count=50,    # ignore with freq lower
    min_alpha=0.025, # min learning-rate
    workers=cores,   # multi cpu
    hs = 1,          # hierarchical softmax / default 0
    negative = 10,   # negative sampling / default 5
    )

doc_vectorizer.build_vocab(tagged_train_docs)
print(str(doc_vectorizer))


# Train document vectors!
import time
start_time = time.time()
for epoch in range(10):
    doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
    doc_vectorizer.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay
    print("epoch : ", epoch)
check_time = time.time() - start_time
print("During Time : {}".format(check_time))


# To save
doc_vectorizer.save('Doc2Vec(dbow+w,d300,n10,hs,w8,mc20,s0.001,t24)_1218.model')
# load
doc_vectorizer = doc2vec.Doc2Vec.load('Doc2Vec(dbow+w,d300,n10,hs,w8,mc20,s0.001,t24)_1218.model')
print(str(doc_vectorizer))

#학습결과
#pprint(doc_vectorizer.most_similar('신선해/Adjective'))
#pprint(doc_vectorizer.most_similar('시들/Verb'))
pprint(doc_vectorizer.most_similar('비싼/Adjective'))
pprint(doc_vectorizer.most_similar('저렴/Noun'))
pprint(doc_vectorizer.most_similar('애용/Noun'))
pprint(doc_vectorizer.most_similar('품질/Noun'))
pprint(doc_vectorizer.most_similar('가격/Noun'))


#pprint(doc_vectorizer.most_similar('삼겹살/Noun'))
#pprint(doc_vectorizer.most_similar('깻잎/Noun'))

#pprint(doc_vectorizer.most_similar(positive=['깻잎/Noun', '삼겹살/Noun'], negative=['시들/Verb']))

#단어가 쓰인 문맥
#text.concordance('신선해/Adjective', lines=10)
#text.concordance('가격/Noun', lines=10)


train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
train_y = [doc.tags[0] for doc in tagged_train_docs]
len(train_x)       # 사실 이 때문에 앞의 term existance와는 공평한 비교는 아닐 수 있다
# => 26913
len(train_x[0])
# => 200
test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]
print(len(test_x))
# => 26913
print(len(test_x[0]))
# => 200
#제목으로 동일하게


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=1234)
classifier.fit(train_x, train_y)
score =classifier.score(test_x, test_y)
print(score)


file = open('mlp_res_mean_banana.txt','w')
from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(train_x, train_y)
print("Training set score: %f" % mlp.score(train_x, train_y))
print("Test set score: %f" % mlp.score(test_x, test_y))
val_chk = mlp.predict_proba(test_x)
#for trans_pork
test_data2=[]
for row in test_docs:       #####수정해야함 현재 평점으로 들어가서 제대로된 정보x
    if(row[1]==0)| (row[1]==1):
        test_data2.append(row)

for i in range(1,len(test_data2)):
    if(val_chk[i][0] > val_chk[i][1]):
        chk = '0'
    else:
        chk = '1'
   # print('내용 : ',test_data[i][5],test_data2[i][0],', 점수 : ',test_data2[i][1], ', 판별 : ', chk, ', 결과 : ', val_chk[i])
    #res = test_data[i][5]+'\t'+test_data[i][0]+'\t'+chk+'\t'+val_chk[i]+'\n'
    file.write(test_data[i][5]+'\t'+str(test_data2[i][1])+'\t'+chk+'\t'+'[{} {}]'.format(val_chk[i][0],val_chk[i][1])+'\n')
    #앞에가 크면 긍정 뒤에가 크면 부정 맞나?

#file.close()
####CNN으로 분류
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf
import random
import numpy as np

train_x_np = np.asarray(train_x)
train_y_np = np.asarray(train_y, dtype=int)

type(train_x_np[0,0])
print(train_x_np.shape)
print(train_y_np.shape)


print(train_x_np[:4, :4])
print(train_y_np[:4])
test_x_np = np.asarray(test_x)
test_y_np = np.asarray(test_y, dtype=int)
test_y_np = np.eye(2)[test_y_np.reshape(-1)]

tf.reset_default_graph()

train_y_np = np.eye(2)[train_y_np.reshape(-1)]

# parameters
learning_rate = 0.01  # we can use large learning rate using Batch Normalization
training_epochs = 50
batch_size = 300
keep_prob = 0.7

# input place holders
X = tf.placeholder(tf.float32, [None, 300])
Y = tf.placeholder(tf.float32, [None, 2])
train_mode = tf.placeholder(tf.bool, name='train_mode')

# layer output size
hidden_output_size = 300
final_output_size = 2

xavier_init = tf.contrib.layers.xavier_initializer()
bn_params = {
    'is_training': train_mode,
    'decay': 0.9,
    'updates_collections': None
}

# We can build short code using 'arg_scope' to avoid duplicate code
# same function with different arguments
with arg_scope([fully_connected],
               activation_fn=tf.nn.relu,
               weights_initializer=xavier_init,
               biases_initializer=None,
               normalizer_fn=batch_norm,
               normalizer_params=bn_params
               ):
    hidden_layer1 = fully_connected(X, hidden_output_size, scope="h1")
    h1_drop = dropout(hidden_layer1, keep_prob, is_training=train_mode)

    hidden_layer2 = fully_connected(h1_drop, hidden_output_size, scope="h2")
    h2_drop = dropout(hidden_layer2, keep_prob, is_training=train_mode)

    hidden_layer3 = fully_connected(h2_drop, hidden_output_size, scope="h3")
    h3_drop = dropout(hidden_layer3, keep_prob, is_training=train_mode)

    hidden_layer4 = fully_connected(h3_drop, hidden_output_size, scope="h4")
    h4_drop = dropout(hidden_layer4, keep_prob, is_training=train_mode)

    hypothesis = fully_connected(h4_drop, final_output_size, activation_fn=None, scope="hypothesis")

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(train_x_np) / batch_size)

    for i in range(0, len(train_x_np), batch_size):
        batch_xs = train_x_np[i:i + batch_size]
        batch_ys = train_y_np[i:i + batch_size]

        feed_dict_train = {X: batch_xs, Y: batch_ys, train_mode: True}
        feed_dict_cost = {X: batch_xs, Y: batch_ys, train_mode: False}

        opt = sess.run(optimizer, feed_dict=feed_dict_train)
        c = sess.run(cost, feed_dict=feed_dict_cost)
        avg_cost += c / total_batch

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost))
    # print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict={
    X: test_x_np, Y: test_y_np, train_mode: False}))
#Accuracy: 0.8228

# Get one and predict
r = random.randint(0, len(test_x_np) - 1)
print("Label: ", sess.run(tf.argmax(test_y_np[r:r + 1], 1)))

print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1),
    feed_dict={X: test_x_np[r:r + 1], train_mode: False}))

r = random.randint(0, len(test_x_np) - 1)
print("Label: ", sess.run(tf.argmax(test_y_np[r:r + 1], 1)))

print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1),
    feed_dict={X: test_x_np[r:r + 1], train_mode: False}))

r = random.randint(0, len(test_x_np) - 1)
print("Label: ", sess.run(tf.argmax(test_y_np[r:r + 1], 1)))

print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1),
    feed_dict={X: test_x_np[r:r + 1], train_mode: False}))