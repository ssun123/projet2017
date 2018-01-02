from collections import Counter

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data=data[1:]
    return data


train_data = read_data('../fashion_review_crawl/crawl_g_prod_prm19.txt') #지마켓 프리미엄리뷰
#train_data = read_data('../fashion_review_crawl/crawl_g_prod_gen19.txt') #지마켓 일반리뷰

#train_data = read_data('test_pre_proc.txt')
print(train_data[1][2])
print(train_data[2])
res = open('pre_proc_ggen_1221.txt','a')
#Nouns이랑 adjective만 쓰고, 긍정부정 1/0으로 치환해서 다시
from konlpy.tag import Twitter
pos_tagger = Twitter()
#롯데마트 상품기준
#default = ['품질이 정말 좋아요','배송이 빨라서 좋아요','자주 구매하는 상품입니다','늘 구매하는 상품 좋아요', '맘에듭니다','맛있어요','늘 구매하는 제품입니다','맘에듭니다맛있어요','배송 빠르고 편해요','괜찮아요','좋아요 만족']
#지마켓
default = ['추천합니다','배송이 보통입니다','배송이 빠릅니다','적극추천 합니다','잘 받았습니다','보통입니다','추천 안 합니다','배송빠름','배송 빨라요','좋아요','잘 입을게요','배송 빠르고','배송이 느립니다']
pos_default = [pos_tagger.pos(i) for i in default]


def tokenize3(doc):
    # 조사,구두점 제외
    list=[]
    sc = '2'
    pos = pos_tagger.pos(doc)
    p_cnt,n_cnt,kp_cnt,ap_cnt,noun_cnt,adj_cnt = 0,0,0,0,0,0
    p_len,n_len,kp_len,ap_len = 0,0,0,0
    dup_noun, dup_adj = 0,0
    prev_noun,prev_adj = '',''
    df = 0
   # print(pos)
    for item in pos:
        word,tag = item
        if tag in ['Punctuation']:
            p_len = len(word) #의미 없는 단어가 몇개나 적혀있는지 제대로 쓰인 (~!,.) 등을 거르기 위함
            if(p_len > 3):
                p_cnt = p_cnt+1 #각각 별로 문장내 카운트
            pass
        elif tag in 'Number':
            n_len = len(word)
            n_cnt = n_cnt+1
            pass
        elif tag in 'KoreanParticle':
            kp_len = len(word)
            kp_cnt = kp_cnt+1
            pass
        elif tag in 'Alpha':
            ap_len = len(word)
            ap_cnt = ap_cnt+1
            pass
        elif tag in 'Noun':
            if(prev_noun == word):
                #print("중복단어 연달아서 나옴")
                dup_noun = dup_noun+1
            prev_noun = word
            noun_cnt = noun_cnt+1
            # 같은 명사가 중복되어 들어오는 경우 체크, 명사 등장 횟수 체크
            list.append('/'.join(item))
        elif tag in 'Adjective':
            if (prev_adj == word):
                #print("중복단어 연달아서 나옴")
                dup_adj = dup_adj+1
            prev_adj = word
            adj_cnt = adj_cnt + 1
            list.append('/'.join(item))
        else:   #그외 품사들
            list.append('/'.join(item))

    #디폴트 문장 확인
    if(pos== pos_default[i] for i in range(0, len(pos_default))):
        for i in pos_default:
            p_tag1 = [(word, tag) for word, tag in i if (tag in ('Noun', 'Adjective', 'Verb'))] #명사,형용사,동사만 이용해서 유사도 계산
            p_tag2 = [(word, tag) for word, tag in pos if (tag in ('Noun', 'Adjective', 'Verb'))]
            bow1 = Counter(p_tag1)
            bow2 = Counter(p_tag2)
            j_index = sum((bow1 & bow2).values()) / sum((bow1 | bow2).values()) # 자카드 유사도 계산
            if(j_index>0.4):
                df = 1
                break
                print(doc)

    if(dup_noun>=2)|(dup_adj >=2):
       # print(doc+', 중복문장')
        res.write(sc+', '+doc+', 1, 중복문장\n')
        score =1
    elif(noun_cnt < 1)|(adj_cnt<1):
        #print(doc+', 명사/형용사 없음')
        res.write(sc+', '+doc + ', 1, 명사/형용사 없음\n')
        score = 1
   # elif(p_len>4)|(n_len>3)|(kp_len>3)|(ap_len>3):
        #print(doc+', 부호나 자음나열')
    #    res.write(sc+', '+doc + ', 1, 부호나 자음나열\n')
     #   score = 1
   # elif(Counter(pos) == pos_default[i] for i in range(0,len(pos_default))):
    elif(df == 1):
        #print(doc+', 기본문장')
        res.write(sc+', '+doc+', 1, 기본문장\n')
        score = 1
    elif(len(doc)<8):
        #print(doc+', too short')
        res.write(sc+', '+doc + ', 1, too short\n')
        score=1
    else:
        #print(doc+', ok')
        res.write(sc+', '+doc + ', 0, ok\n')
        score = 0
        # 의미 있는 리뷰는 0, 의미 없는 리뷰는 1
        #리스트와 함께 의미없음에 대한 flag추가 ---> 의미 없는 리뷰로
    return (list,score)



train_docs = [(tokenize3(row[2])) for row in train_data] # g마켓 프라임 상품평 기준
#train_docs = [(tokenize3(row[3])) for row in train_data] # g마켓 일반 상품평 기준
print('fin')


#train_docs = [(tokenize3(row[5])) for row in train_data]  #pork
#train_docs = [(tokenize3(row[5]), score3(row[0])) for row in train_data if(row[0]=='0')| (row[0]=='1')]  #pork
#test_docs = [(tokenize3(row[1]), score3(row[0])) for row in test_data  if(row[0]=='0')| (row[0]=='1')]   #pork2
#test_docs = [(tokenize3(row[5]), score(row[0])) for row in test_data ]  #pork



#docs1 = proc(train_docs)