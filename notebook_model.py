#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

opinions_df = pd.read_csv('dataset2300.csv', encoding='CP949')
opinions_df.head(3)


# In[2]:


from sklearn.model_selection import train_test_split

# 사전 데이터 가공 후 학습과 테스트 데이터 세트를 반환하는 함수
def get_train_test_dataset(df=None):
    # 인자로 입력된 DataFrame 복사
    df_copy = df.copy()
    # DataFrame의 맨 마지막 칼럼이 레이블, 나머지는 피처들
    X_features = df_copy.iloc[:,0]
    y_target = df_copy.iloc[:,1]
    # train_test_split()으로 학습과 테스트 데이터 분할. stratify=y_target으로 Stratified기반 분할
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    # 학습과 테스트 데이터 세트 반환
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_dataset(opinions_df)


# In[3]:


print('학습 데이터 레이블 값 비율')
print(y_train.value_counts()/y_train.shape[0] * 100)
print('테스트 데이터 레이블 값 비율')
print(y_test.value_counts()/y_test.shape[0] * 100)


# In[4]:


train_set = pd.concat([X_train, y_train], axis =1)
test_set = pd.concat([X_test, y_test], axis =1 )
train_set = train_set.reset_index()
test_set= test_set.reset_index()


# In[6]:


train_set


# # 최빈 단어 추출

# In[94]:


from konlpy.tag import Okt

okt = Okt()

s = '이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요~^^'
okt.pos(s)


# In[95]:


import re

def tokenize(doc):
    # 한글 자음, 모음 제거
    doc = re.sub(pattern='([ㄱ-ㅎㅏ-ㅣ]+)', repl='', string=doc)
    # 특수기호 제거
    doc = re.sub(pattern='[^\w\s]', repl='', string=doc)
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    doc = okt.pos(doc, norm=True, stem=True)
    # 2글자 이상만 포함
    token = []
    for i in doc:
        if len(i[0])>1:
            token.append(i)
    
    return ['/'.join(t) for t in token]


# In[104]:


train_docs = [(tokenize(row['content']), row['star']) for idx, row in tqdm(train_set.iterrows())]
test_docs = [(tokenize(row['content']), row['star']) for idx, row in tqdm(test_set.iterrows())]

# 위에서 만든 데아터에서 긍/부정을 제외하고 token에 넣어준다.
# [[a],b] 에서 a만 넣는다고 생각하면 됨
tokens = [t for d in train_docs for t in d[0]]
tokens[:10]


# In[97]:


import nltk
#nltk라이브러리를 통해서 텍스트 데이터 나열
text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))            

# 출현 빈도가 높은 상위 토큰 10개
print(text.vocab().most_common(10))


# In[98]:


#단어 빈도수가 높은 10000개의 단어만 사용하여 각 리뷰 문장마다의 평가지표로 삼는다.
selected_words = [f[0] for f in text.vocab().most_common(10000)]

#term_frequency()함수는 위에서 만든 selected_words의 갯수에 따라서 각 리뷰와 매칭하여 상위 텍스트가 
#각 리뷰에 얼만큼 표현되는지 빈도를 만들기 위한 함수
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]


# In[ ]:


train_x


# # 예측 모델 구축

# In[99]:


#모델링을 하기 위해 리스트로 되어 있는 데이터 형식을 array로 바꿔주고 dtype도 실수로 바꿔준다.
import numpy as np

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')


# In[100]:


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

#tensorflow.keras를 활용하여 모델의 층 입력하기
model = models.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#모델 생성
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])


# In[101]:


#모델 학습
model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test, y_test)

#예측 결과
results


# In[102]:


def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))


# In[103]:


predict_pos_neg("2015년식 삼성노트북을 교체했습니다 그래픽카드와 화면크기가 아쉬웠는데 아주만족합니다 터치스크린은 최고내요")
predict_pos_neg("리뷰보고 괜찮아보여서 샀는데 엄청 느리네요. 게임하면 버벅 거리고 돈 더 쓸 걸 그랬어요..")
predict_pos_neg("16인치 소식으로 너무 아쉽게됐지만 잘 쓰고 있어요")


# In[ ]:




