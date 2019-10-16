"""
TextDeepFM模型
"""
import pandas as pd
import numpy as np
from collections import OrderedDict, namedtuple
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from itertools import chain
import torch
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import math
import itertools
from utils import *
from gensim.models import KeyedVectors
import pickle
import gc


""" 运行TextDeepFM """

path = 'E:\\competition\\看山杯\\data\\data_set_0926\\'
train = pd.read_csv(path + 'train.txt', sep='\t')
test = pd.read_csv(path + 'test.txt', sep='\t')
data = pd.concat([train, test], ignore_index=True, sort=False)


# print(data[['m_num_interest_topic', 'q_num_topic_words', 'm_num_atten_topic']].describe())   maxlen  10 ,13, 100

"""单值类别特征、数值特征处理"""
# 单值类别特征
fixlen_category_columns = ['m_sex', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA', 'm_categoryA',
                           'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE', 'm_num_interest_topic', 'num_topic_attention_intersection',
                           'q_num_topic_words', 'num_topic_interest_intersection'
                         ]
# 数值特征
fixlen_number_columns = ['m_salt_score', 'm_num_atten_topic', 'q_num_title_chars_words', 'q_num_desc_chars_words', 'q_num_desc_words', 'q_num_title_words',
                         'days_to_invite'
                        ]

target = ['label']

data[fixlen_category_columns] = data[fixlen_category_columns].fillna('-1', )
data[fixlen_number_columns] = data[fixlen_number_columns].fillna(0, )

for feat in fixlen_category_columns:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[fixlen_number_columns] = mms.fit_transform(data[fixlen_number_columns])

"""文本特征处理"""
# 截断补齐
def pad_sequences(x, maxlen):
    x = str(x)
    if x == 'nan':
        return [0] * maxlen
    words = []
    for t in x.split(" "):
        words.append(int(t))
    words = words[0:maxlen]
    words += [0] * (maxlen-len(words))
    return words

def deal_text(values, maxlen=10):
    temp = []
    for text in values:
        temp.append(pad_sequences(text, maxlen))
    return np.array(temp)


# 初始化词向量矩阵
topic2index = pickle.load(open(path+"topic2index.pick", 'rb'))
topic_word_vector = KeyedVectors.load_word2vec_format(path + "topic_vectors_64d.txt", binary=False)
embedding_matrix = np.random.uniform(size=(len(topic2index) + 1, 64))
miss_count = 0
for word, index in topic2index.items():
    try:
        word_vector = None
        word_vector = topic_word_vector[word]
        embedding_matrix[index] = word_vector
    except:
        miss_count += 1
print(miss_count, " 个词没有词向量")

text_feature_columns = [TextFeat('q_topic_words', len(topic2index) + 1, 10),
                        # TextFeat('m_attention_topics_words', len(topic2index) + 1, 10),
                        TextFeat('m_interested_topics_words', len(topic2index) + 1, 10)
                        ]

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in fixlen_category_columns] + [DenseFeat(feat, 1,)for feat in fixlen_number_columns]

dnn_feature_columns = fixlen_feature_columns + text_feature_columns
linear_feature_columns = fixlen_feature_columns + text_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train = data[~data['label'].isnull()]
test = data[data['label'].isnull()]

del train['time']
del train['memberID']
del train['questionID']
del train['m_attention_topics_words']
del test['m_attention_topics_words']
gc.collect()


train, vaild = train_test_split(train, test_size=0.2)
train_model_input = {name: train[name] for name in feature_names}
vaild_model_input = {name: vaild[name] for name in feature_names}
train_model_input['q_topic_words'] = deal_text(train_model_input['q_topic_words'])
# train_model_input['m_attention_topics_words'] = deal_text(train_model_input['m_attention_topics_words'])
train_model_input['m_interested_topics_words'] = deal_text(train_model_input['m_interested_topics_words'])
vaild_model_input['q_topic_words'] = deal_text(vaild_model_input['q_topic_words'])
# vaild_model_input['m_attention_topics_words'] = deal_text(vaild_model_input['m_attention_topics_words'])
vaild_model_input['m_interested_topics_words'] = deal_text(vaild_model_input['m_interested_topics_words'])

gc.collect()


device = 'cuda:0'
"""第一步：初始化一个模型类"""
model = TextDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary',
                   l2_reg_embedding=1e-5, device=device, use_textcnn=True, filter_sizes=[1, 2, 3], num_filters=64, text_vocab_size=len(topic2index)+1, text_embedding_size=64, text_embedding_matrix=embedding_matrix, cnn_dropout=0.0)

"""第二步：调用compile()函数配置模型的优化器、损失函数、评价函数"""
model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"],)

"""第三步：调用fit()函数训练模型"""
model.fit(train_model_input, train[target].values, batch_size=2048, epochs=40, validation_data=[vaild_model_input, vaild[target].values], verbose=1, model_cache_path='E:\\competition\\看山杯\\models\\textdeepfm.model', early_stop=2)

"""预测"""
test_model_input = {name: test[name] for name in feature_names}
test_model_input['q_topic_words'] = deal_text(test_model_input['q_topic_words'])
# test_model_input['m_attention_topics_words'] = deal_text(test_model_input['m_attention_topics_words'])
test_model_input['m_interested_topics_words'] = deal_text(test_model_input['m_interested_topics_words'])
pred_ans = model.predict(test_model_input, 2048)
pred_ans = pred_ans.reshape(pred_ans.shape[0])
result = test[['questionID', 'memberID', 'time']]
result['result'] = pred_ans
result.to_csv(path + 'submit.txt', sep='\t', index=False)   # 注意提交的时候请把表头去掉