"""
模型：DeepFM

运行环境： DeepCTR-Torch (https://github.com/shenweichen/DeepCTR-Torch)


特征说明

1.用户特征
用户原始特征：gender、frequency、A1、...
用户关注和感兴趣的topics数目

2.问题特征
问题标题的字、词计数
问题描述的字、词计数
问题绑定的topic数目

3.用户问题交叉特征
用户关注、感兴趣的话题和问题绑定的话题交集计数
邀请距离问题创建的天数

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

""" 运行DeepFM """

path = 'E:\\competition\\看山杯\\data\\data_set_0926\\'
train = pd.read_csv(path + 'train.txt', sep='\t', nrows=5000000)
test = pd.read_csv(path + 'test.txt', sep='\t')

# 测试
# train = train[0:10000]
# test = test[0:10000]

# print(train.head())
data = pd.concat([train, test], ignore_index=True, sort=False)
# print(data.head())

# 盐值分数转换
def salt_score_trans(x):
    if x <= 0:
        return x
    if 1 <= x <= 10:
        return 1
    if 10 < x <= 100:
        return 2
    if 100 < x <= 200:
        return 3
    if 200 < x <= 300:
        return 4
    if 300 < x <= 400:
        return 5
    if 400 < x <= 500:
        return 6
    if 500 < x <= 600:
        return 7
    if x > 600:
        return 8

data['m_salt_score'] = data['m_salt_score'].apply(lambda x: salt_score_trans(x))

# 单值类别特征
fixlen_category_columns = ['m_sex', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA', 'm_categoryA',
                           'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE', 'm_num_interest_topic', 'num_topic_attention_intersection',
                           'q_num_topic_words', 'num_topic_interest_intersection', 'm_salt_score'
                         ]
# 数值特征
fixlen_number_columns = ['m_num_atten_topic', 'q_num_title_chars_words', 'q_num_desc_chars_words', 'q_num_desc_words', 'q_num_title_words',
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


fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in fixlen_category_columns] + [DenseFeat(feat, 1,)for feat in fixlen_number_columns]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train = data[~data['label'].isnull()]
test = data[data['label'].isnull()]

train, vaild = train_test_split(train, test_size=0.2)
train_model_input = {name: train[name] for name in feature_names}
vaild_model_input = {name: vaild[name] for name in feature_names}


device = 'cuda:0'
"""第一步：初始化一个模型类"""
model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=device)

"""第二步：调用compile()函数配置模型的优化器、损失函数、评价函数"""
model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"],)

"""第三步：调用fit()函数训练模型"""
model.fit(train_model_input, train[target].values, batch_size=8192, epochs=10, validation_data=[vaild_model_input, vaild[target].values], verbose=1, model_cache_path='E:\\competition\\看山杯\\models\\deepfm.model')

"""预测"""
test_model_input = {name: test[name] for name in feature_names}
pred_ans = model.predict(test_model_input, 8192)
pred_ans = pred_ans.reshape(pred_ans.shape[0])
result = test[['questionID', 'memberID', 'time']]
result['result'] = pred_ans
result.to_csv(path + 'submit.txt', sep='\t', index=False)   # 注意提交的时候请把表头去掉