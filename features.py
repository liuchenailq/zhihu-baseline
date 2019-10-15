"""
用户特征
1. 用户原始特征：m_sex、m_access_frequencies、......
2. 用户关注和感兴趣的topic数

问题特征：
2. 标题字 词计数
3. 描述字 词计数
4. topic计数

用户问题交叉特征
1. 用户关注、感兴趣的话题和问题绑定的话题交集计数
2. 邀请发送时间较问题创建时间的距离
"""
import pandas as pd
import gc

path = 'E:\\competition\\看山杯\\data\\data_set_0926\\'
member_path = path + 'member_info_0926.txt'   # 用户信息路径
question_path = path + 'question_info_0926.txt'  # 问题信息路径
invito_path = path + 'invite_info_0926.txt'   # 邀请记录路径

def count(x):
    x = str(x)
    if x == '-1':
        return 0
    else:
        return len(x.split(","))

user_feature = pd.read_csv(open(member_path, "r", encoding='utf-8'), sep='\t', header=None,
                           names=['memberID', 'm_sex', 'm_keywords', 'm_amount_grade', 'm_hot_grade', 'm_registry_type',
                                  'm_registry_platform', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD',
                                  'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE',
                                  'm_salt_score', 'm_attention_topics', 'm_interested_topics'])

user_feature['m_num_atten_topic'] = user_feature['m_attention_topics'].apply(lambda x: count(x))
user_feature['m_num_interest_topic'] = user_feature['m_interested_topics'].apply(lambda x: count(x))


question_feature = pd.read_csv(open(question_path, "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'q_createTime', 'q_title_chars', 'q_title_words', 'q_desc_chars',
                               'q_desc_words', 'q_topic_IDs'])

question_feature['q_num_title_chars_words'] = question_feature['q_title_chars'].apply(lambda x: count(x))
question_feature['q_num_desc_chars_words'] = question_feature['q_desc_chars'].apply(lambda x: count(x))
question_feature['q_num_desc_words'] = question_feature['q_desc_words'].apply(lambda x: count(x))
question_feature['q_num_title_words'] = question_feature['q_title_words'].apply(lambda x: count(x))
question_feature['q_num_topic_words'] = question_feature['q_topic_IDs'].apply(lambda x: count(x))

"""训练集"""
invite_info_data = pd.read_csv(open(invito_path, "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'memberID', 'time', 'label'])

invite_info_data = invite_info_data.merge(user_feature, how='left', on='memberID')
invite_info_data = invite_info_data.merge(question_feature, how='left', on='questionID')

invite_info_data['day'] = invite_info_data['time'].apply(lambda x: int(x.split('-')[0][1:]))

"""用户关注、感兴趣的话题和问题绑定的话题交集计数"""
def intersection(x, y):
    x = str(x)
    y = str(y)
    if x == '-1' or y == '-1':
        return 0
    return len(set(x.split(",")) & set(y.split(",")))

invite_info_data['num_topic_attention_intersection'] = invite_info_data.apply(lambda row: intersection(row['q_topic_IDs'], row['m_attention_topics']), axis=1)
invite_info_data['num_topic_interest_intersection'] = invite_info_data.apply(lambda row: intersection(row['q_topic_IDs'], row['m_interested_topics']), axis=1)

invite_info_data['q_day'] = invite_info_data['q_createTime'].apply(lambda x: int(x.split('-')[0][1:]))
invite_info_data['days_to_invite'] = invite_info_data['day'] - invite_info_data['q_day']
invite_info_data.to_csv(path + "train.txt", sep='\t', index=False)

del invite_info_data
gc.collect()

"""测试集"""
evaluate_path = path + 'invite_info_evaluate_1_0926.txt'   # 测试集路径
evaluate_info_data = pd.read_csv(open(evaluate_path, "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'memberID', 'time'])
evaluate_info_data['day'] = evaluate_info_data['time'].apply(lambda x: int(x.split('-')[0][1:]))
evaluate_info_data = evaluate_info_data.merge(user_feature, how='left', on='memberID')
evaluate_info_data = evaluate_info_data.merge(question_feature, how='left', on='questionID')
evaluate_info_data['num_topic_attention_intersection'] = evaluate_info_data.apply(lambda row: intersection(row['q_topic_IDs'], row['m_attention_topics']), axis=1)
evaluate_info_data['num_topic_interest_intersection'] = evaluate_info_data.apply(lambda row: intersection(row['q_topic_IDs'], row['m_interested_topics']), axis=1)

evaluate_info_data['q_day'] = evaluate_info_data['q_createTime'].apply(lambda x: int(x.split('-')[0][1:]))
evaluate_info_data['days_to_invite'] = evaluate_info_data['day'] - evaluate_info_data['q_day']
evaluate_info_data.to_csv(path + "test.txt", sep='\t', index=False)



