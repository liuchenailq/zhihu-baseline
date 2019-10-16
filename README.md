# [智源 - 看山杯 专家发现算法大赛 2019 ](https://www.biendata.com/competition/zhihu2019/)

## 文件说明

features.py : 从原始数据集中生成特征   
deepfm.py: 运行deepfm模型并取得预测结果  
textdeepfm.py: 运行textdeepfm模型并取得预测结果  
utils.py: deepctr运行环境

## 模型说明

"""   
模型一：DeepFM

运行环境： DeepCTR-Torch (https://github.com/shenweichen/DeepCTR-Torch)

分数（AUC）：线下0.6903   线上0.691804111317667 （没有采用五折交叉验证，可能还没收敛，设置epoch大些）

特征说明
 
**1.用户特征**  
用户原始特征：gender、frequency、A1、...  
用户关注和感兴趣的topics数目

**2.问题特征 ** 
问题标题的字、词计数   
问题描述的字、词计数   
问题绑定的topic数目

**3.用户问题交叉特征**  
用户关注、感兴趣的话题和问题绑定的话题交集计数   
邀请距离问题创建的天数

"""


"""  
模型二：TextDeepFM

模型说明：在DeepFM基础上增加文本特征，将用户感兴趣的话题作为用户的embedding，将问题绑定的话题作为问题的embedding。
文本特征利用TextCNN作为特征提取器，提取的特征和原始特征向量拼接一起传给DNN训练。

分数（AUC）：线下  线上  （没有采用五折交叉验证）


"""

**最后，哪位大佬愿意带带小弟的请添加个人微信号**
![image](https://img-blog.csdnimg.cn/20191016095835892.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMzc0NTQ5,size_16,color_FFFFFF,t_70)

