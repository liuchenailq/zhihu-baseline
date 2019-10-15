# [智源 - 看山杯 专家发现算法大赛 2019 ](https://www.biendata.com/competition/zhihu2019/)

## 文件说明

features.py : 从原始数据集中生成特征   
Main.py: 运行模型并取得预测结果

## 模型说明

  
模型：DeepFM

运行环境： DeepCTR-Torch (https://github.com/shenweichen/DeepCTR-Torch)

分数（AUC）：线下0.6903   线上0.691804111317667 （没有采用五折交叉验证，可能还没收敛（毕竟计算资源有限））

特征说明
 
**1.用户特征**  
用户原始特征：gender、frequency、A1、...  
用户关注和感兴趣的topics数目

**2.问题特征** 
问题标题的字、词计数   
问题描述的字、词计数   
问题绑定的topic数目

**3.用户问题交叉特征**  
用户关注、感兴趣的话题和问题绑定的话题交集计数   
邀请距离问题创建的天数



**最后，有意向一起参赛的伙伴可添加个人微信号（a2422701543）**

