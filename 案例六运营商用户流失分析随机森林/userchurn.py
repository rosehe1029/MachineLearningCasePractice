'''
Author: philosophylato
Date: 2022-11-15 15:52:42
LastEditors: philosophylato
LastEditTime: 2022-11-15 15:52:47
Description: your project
version: 1.0
'''
#随机森林分类

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # 导入sklearn库的RandomForestClassifier函数
from sklearn.model_selection import train_test_split
from sklearn import metrics  # 分类结果评价函数

data = pd.read_csv('./wm.csv', index_col=False, encoding='gb18030')
print(data)

'''   
    y    x1    x2    x3
0   1  0.697  0.460   0
1   1  0.774  0.376   0
2   1  0.634  0.264   0
3   1  0.608  0.318   0
4   1  0.556  0.215   0
5   1  0.403  0.237   1
6   1  0.481  0.149   1
7   1  0.437  0.211   1
8   0  0.666  0.091   1
9   0  0.243  0.267   2
10  0  0.245  0.057   2
11  0  0.343  0.099   2
12  0  0.639  0.161   0
13  0  0.657  0.198   0
14  0  0.360  0.370   1
15  0  0.593  0.042   2
16  0  0.719  0.103   1
'''

x = data[['x1', 'x2', 'x3']]  # 特征
y = data['y']  # 标签

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.8) 
print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = RandomForestClassifier()  # 实例化模型RandomForestClassifier
model.fit(x_train, y_train)  # 在训练集上训练模型
print(model)  # 输出模型RandomForestClassifier

# 在测试集上测试模型
expected = y_test  # 测试样本的期望输出
predicted = model.predict(x_test)  # 测试样本预测

# 输出结果
print(metrics.classification_report(expected, predicted))  # 输出结果，精确度、召回率、f-1分数
print(metrics.confusion_matrix(expected, predicted))  # 混淆矩阵

auc = metrics.roc_auc_score(y_test, predicted)
accuracy = metrics.accuracy_score(y_test, predicted)  # 求精度
print("Accuracy: %.2f%%" % (accuracy * 100.0))

