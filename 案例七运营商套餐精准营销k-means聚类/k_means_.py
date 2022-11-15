'''
Author: philosophylato
Date: 2022-11-15 16:20:06
LastEditors: philosophylato
LastEditTime: 2022-11-15 19:26:17
Description: your project
version: 1.0
'''
from sklearn.datasets import load_iris
import xlwt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#print(iris)

iris=load_iris()
iris_data=iris['data']
iris_target=iris['target']

iris_names=iris['feature_names']
print("是骡子是马打印出来看看就知道了：\n",'第一个',iris_data,'\n','第二个',iris_target,'\n','第三个',iris_names)
# 依次是：花瓣长度、花瓣宽度、花萼长度、花萼宽度。

## Z分数标准化标准化
data_zs=(iris_data-iris_data.mean())/iris_data.std()
## 也可以自定义函数minmax标准化、或者现成的函数
scale=MinMaxScaler().fit(iris_data)
iris_datascale=scale.transform(iris_data)

kmeans=KMeans(n_clusters=3,random_state=123,n_jobs=4).fit(iris_datascale)
result=kmeans.predict([[5.6,2.8,4.9,2.0]])
## 这里有点小问题，就是预测的数据需要使用和训练数据同样的标准化才行。
print(result)

#简答打印结果
r1=pd.Series(kmeans.labels_).value_counts()
r2=pd.DataFrame(kmeans.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
r.columns=list(iris_names)+[u'类别数目']
print(r)

#详细输出原结果
r_new=pd.concat([pd.DataFrame(iris_data),pd.Series(kmeans.labels_)],axis=1)
r_new.columns=list(iris_names)+[u'类别数目']
r_new.to_excel("path")## 自定义一个路径，保存在excel里面

from sklearn.datasets import load_iris
import xlwt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

print(iris)

iris=load_iris()
iris_data=iris['data']
iris_target=iris['target']

iris_names=iris['feature_names']
print("是骡子是马打印出来看看就知道了：\n",'第一个',iris_data,'\n','第二个',iris_target,'\n','第三个',iris_names)
# 依次是：花瓣长度、花瓣宽度、花萼长度、花萼宽度。

## Z分数标准化标准化
data_zs=(iris_data-iris_data.mean())/iris_data.std()
## 也可以自定义函数minmax标准化、或者现成的函数
scale=MinMaxScaler().fit(iris_data)
iris_datascale=scale.transform(iris_data)

kmeans=KMeans(n_clusters=3,random_state=123,n_jobs=4).fit(iris_datascale)
result=kmeans.predict([[5.6,2.8,4.9,2.0]])
## 这里有点小问题，就是预测的数据需要使用和训练数据同样的标准化才行。
print(result)

#简答打印结果
r1=pd.Series(kmeans.labels_).value_counts()
r2=pd.DataFrame(kmeans.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
r.columns=list(iris_names)+[u'类别数目']
print(r)

#详细输出原结果
r_new=pd.concat([pd.DataFrame(iris_data),pd.Series(kmeans.labels_)],axis=1)
r_new.columns=list(iris_names)+[u'类别数目']
r_new.to_excel("path")## 自定义一个路径，保存在excel里面



