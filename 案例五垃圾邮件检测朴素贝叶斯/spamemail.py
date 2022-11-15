# -*- coding: utf-8 -*-
'''
Author: philosophylato
Date: 2022-11-11 16:15:12
LastEditors: philosophylato
LastEditTime: 2022-11-11 16:32:42
Description: your project
version: 1.0
'''

# 读取所有训练数据并按照空格分隔，保存在一个列表里返回
def load_file(path):
    cab = []
    for i in range(1, 25):
        data = open(path % i)
        for line in data.readlines():
            cab.append(line.strip().split(','))
    cab_f = []
    for i in range(len(cab)):
        for j in range(len(cab[i])):
            if cab[i][j] != '':
                cab_f.append(cab[i][j].strip())
    cab_final = []
    for i in cab_f:
        for j in i.split(' '):
            cab_final.append(j)
    return cab_final


# 朴素贝叶斯分类器
def bayes(sample):
    path1 = 'Emails\\Training\\normal\\%d.txt'
    path2 = 'Emails\\Training\\spam\\%d.txt'
    normal_data = load_file(path1)
    spam_data = load_file(path2)
    # 计算p(x|C1)=p1与p(x|C2)=p2
    p1 = 1.0
    p2 = 1.0
    for i in range(len(sample)):
        x = 0.0
        for j in normal_data:
            if sample[i] == j:
                x = x + 1.0
        p1 = p1 * ((x + 1.0) / (len(normal_data) + 2.0))  # 拉普拉斯平滑

    for i in range(len(sample)):
        x = 0.0
        for j in spam_data:
            if sample[i] == j:
                x = x + 1.0
        p2 = p2 * ((x + 1.0) / (len(spam_data) + 2.0))  # 拉普拉斯平滑

    pc1 = len(normal_data) / (len(normal_data) + len(spam_data))
    pc2 = 1 - pc1

    if p1 * pc1 > p2 * pc2:
        return 'normal'
    else:
        return 'spam'


# 测试
def test(path):
    data = open(path)
    cab = []
    for line in data.readlines():
        cab.append(line.strip().split(','))
    cab_f = []
    for i in range(len(cab)):
        for j in range(len(cab[i])):
            if cab[i][j] != '':
                cab_f.append(cab[i][j].strip())
    cab_final = []
    for i in cab_f:
        for j in i.split(' '):
            cab_final.append(j)
    return bayes(cab_final)


if __name__ == '__main__':
    print(test('案例五垃圾邮件检测朴素贝叶斯\Emails\test\normal.txt'))
    print(test('Emails\\test\\spam.txt'))
    sum1 = 0
    sum2 = 0
    # 再试试训练集
    for i in range(1, 25):
        if test('Emails\\Training\\normal/%d.txt' % i) == 'normal':
            sum1 = sum1 + 1
    for i in range(1, 25):
        if test('Emails\\Training\\spam/%d.txt' % i) == 'spam':
            sum2 = sum2 + 1
    print('normal分类正确率：', sum1 / 24)
    print('spam分类正确率：', sum2 / 24)

