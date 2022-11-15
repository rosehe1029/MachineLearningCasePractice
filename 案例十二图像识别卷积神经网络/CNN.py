'''
Author: philosophylato
Date: 2022-11-11 17:05:30
LastEditors: philosophylato
LastEditTime: 2022-11-11 17:17:19
Description: your project
version: 1.0
'''
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='案例十二图像识别卷积神经网络/data',train=True,download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset=torchvision.datasets.CIFAR10(root='案例十二图像识别卷积神经网络/data',train=False,download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


import matplotlib.pyplot as plt

#构建展示图片的函数
def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#从数据迭代器中读取一张图片
dataiter=iter(trainloader)
images,labels=dataiter.next()

#展示图片
imshow(torchvision.utils.make_grid(images))
#打印标签
print(''.join('%5s'% classes[labels[j]] for j in range(4)))

