import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
import matplotlib.pyplot
from matplotlib import pyplot as plt
%matplotlib inline
 
class neuralNetwork :
    def __init__(self, inputnodes, hiddennodes, outputnodes,learningrate) :
#        定义输入层。隐藏层和输出层的节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
#       定义,系数矩阵，服从高斯分布的概率密度函数(正态分布)，numpy.random.normal(loc=0.0, scale=1.0, size=None)    
#       loc:float概率分布的均值，对应着整个分布的中心center
#       scale:float概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高
#      size:int or tuple of ints输出的shape，默认为None，只输出一个值我们更经常会用到
#      np.random.randn(size)所谓标准正太分布（μ=0, σ=1），对应于np.random.normal(loc=0, scale=1, size)
#   (self.onodes, self.hnodes)是得到的数组形状，pow(self.onodes, -0.5)对self.onodes进行-0.2开方
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
#       self.lr是学习率
        self.lr = learningrate
#     激活函数sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list) :
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
#       计算损失矩阵，目标值-实际输出值
        output_errors = targets - final_outputs
#       隐藏层的损失矩阵
        hidden_errors = numpy.dot(self.who.T, output_errors)
    #更新权重矩阵
        self.who += self.lr * numpy.dot((output_errors *final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors *hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
 
        #     返回输出结果
    def query(self, inputs_list) :
        inputs = numpy.array(inputs_list, ndmin=2).T
#         权重矩阵（系数矩阵）和输入进行点乘，得到隐藏层节点的数据矩阵
        hidden_inputs = numpy.dot(self.wih, inputs)
#     将得到的矩阵作用于激活函数,得到隐藏层节点的输出矩阵
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    pass
 
 
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file= open("F:/matlabCode/hands datasets/mnist_train.csv", 'r')
training_data_list= training_data_file.readlines()
training_data_file.close()
 
 
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass
 
 
 
 
test_data_file=open("F:/matlabCode/hands datasets/mnist_test.csv", 'r')
test_data_list=test_data_file.readlines()
test_data_file.close()
all_values=test_data_list[0].split(',')
 
 
scorecard = []
for record in test_data_list:
#     record是字符串，split将字符串进行切割并以list存储
    all_values = record.split(',')
#     将列表第一个元素得到，即标签值，并将其强制转换为int型
    correct_label = int(all_values[0])
#     print(correct_label, "correct label")
# 输入值是0-255，将其范围进行转换，使其范围在0.01-0.99之间
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
#     print(label, "network's answer")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
pass
 
scorecard_array = numpy.asarray(scorecard)
print(scorecard_array.sum())
# 输出成功率
print ("performance = ", scorecard_array.sum() /scorecard_array.size)
 
 
 import numpy
import matplotlib.pyplot
%matplotlib inline
# all_values= training_data_list[0].split(',')
image_array= numpy.asfarray(all_values[1:]).reshape((28,28))
# 将值转换在0到1之间
# scaled_input=(numpy.asfarray(all_values[1:])/255.0*0.99+0.01)
 
matplotlib.pyplot.imshow( image_array, cmap='Greys',interpolation=None)
 
 
 
# input_nodes=3
# hidden_nodes=3
# out_puts=3
# learning_rate=0.3
# n=neuralNetwork(input_nodes,hidden_nodes,out_puts,learning_rate)
# n.query([1.0,0.5,-1.5])