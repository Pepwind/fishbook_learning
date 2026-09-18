import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
"""
    Python有 pickle这个便利的功能。这个功能可以将程序运行中的对
    象保存为文件。如果加载保存过的 pickle文件，可以立刻复原之前
    程序运行中的对象。 
"""

from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

"""获得MNIST数据集: 生成网络"""
"""
    正规化：把数据限定到某个范围内的处理称为正规化（normalization）
    预处理：
        对神经网络的输入数据 进行某种既定的转换称为预处理（pre-processing）
    --> 很多预处理都会考虑到数据的整体分布。比如，利用数据整体的均值或标准差，
        移动数据，使数据整体以 0为中心分布，或者进行正规化，把数据的延展控制在一定范围内。
    白化：将数据整体的分布形状均匀化的方法。
"""
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

"""
    读入保存在pickle文件sample_weight.pkl中的学习到的权重参数。
    该文件中以字典变量的形式保存了权重和偏置参数
"""
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


"""
    以numpy数组的形式输出各个标签对应的概率
"""
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3= network['b1'], network['b2'], network['b3']

    """
       a1、a2为隐藏层，a1有50个神经元，a2有100个神经元（50和100可设置为任意值）
       Z1、Z2为对应层的激活函数
    """
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) #(50,)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2) #(100,)

    a3 = np.dot(z2, W3) + b3

    """由于是分类问题，故输出层用softmax函数（可省略）"""
    y = softmax(a3)

    return y

"""获得MNIST数据集: 生成网络"""
x, t = get_data()
network = init_network()

accuracy_cnt = 0

"""
    用for语句逐一取出保存在x中的图像数据，用predict()函数进行分类。
    predict()函数以NumPy数组的形式输出各个标签对应的概率。
"""
for i in range(len(x)):
    y = predict(network, x[i]) #用参数进行预测
    p = np.argmax(y) #取出数组中的最大值的索引（第几个元素的概率最高）
    if p == t[i]:
        accuracy_cnt += 1
"""评价其识别精度"""
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
