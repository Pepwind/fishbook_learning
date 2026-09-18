import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

"""获取MNIST数据集"""
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=False, normalize=True)
    return x_test, t_test

"""读取权重参数"""
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

"""
    以numpy数组的形式输出各个标签对应的概率
"""
def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

"""获取MNIST数据集"""
# x:(10000, 784)
# t:(10000,)
x, t = get_data()

"""读取训练好的参数"""
network = init_network()

"""
    进行批处理:
        输入数据的集合称为批。通过以批为单位进行推理处理，能够实现
    高速的运算。
"""
batch_size = 100 #批数量
accuracy_cnt = 0 #正确数


"""
    1.range(start, end, step)
       >>>list( range(0, 10, 3) ) --> [0, 3, 6, 9]
    2.通过x[i:i+batch_size]从输入数据中抽出批数据
       >>>batch_size=100: 像x[0:100]、x[100:200]
    3.predict函数进行预测
    4.p为预测结果中概率最大者
"""
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) #统计预测值与实际值相等的数量

print(f"Accuracy:{str(float(accuracy_cnt / len(x)))}")

















