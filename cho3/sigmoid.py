import numpy as np
from matplotlib import pyplot as plt

"""
    一般为隐藏层的激活函数   
    激活函数：
        目的：将输入信号的总和转换为输出信号
            e.g.   输入信号：a = w1x1 + w2x2 + b
                   激活函数：h()
                   输出：y = h(a)
                   
        作用：决定如何来激活输入信号的总和、
        为啥只能用线性函数？：
            防止多层塌缩为一层，每一层不会被"合并抵消"，网络获得了逐层复合，逐层抽象的能力。
        --> 万能逼近定理（Universal Approximation Theorem）：一个含非线性激活的单隐藏层网络，只要神经元足够多，就能以任意精度逼近任意连续函数。
    
    激活函数2：sigmoid 函数
        1.相对于阶跃函数，非常平滑，随着输入信号做连续变化。 
        2.随着输入信号的增大而增大，输出值的范围为 0-1 之间
"""

def sigmoid(x):
    """

    :param x: 传入张量
    :return: 通过广播机制实现计算
    """
    return 1/(1 + np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()