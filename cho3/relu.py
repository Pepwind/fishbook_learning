import numpy as np
from matplotlib import pyplot as plt

"""
    神经网络中的激活函数使用平滑变化的sigmoid函数或ReLU函数。
"""

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()