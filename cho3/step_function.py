import numpy as np
from matplotlib import pyplot as plt
from mpmath import plot

"""
    激活函数1：阶跃函数
        以阈值为界，超过阈值就切换为输出。“急剧变化”
"""
def step_function(x):
    return np.array(x > 0, dtype=np.uint)

x  = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-1.1, 1.1) #指定图中绘制y轴的范围
plt.xlabel('x')
plt.ylabel('y')
plt.show()
