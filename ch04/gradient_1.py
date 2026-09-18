import numpy as np

"""
    matplotlib:python 里主流数据可视化/绘图库
    pylab：它把 matplotlib.pyplot（绘图函数）和 numpy（数值计算）的一部分函数打包在一起导入
"""
import matplotlib.pylab as plt

"""
    了舍入误差（rounding error）:
        所谓舍入误差，是指因省略小数的精细部分的数值（比如，小数点第8位以后的数值）而造成最终的计算结果上的误差。
        比如，在Python中，舍入误差可如下表示。
        -->np.float32(1e-50) = 0
     
     中心差分：f(x+h) - f(x) 减少由于h无法真趋近与0导致的误差。
"""

"""
    数值微分(numerical_diff)：利用微小的差分求导数的过程
    
    解析性：基于数学式的推导求导数的过程，通常有“解析性求解”或“解析性求导”
"""

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


"""
    lambda是匿名函数，一行写一个小函数，不用def起名字
    
    tangent_line函数在求f(x)的一阶泰勒展开,就是f在x处的一条切线。
"""
def tangent_line(f, x):
    d  = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')

"""f(x)在x=5处的1一阶泰勒展开"""
tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()










