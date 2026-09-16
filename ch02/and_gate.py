import numpy as np

def AND(x1, x2):
    """
    :variate x: 输入x
    :variate w: 权重  (控制各个信号的重要性)
    :variate b： 偏置 (调整神经元被激活的容易程度)
    :variate temp = x1*w1 + x2*w2 + b

    :return: 与的结果
    """

    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = - 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(f'{xs[0]}和{xs[1]}相与为： {y}')