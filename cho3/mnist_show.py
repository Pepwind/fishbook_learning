import numpy as np
from mpmath.libmp import normalize

"""
    前向传播：
        利用训练数据(学习数据)进行权重的学习，进行
        推理时，利用学习到的参数，对输入数据进行分类。
"""

"""
    MINIST图像数据：
        28像素 * 28像素的灰度图像（1通道）,各个像素的
        取值在0到255之间。每个图像数据都相应地标有“7” “2” “1”等标签。
"""

import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
"""
    把父目录deep-learning-from-scratch加入到sys.path（Python的搜索模块的路径集）中，
    从而可以导入deep-learning-from-scratch下的任何目录（包括dataset目录）中的任何文件
"""


import numpy as np
from dataset.mnist import load_mnist
"""
    Python Imaging Library是python图像数据库的简称，Image是图像处理模块
"""
from PIL import Image

def img_show(img):
    """
        Image.fromarray:把保存为NumPy数组的图像数据转换为PIL用的数据对象
    """
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

"""读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0（图像的各个像素值除以255）
    flatten : 是否将图像展开为一维数组(flat是平的意思) 1*28*28=784
    one_hot_label : 是否用one-hot表示
    
    注意：one-hot表示是仅正确解标签为1，其余皆为0的数组，就像[0,0,1,0,0,0,0,0,0,0]这样。
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
"""
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)

"""输出各个数据的形状"""
print(x_train.shape) # (60000, 784)
print(y_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(y_test.shape) # (10000,)


img = x_train[0]
label = y_train[0]
print(label)

print(img.shape) # (784,)
img = img.reshape(28,28) ## 把图像的形状变为原来的尺寸
print(img.shape) # (28, 28)

img_show(img)

















