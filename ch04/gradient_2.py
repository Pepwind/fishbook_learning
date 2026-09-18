import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


"""
    偏导数公式为：
        lim[f(x0,……,xi + h,……,xn) - f(x0,……,xi - h,……,xn)] / 2h
    求函数的偏导数
    
    注意：x 必须是 float 数组。如果 x = np.array([3, 4])（int 型），x[idx] = tmp_val - h 会被截断成整数，h 直接丢失，梯度全变 0 或出错。所以调用前先 x = x.astype(float)，或者初始化就写 np.array([3.0, 4.0])。
"""
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad  = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        #求f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        #求f(x-h)
        x[idx] = float(tmp_val) - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val

    """
        方法2：
        x1 = x.copy()
        x1[idx] += h
        
        x2 = x.copy()
        x2[idx] -= h
        
        grad[idx] = (f(x1) - f(x2)) / (2*h)
    """

    return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

"""
    注意：
    np.zeros_like(X) 会继承 X 的类型。如果 X 是整型（np.array([[3,4]])），
    grad 也是整型，grad[idx] = [6.2, 8.4] 会被截断成 [6, 8]。所以 X 一定要是 float，
"""
def numerical_gradient(f, X):
    #若为1维： e.g.[1,2]
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    # 若为2维：[[1,2],[1,2]]
    else:
        grad = np.zeros_like(X)
        """
            enumerate函数同时拿到"下标"和"元素"
            等价写法：
            for idx in range(len(X)):
                x = X[idx]
        """
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

"""行是样本，列是维度。"""
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25) #(18,)
    x1 = np.arange(-2, 2.5, 0.25) #(18,)
    """
        np.meshgrid(*xi, copy=True, sparse=False, indexing='xy')
            *xi: 任意个一维数组（坐标刻度），个数 = 输出维度，至少 1 个
            copy: True（默认，复制数据）/False（返回视图，共享内存，危险）
            sparse: False（默认，返回完整稠密网格）/True（返回稀疏网格，靠广播省内存）
            indexing: 'xy'（默认，绘图约定）或 'ij'（矩阵约定），只影响哪个轴对应哪个输入
        返回	,与输入个数相同的数组列表，形状完全一致:
            设输入n个一维数组，长度分别是 N_1, N_2, ..., N_n
                1.返回n个n维数组 
                2.每个的形状都相同，总元素数 = N_1 * N_2 *……* N_n$（笛卡尔积的全部组合）
                3.第 $k$ 个返回数组，在第 k 个轴方向上"变化"，在其余轴方向上"复制"
    """
    X, Y = np.meshgrid(x0, x1) #(18, 18)  网格坐标矩阵，324 个点

    X = X.flatten() #(324,) 拉平成一维
    Y = Y.flatten() #(324,) 拉平成一维

    # np.array([X,Y]).T: (324, 2) 324 行，每行一个点的坐标
    # numerical_gradient  (324, 2) 	每行一个点的梯度
    grad = numerical_gradient(function_2, np.array([X, Y]).T)

    plt.figure()
    """
        画箭头
        quiver(X, Y, U, V) 的参数含义：在坐标 (X[i], Y[i]) 处画一个箭头，箭头的水平分量 U、垂直分量 V。
            1. X, Y：箭头起点的位置（324 个点的坐标）
            2. -grad[:,0], -grad[:,1]：箭头的方向向量（取了负号！）
                梯度指向函数值上升最快的方向；function_2 的最小值是原点，所以：+grad：箭头从原点向外发散（背离最小值）;-grad：箭头从四周指向原点（趋向最小值）.
            3. angles="xy"：箭头的方向按数据坐标解释。默认是 "uv"，会按屏幕像素解释——那样一旦 x/y 轴比例不同、或后续 set_aspect 不等比，箭头方向就会画歪。加上 angles="xy" 才能保证"箭头方向 = 真实的梯度方向"。
    """
    plt.quiver(X, Y, -grad[:, 0], -grad[:, 1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2]);
    plt.ylim([-2, 2])  # 显示范围
    plt.xlabel('x0');
    plt.ylabel('x1')  # 轴标签
    plt.grid()  # 网格线
    plt.legend()  # ⚠️ 无标签，会警告且什么都不显示
    plt.draw()  # ⚠️ 多余
    plt.show()