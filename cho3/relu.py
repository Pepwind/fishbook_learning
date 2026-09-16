import numpy as np
from matplotlib import pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()
plt.ylim(-1.0, 5.5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()