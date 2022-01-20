import numpy as np

s = 10.0
l = 10.0

N = 100000
# 投针法计算PI值

# 对方位角余弦的抽样
epsilon_1 = np.random.uniform(0, 1, N)
epsilon_2 = np.random.uniform(0, 1, N)
x = epsilon_1
y = epsilon_2 * 2 - 1
indx = np.where(np.power(x, 2)+np.power(y, 2) < 1)
x = x[indx]
y = y[indx]
N = len(y)
cos_value = (np.power(x, 2) - np.power(y, 2)) / (np.power(x, 2)+np.power(y, 2))

# 对“针”的抽样
center = np.random.uniform(-s/2, s/2, N)
span = np.maximum(np.abs(center + s/2 * cos_value), np.abs(center - s/2 * cos_value))


M = np.count_nonzero(span > s/2)
my_pi = N * 2 / M
delta = abs(my_pi - np.pi) / np.pi

print("M =", M, "\tN =", N)
print("the estimation of PI is", my_pi)
print("与精确值相对误差：", delta)

