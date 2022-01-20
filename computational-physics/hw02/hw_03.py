import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

###############
# 第5题

lam = 10
N = 10000


def func(x):
    if x <= 0:
        return 0.0
    else:
        return lam * np.exp(-lam * x)


func = np.frompyfunc(func, 1, 1)

epsilon = np.random.uniform(size=N)
eat = -np.log(epsilon) / lam
sn.distplot(eat, rug=True, hist=True, label="analogous distribution")
max_value = np.max(eat)
plt.xlim(0.0, max_value)

x_real = np.linspace(0.0, max_value, N)
y_real = func(x_real)
plt.plot(x_real, y_real, label="real distribution")

plt.legend()
plt.title("Comparision of real distribution and analogous distribution,\n lamada = 10, N = 1e+4")




# plt.savefig("course_5.pdf")
plt.show()

##############
# 第11题
N = 100000
miu_set = 3
sigma_set = 10


def func_gauss(x, miu, sigma):
    return np.exp(-np.power(x - miu, 2) / 2 / np.power(sigma, 2)) / sigma / np.sqrt(np.pi * 2)


def func_std_gauss(x):
    return np.exp(-np.power(x, 2) / 2) / np.sqrt(2 * np.pi)


func_std_gauss = np.frompyfunc(func_std_gauss, 1, 1)
func_gauss = np.frompyfunc(func_gauss, 3, 1)


# func_std_gauss = L * h(x) * g(x)
def h_func(x):
    return np.exp(-x)


def g_func(x):
    return np.exp(-np.power(x - 1, 2))


# 抽样(先求标准正态分布抽样）
epsilon_1 = np.random.uniform(size=N)
epsilon_2 = np.random.uniform(size=N)
eat_1 = -np.log(epsilon_1)
eat_1 = eat_1[np.where(np.power(eat_1 - 1, 2) <= -2 * np.log(epsilon_2))]
# 正半轴映射到负半轴，取对称
eat_1 = np.append(eat_1, -eat_1)

N = len(eat_1)

# func_std_gauss -> func_gauss
eat_1 = eat_1 * sigma_set + miu_set

min_value = np.min(eat_1)
max_value = np.max(eat_1)

sn.distplot(eat_1, rug=True, label="analogous distribution")

x_real = np.linspace(min_value, max_value, N)
y_real = func_gauss(x_real, miu_set, sigma_set)
plt.plot(x_real, y_real, label="real distribution")

plt.legend()
plt.title("Comparision of real distribution and analogous distribution,\n,miu = 3, sigma = 10")
# plt.savefig('hw_02_11.pdf')
plt.show()
