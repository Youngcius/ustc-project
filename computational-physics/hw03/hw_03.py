import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import numba
from numba import jit


@jit(nopython=True)
def velo_trans(rates, angles):
    # rates[0], rates[1], angles[0], angles[1]

    theta = np.random.uniform(0, np.pi)
    rate_cent = np.sqrt((rates[0] ** 2 + rates[1] ** 2) / 4
                        + 0.5 * rates[0] * rates[1] * np.cos(angles[0] - angles[1]))
    rate_in_cent = np.sqrt((rates[0] ** 2 + rates[1] ** 2) / 4
                           - 0.5 * rates[0] * rates[1] * np.cos(angles[0] - angles[1]))
    ang_cent = (angles[0] + angles[1]) / 2

    rates_term = np.linspace(1, 2, 2)
    angles_term = np.linspace(1, 2, 2)
    angles_term[0] = float(ang_cent + theta)  # % (np.pi * 2)
    angles_term[1] = float(ang_cent + theta + np.pi)  # % (np.pi * 2)
    # print(angles_term, "*" * 20)

    rates_term[0] = np.sqrt(((rates[0] ** 2 + rates[1] ** 2) / 2
                             + 2.0 * rate_in_cent * rate_cent * np.cos(theta)))
    rates_term[1] = np.sqrt(((rates[0] ** 2 + rates[1] ** 2) / 2
                             + 2.0 * rate_in_cent * rate_cent * np.cos(theta + np.pi)))

    return (rates_term, angles_term)


@jit(nopython=True)
def collide(velo_set, count):
    # rate_set = np.linspace(1,N,N)*0.1
    # angle_set = np.random.uniform(0.0,np.pi*2,size=N)
    indice = np.arange(velo_set.shape[1])
    velo_set = np.vstack((np.linspace(1, N, N) * 0.1,
                          np.random.uniform(0.0, np.pi * 2, size=N)))
    for i in range(count):
        # print(i)
        indx = np.random.choice(indice, 2)
        # velo_set[:, indx[0]], velo_set[:, indx[1]]
        rate_exp_ini = np.array([velo_set[:, indx[0]][0], velo_set[:, indx[1]][0]])
        ang_exp_ini = np.array([velo_set[:, indx[0]][1], velo_set[:, indx[1]][1]])

        rate_exp_term, ang_exp_term = velo_trans(rate_exp_ini, ang_exp_ini)
        velo_set[:, indx[0]][0] = rate_exp_term[0]  # 粒子1速率
        velo_set[:, indx[0]][1] = ang_exp_term[0]  # 粒子1角度
        velo_set[:, indx[1]][0] = rate_exp_term[1]  # 粒子2速率
        velo_set[:, indx[1]][1] = ang_exp_term[1]  # 粒子2角度
    return velo_set


# # plot for rate
N = int(1e3)
count = int(1e6)

print("总数：", N)
print("碰撞次数：", count)
velo_set = np.vstack((np.linspace(1, N, N) * 0.1,
                      np.random.uniform(0.0, np.pi * 2, size=N)))
velo_set = collide(velo_set, count)


def maxwell_densi_2d(velo, factor):
    # 二维麦克斯韦速率分布密度函数
    # factor 为 m/kt
    return factor * velo * np.exp(-velo ** 2 * factor / 2)


maxwell_densi_2d = np.frompyfunc(maxwell_densi_2d, 2, 1)

theo_rate = np.linspace(0, np.max(velo_set[0, :]), int(1e5))
fac = 2.0 / np.mean(velo_set[0,:] ** 2)
theo_densi = maxwell_densi_2d(theo_rate, fac)

sn.distplot(velo_set[0, :], rug=True,
            label=str(count) + " times collision -- " + str(N) + " particles")

plt.plot(theo_rate, theo_densi, label="theoretic 2-D Maxwell distribution curve")
plt.legend()

plt.show()

