import numpy as np
import matplotlib.pyplot as plt


# 左边10000个分子，随机扩散，给出其随次数的数目分布
# 0 代表左，1 代表右边


def diffuse(lhs, rhs, cnt, ret_left=True):
    lhs_count = np.zeros(cnt)
    lhs_count[0] = lhs
    sum = lhs + rhs

    for i in range(cnt):
        chosen = np.random.choice(2, p=[lhs / sum, rhs / sum])
        is_change = np.random.choice(2)
        if chosen == 0:  # 第 i 次是左边分子扩散
            if is_change == 1:
                lhs -= 1
                rhs += 1
        else:  # 第 i 次是右边分子扩散
            if is_change == 0:
                lhs += 1
                rhs -= 1

        lhs_count[i] = lhs  # 第i次扩散后分布

    if ret_left:
        return lhs_count
    else:
        return lhs + rhs - lhs_count


lhs_ini = int(1e4)
rhs_ini = int(0)
sum_ini = lhs_ini + rhs_ini
count = int(1e5)

lhs_count = diffuse(lhs=lhs_ini, rhs=rhs_ini, cnt=count, ret_left=True)

fac = np.log(sum_ini / (sum_ini - 1))
lhs_count_theo = np.exp(- fac * np.arange(count)) * sum_ini / 2 + sum_ini / 2

plt.figure(dpi=400)
plt.plot(lhs_count, label="left count", color='black', linewidth=0.5)
plt.plot(lhs_count_theo, label="theoretic curve", color='red', linewidth=0.5)

plt.legend()
plt.title("distribution of random diffusion")
plt.show()
# plt.savefig("hw_04.pdf")
