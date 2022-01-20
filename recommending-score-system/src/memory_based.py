import numpy as np
from scipy import sparse, spatial  # 内置spatial.cosine函数略有问题(average)
import os
import scipy
import numba
from numba import prange
import scipy


#################################
# 评分相似度计算函数
#################################
@numba.jit(nopython=True, fastmath=True)
def user_sim(R: np.ndarray, R_avg: np.ndarray, Idx_eff: np.ndarray, Sim_user: np.ndarray) -> np.ndarray:
    # 对某个物品有共同评分的用户的打分值才能计入相似度
    #############################################
    # 为了编译写成循环
    Score = np.zeros(len(Idx_eff))
    for idx, (u, i) in enumerate(Idx_eff):
        Score[idx] = R[u, i]
    for u, i in Idx_eff:
        Score.append(R[u, i])

    R[:, :] = -np.inf
    for idx, (u, i) in enumerate(Idx_eff):
        R[u, i] = Score[idx]

    R = np.where(R >= 0, R - R_avg.reshape(-1, 1), 0)  # 中心化
    #############################################

    #############################################
    # Score = R[Idx_eff[:, 0].tolist(), Idx_eff[:, 1].tolist()]
    # R[:, :] = -np.inf
    # R[Idx_eff[:, 0].tolist(), Idx_eff[:, 1].tolist()] = Score
    # R = np.where(R >= 0, R - R_avg.reshape(-1, 1), 0)  # 中心化
    #########################################
    # np.savetxt('R-sum-0.txt', R.sum(0), fmt='%.2f')
    # np.savetxt('R-sum-1.txt', R.norm(0), fmt='%.2f')
    # np.savetxt('R-norm-0.txt', np.linalg.norm(R, axis=0), fmt='%.2f')
    # np.savetxt('R-norm-1.txt', np.linalg.norm(R, axis=1), fmt='%.2f')
    #########################################
    # Sim_user = np.zeros((R.shape[0], R.shape[0])) # numba下不可用
    for i in range(R.shape[0]):
        if i % 100 == 0:
            print(i, '/', R.shape[0])
            # print('用户评分相似度计算到：[{}/{}]'.format(i, R.shape[0]))
        for j in range(i, R.shape[0]):
            nume = (R[i] * R[j]).sum()
            deno = np.linalg.norm(R[i]) * np.linalg.norm(R[j])
            if nume != 0:
                Sim_user[i, j] = nume / deno
                Sim_user[j, i] = Sim_user[i, j]

    # Sim_user = [[(R[i] * R[j]).sum() / np.linalg.norm(R[i]) * np.linalg.norm(R[j]) if (R[i] * R[j]).sum() != 0 else 0 for i in range(R.shape[0])] for j in range(R.shape[0])]

    # print('-------debug--------')
    print('用户相似度计算完成')
    return np.asarray(Sim_user)


@numba.jit(nopython=True, fastmath=True)
def item_sim(R: np.ndarray, R_avg: np.ndarray, Idx_eff: np.ndarray, Sim_item: np.ndarray) -> np.ndarray:
    # Score = R[Idx_eff[:, 0].tolist(), Idx_eff[:, 1].tolist()]
    Score = np.zeros(len(Idx_eff))
    for idx, (u, i) in enumerate(Idx_eff):
        Score[idx] = R[u, i]
    R[:, :] = -np.inf
    for idx, (u, i) in enumerate(Idx_eff):
        R[u, i] = Score[idx]
    # R[Idx_eff[:, 0].tolist(), Idx_eff[:, 1].tolist()] = Score
    R = np.where(R >= 0, R - R_avg.reshape(1, -1), 0)  # 中心化
    # Sim_item = np.zeros([R.shape[1], R.shape[1]])
    for i in range(R.shape[1]):
        if i % 1000 == 0:
            print(i, '/', R.shape[1])
            # print('商品得分相似度计算到：[{}/{}]'.format(i, R.shape[1]))
        for j in range(i, R.shape[1]):
            nume = (R[:, i] * R[:, j]).sum()
            deno = np.linalg.norm(R[:, i]) * np.linalg.norm(R[:, j])
            if nume != 0:
                Sim_item[i, j] = nume / deno
                Sim_item[j, i] = Sim_item[i, j]

    # Sim_item = [[(R[:, i] * R[:, j]).sum() / (np.linalg.norm(R[:, i]) * np.linalg.norm(R[:, j])) if (R[:, i] * R[:,j]).sum() != 0 else 0 for i in range(R.shape[1])] for j in range(R.shape[1])]

    # print('-------debug--------')
    return np.asarray(Sim_item)


#######################
@numba.jit(nopython=True, parallel=True)
def is_exist(M, i, j):
    for ii, jj in M:
        if ii == i and jj == j:
            return True
    return False


#################################
# 预测函数
#################################
# @numba.jit(nopython=True, fastmath=True)
# def predict_by_user(R: np.ndarray, Sim: np.ndarray, Sim_argsort: np.ndarray, Idx_eff: np.ndarray, k: int,
#                     Idx_pred: np.ndarray, R_avg: np.ndarray) -> np.ndarray:
@numba.jit(nopython=True, fastmath=True)
def predict_by_user(R: np.ndarray, Sim: np.ndarray, Idx_eff: np.ndarray, k: int,
                    Idx_pred: np.ndarray, R_avg: np.ndarray) -> np.ndarray:
    """
    基于用户的评分预测需要注意用户评分均值
    2273028次循环，需要JIT
    """
    pred = []
    for iii, (u, i) in enumerate(Idx_pred):
        # k个用户u的近邻(对物品i有评分的用户）
        if iii % 1000 == 0:
            print('正在预测第[', iii, '/', len(Idx_eff), ']个评分记录')
        idx = 0
        # Sim_near = np.array([])
        # R_near = np.array([])
        Sim_near = []
        R_near = []
        # sim=Sim.argsort()[u]
        # Sim_argsort[u]
        # sorted()
        for u_ in Sim[u].argsort():
            # [u_, i] exists in Idx_eff ?!
            # if (u_, i) in Idx_eff:
            if is_exist(Idx_eff, u_, i):
                idx += 1
                # np.append(R_near, R[u_, i] - R_avg[u_])
                # np.append(Sim_near, Sim[u, u_])
                R_near.append(R[u_, i] - R_avg[u_])
                Sim_near.append(Sim[u, u_])
            if idx >= k:
                break
        Sim_near = np.asarray(Sim_near)
        R_near = np.asarray(R_near)
        pred.append(R_avg[u] + (R_near * Sim_near).sum() / Sim_near.sum())

    # pred = np.clip(np.round(pred), 0, 5)
    print('长度：', len(pred))
    # pred = np.round(pred)
    for i in range(len(pred)):
        pred[i] = round(pred[i])
        if pred[i] < 0:
            pred[i] = 0
        if pred[i] > 5:
            pred[i] = 5
    return np.asfarray(pred)


# @numba.jit(nopython=True,fastmath=True)
# def predict_by_item(R: np.ndarray, Sim: np.ndarray, Sim_argsort: np.ndarray, Idx_eff: np.ndarray, k: int,
#                     Idx_pred: np.ndarray) -> np.ndarray:
@numba.jit(nopython=True, fastmath=True)
def predict_by_item(R: np.ndarray, Sim: np.ndarray, Idx_eff: np.ndarray, k: int, Idx_pred: np.ndarray) -> np.ndarray:
    """
    R: [N,M]
    Sim: [M,M]
    """
    pred = []
    # for u, i in Idx_pred:
    for iii, (u, i) in enumerate(Idx_pred):
        # k个商品i的近邻(被用户u打过分的商品）
        if iii % 1000 == 0:
            print('正在预测第[', iii, '/', len(Idx_eff), ']个评分记录')
        idx = 0
        # Sim_near = np.array([])
        # R_near = np.array([])
        Sim_near = []
        R_near = []
        # sim=Sim.argsort()[u]
        for i_ in Sim[i].argsort():
            # [u, i_] exists in Idx_eff ?!
            # if (u, i_) in Idx_eff:
            if is_exist(Idx_eff, u, i_):
                idx += 1
                # np.append(R_near, R[u, i_])
                # np.append(Sim_near, Sim[i, i_])
                R.near.append(R[i, i_])
                Sim_near.append(Sim[i, i_])
            if idx >= k:
                break
        Sim_near = np.asarray(Sim_near)
        R_near = np.asarray(R_near)
        pred.append((R_near * Sim_near).sum() / Sim_near.sum())

    # pred = np.clip(np.round(pred), 0, 5)
    print('长度：', len(pred))
    for i in range(len(pred)):
        pred[i] = round(pred[i])
        if pred[i] < 0:
            pred[i] = 0
        if pred[i] > 5:
            pred[i] = 5
    return np.asfarray(pred)
