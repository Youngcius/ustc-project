import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


class GmmCluster(object):
    def __init__(self, n_cluster=5, mu=None):
        self.K = n_cluster  # K 个Gaussian model
        self.M = 1  # 属性个数
        self.alpha = np.ones(self.K) / self.K  # 初始化混合系数
        self.mu = mu  # (K,M)
        self.sigma = np.array([])  # (K,M,M)
        self.clusters = np.array([])  # (N,) 聚类结果
        self.center = np.array([])  # (M,)
        self.N = 0
        self.gamma = np.array([])  # (N,K) 后验概率
        self.iteration = 1000
        self.within_ss = 0
        self.between_ss = 0
        self.tot_ss = 0
        self.a = 0  # SS
        self.b = 0  # SD
        self.c = 0  # DS
        self.d = 0  # DD
        self.RI = 0  # Rand Index

    def cluster(self, X):
        self.N = len(X)
        self.clusters = np.zeros(self.N)
        self.M = X.shape[1]
        self.sigma = np.array([np.cov(X.transpose()) for i in range(self.K)])  # (K,M,M)
        if self.mu is None:
            self.mu = []
            for i in range(self.K):  # (K,M)
                self.mu.append(X[i])
            self.mu = np.vstack(self.mu)
        else:
            self.mu = np.array(self.mu)
        self.gamma = np.zeros([len(X), self.K])  # (N,K) 响应度
        for i in range(self.iteration):  # e.g. 1000次迭代
            # E 步，计算响应度
            for j in range(self.N):  # 对低 j 个样本
                sum = 0
                for k in range(self.K):
                    sum += self.alpha[k] * self.gaussian(X[j], k)
                for k in range(self.K):
                    self.gamma[j, k] = self.alpha[k] * self.gaussian(X[j], k) / sum

            # M 步，重新计算模型参数
            self.mu = np.dot(self.gamma.transpose(), X) / np.sum(self.gamma, axis=0).reshape(-1, 1)  # (K,M)
            self.alpha = np.sum(self.gamma, axis=0) / self.N  # (K,)
            for k in range(self.K):  # 对第 k 个高斯模型
                self.sigma[k] = np.dot((X - self.mu[k]).transpose() * self.gamma[:, k], X - self.mu[k]) / np.sum(
                    self.gamma[:, k])
                # self.alpha[k] =
        self.clusters = np.argmax(self.gamma, axis=1)  # (N,)
        self.center = np.mean(X, axis=0)
        self.evaluate_internal(X)

    def gaussian(self, x, k):
        x = x.ravel()  # x 为一维向量
        return np.exp(-(np.dot(np.dot(x - self.mu[k], np.linalg.pinv(self.sigma[k])), x - self.mu[k])) / 2) / np.power(
            2 * np.pi, self.K / 2) / np.sqrt(np.linalg.det(self.sigma[k]))

    def evaluate_internal(self, X):

        for i in range(self.N):  # 类总距离 sum of square
            self.tot_ss += np.sum(np.square(X[i] - self.center))

        for k in range(self.K):  # 类内总距离 sum of square
            X_part = X[self.clusters == k]
            for i in range(len(X_part)):
                self.within_ss += np.sum(np.square(X_part[i] - self.mu[k]))
        self.between_ss = self.tot_ss - self.within_ss  # 类间总距离 sum of square

    def evaluate_external(self, X, clusters):
        clusters = np.array(clusters).ravel()
        self.a = self.b = self.c = self.d = 0
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if self.clusters[i] == self.clusters[j] and clusters[i] == clusters[j]:
                    self.a += 1
                elif self.clusters[i] == self.clusters[j] and clusters[i] != clusters[j]:
                    self.b += 1
                elif self.clusters[i] != self.clusters[j] and clusters[i] == clusters[j]:
                    self.c += 1
                else:
                    self.d += 1
        self.RI = 2 * (self.a + self.d) / len(X) / (len(X) - 1)


if __name__ == '__main__':
    ##########################
    # Iris数据聚类
    warnings.filterwarnings('ignore')

    iris = load_iris()
    data = iris.data
    target = iris.target
    scaler = StandardScaler()
    data = scaler.fit_transform(data)  # 标准化
    indx = np.random.permutation(len(data))
    data = data[indx]  # 随机排序
    target = target[indx]

    # 内置KMean聚类
    km_cluster = KMeans(n_clusters=3)
    km_cluster.fit(data)
    km_pred = km_cluster.predict(data)

    # GMM聚类
    gmm_cluster = GmmCluster(n_cluster=3, mu=km_cluster.cluster_centers_)
    # gmm_cluster = GmmCluster(n_cluster=3)
    gmm_cluster.cluster(data)

    # KMeans
    plt.figure(figsize=(7, 7))
    plt.subplot(2, 2, 1)
    sns.scatterplot(data[:, 0], data[:, 1], style=target,hue=target)
    plt.title("True")
    plt.subplot(2, 2, 2)
    sns.scatterplot(data[:, 2], data[:, 3], style=target,hue=target)
    plt.title("True")
    plt.subplot(2, 2, 3)
    sns.scatterplot(data[:, 0], data[:, 1], style=km_pred, hue=km_pred)
    plt.title("Cluster")
    plt.subplot(2, 2, 4)
    sns.scatterplot(data[:, 2], data[:, 3], style=km_pred,hue=km_pred)
    plt.title("Cluster")
    plt.suptitle("K-Means Cluster", fontsize=12)
    plt.show()

    # GMM
    plt.figure(figsize=(7, 7))
    plt.subplot(2, 2, 1)
    sns.scatterplot(data[:, 0], data[:, 1], style=target, hue=target, legend=False)
    plt.title("True")
    plt.subplot(2, 2, 2)
    sns.scatterplot(data[:, 2], data[:, 3], style=target, hue=target, legend=False)
    plt.title("True")
    plt.subplot(2, 2, 3)
    sns.scatterplot(data[:, 0], data[:, 1], style=gmm_cluster.clusters, hue=gmm_cluster.clusters, legend=False)
    plt.title("Cluster")
    plt.subplot(2, 2, 4)
    sns.scatterplot(data[:, 2], data[:, 3], style=gmm_cluster.clusters, hue=gmm_cluster.clusters, legend=False)
    plt.title("Cluster")
    plt.suptitle("GMM Cluster", fontsize=12)
    plt.show()

    # RI -- GMM
    gmm_cluster.evaluate_external(data, target)
    print('GMM cluster', gmm_cluster.RI)

    # RI -- KMeans
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            indx += 1
            if km_pred[i] == km_pred[j] and target[i] == target[j]:
                a += 1
            elif km_pred[i] == target[j] and target[i] != target[j]:
                b += 1
            elif km_pred[i] != km_pred[j] and target[i] == target[j]:
                c += 1
            else:
                d += 1
    RI = 2 * (a + d) / len(data) / (len(data) - 1)

    print('KMeans RI:', RI)
