import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator
from skimage import io

"""
构造支持向量机分类器
labels: {1, -1}
OVO 多分类SVM分类器
---
该优化算法计算量消耗较大，运行时需要耐心等几分钟~
"""


class SVF(BaseEstimator):

    def __init__(self, C=1):
        # 可选设置精度或是迭代次数，鉴于样本量较大，SMO设置迭代次数较能减小开销
        self.attr_num = 1
        self.class_num = 2
        self.sample_num = 1
        self.labels_unique = np.array([1, -1])
        self.C = C
        self.eps = 1e-4
        self.iteration = 2000
        self.coef = np.array([])
        self.intercept = np.array([])

    def fit(self, X, y):
        y = y.ravel()
        self.class_num = len(np.unique(y))
        self.attr_num = int(X.size / len(X))
        self.sample_num = len(X)
        self.labels_unique = np.unique(y)
        if self.class_num == 2:
            self.coef, self.intercept = self.binary_fit(X, y)
        else:
            self.multi_fit(X, y)

    def binary_fit(self, X, y):
        # 拟合函数返回系数值coef和intercept
        # SMO方法计算参数
        alpha = np.ones_like(y)

        b = 0
        y_pred = np.array([(np.sum(alpha * y * np.dot(X, X[i])) + b) for i in range(len(y))])
        error = y_pred - y  # 预测值和实际标签的差值
        value = y * y_pred  # value 和 1 作比较
        kkt = self.kkt_condition(X, y, alpha, value)
        for it in range(self.iteration):
            # 默认以迭代次数为基准而不是精度检验为基准进行迭代
            # KKT algorithm
            if np.sum(kkt) == 0:
                break

            # 选取第一个变量i
            if np.count_nonzero(kkt == 2) == 0:
                if np.count_nonzero(kkt == 1) == 0:
                    break
                else:
                    i = int(np.argwhere(kkt == 1)[0])
            else:
                i = int(np.argwhere(kkt == 2)[0])

            # 选取第二个变量j
            j = self.select_rand(i, len(y))

            alpha_i, alpha_j = alpha[i], alpha[j]

            # calculate: eta = K_11 + K_22 - 2*K_12
            eta = np.dot(X[i], X[i]) + np.dot(X[j], X[j]) - 2 * np.dot(X[i], X[j])
            if y[i] == y[j]:
                L = max(0, alpha_j + alpha_i - self.C)
                H = min(self.C, alpha_i + alpha_j)
            else:
                L = max(0, alpha_j - alpha_i)
                H = min(self.C, self.C + alpha_j - alpha_i)
            # 修正 alpha_i, alph_j
            alpha_j_new = alpha_j + y[j] * (error[i] - error[j]) / eta
            alpha_j_new = np.clip(alpha_j_new, L, H)
            alpha_i_new = alpha_i + y[i] * y[j] * (alpha_j - alpha_j_new)
            alpha[i], alpha[j] = alpha_i_new, alpha_j_new

            # 修正 b,kkt, error, value
            b_i = y[i] - np.sum(alpha * y * np.dot(X, X[i]))
            b_j = y[j] - np.sum(alpha * y * np.dot(X, X[j]))

            if ((alpha[i] < self.C - self.eps) & (alpha[i] > self.eps)) & (
                    (alpha[j] < self.C - self.eps) & (alpha[j] > self.eps)):
                b = b_i
            else:
                b = (b_i + b_j) / 2

            y_pred = np.array([(np.sum(alpha * y * np.dot(X, X[i])) + b) for i in range(len(y))])
            error = y_pred - y  # 预测值和实际标签的差值
            value = y * y_pred  # value 和 1 作比较
            kkt = self.kkt_condition(X, y, alpha, value)

        w = np.sum(alpha.reshape(-1, 1) * X * y.reshape(-1, 1), axis=0)  # axis =0 表示求每列的和
        indx = int(np.argwhere((alpha > 0) & (alpha < self.C))[0])
        b = y[indx] - np.sum(alpha * y * np.dot(X, X[indx]))
        return w, b

    def multi_fit(self, X, y):
        self.coef = np.ones([self.attr_num, int(self.class_num * (self.class_num - 1) / 2)])
        self.intercept = np.ones(int(self.class_num * (self.class_num - 1) / 2))
        # coef矩阵行数为数据属性数，列数为标签类别数的Cn2组合数
        col_indx = 0

        # 总共 calss_num * (class_num - 1) / 2 次循环
        for i in range(self.class_num):
            for j in range(i + 1, self.class_num):
                binary_target = np.zeros_like(y)
                binary_target[y == self.labels_unique[i]] = 1
                binary_target[y == self.labels_unique[j]] = -1
                binary_X = X[np.abs(binary_target) == 1, :]  # X 行数有缩减
                binary_target = binary_target[np.abs(binary_target) == 1]  # binary_target 总长缩减只与binary_X行数
                self.coef[:, col_indx], self.intercept[col_indx] = self.binary_fit(binary_X, binary_target)
                col_indx = col_indx + 1

    def predict(self, X):
        # return np.sign(np.dot(x, self.coef) + self.intercept)
        if self.class_num == 2:
            return np.sign(self.model_binary(X))
        else:
            Y = np.sign(self.model_multi(X))  # 返回 一个元素为 1  或 -1 的矩阵
            y = np.array([np.argmax(np.bincount(self.map_class(Y_i).astype(int))) for Y_i in Y])
            return y

    def model_binary(self, X):
        """
        :param X: sample matrix, i.e., 2-D array
        :return: 1-D array
        """
        return np.dot(X, self.coef) + self.intercept

    def model_multi(self, X):
        """
        :param X: sample matrix, i.e., 2-D array
        :return: 2-D array
        """
        return np.dot(X, self.coef) + self.intercept

    def select_rand(self, i, n):
        # 随机选取 range(n) 内不等于i的数作为j并返回
        j = np.random.randint(n)
        while j == i:
            j = np.random.randint(n)
        return j

    def kkt_condition(self, X, y, alpha, value):
        kkt = np.zeros(len(X))
        # alpha_i == 0    <= =>  yi * (W * Xi + b) >= 1      otherwise kkt[i] = 1
        # 0 < alpha_i < C <= =>  yi * (W * Xi + b) == 1      otherwise kkt[i] = 2
        # alpha_i = C     <= =>  yi * (W * Xi + b) <= 1      otherwise kkt[i] = 1
        kkt[(alpha <= self.eps) & ~(value >= 1)] = 1
        kkt[(alpha >= self.C - self.eps) & ~(value <= 1)] = 1
        kkt[((alpha < self.C - self.eps) & (alpha > self.eps)) & ~(abs(value - 1) < self.eps)] = 2
        return kkt

    def map_class(self, vec):
        labels = vec.copy()
        indx = 0
        for i in range(self.class_num):
            for j in range(i + 1, self.class_num):
                labels[indx] = self.labels_unique[i] if labels[indx] == 1 else self.labels_unique[j]
                indx = indx + 1
        return labels


if __name__ == '__main__':
    face = pd.read_excel("face.xlsx")
    columns = face.columns
    # Index(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11',
    #        'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21',
    #        'f22', 'f23', 'f24', 'Expression', 'Age '],
    #       dtype='object')
    face = np.asarray(face)

    # 划分训练集和测试集并规范化数据
    test_rate = 0.3
    random_state = 3
    x = face[:, columns != 'Expression']
    y = face[:, columns == 'Expression']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    data_train, data_test, target_train, target_test = train_test_split(x, y, test_size=test_rate,
                                                                        random_state=random_state)
    del x, y

    # 分类器拟合得到模型参数
    svf = SVF(C=100)
    svf.fit(data_train, target_train)
    # print("parameters of learned model:\n")
    # print("coefficient:",svf.coef)
    # print("intercept:",svf.intercept)

    # 5折交叉验证
    pred_train = cross_val_predict(svf, data_train, target_train.ravel(), cv=5)
    score = cross_val_score(svf, data_train, target_train.ravel(), cv=5, scoring='accuracy')
    # pred_train = svf.predict(data_train)

    cmat = confusion_matrix(target_train, pred_train)
    print(cmat)
    io.imshow(cmat, cmap=plt.cm.gray)
    plt.title("Confusion Matrix")
    plt.show()

    # 以下可选以sklearn内置LinearSVC作为比较
    # from sklearn.svm import LinearSVC
    # svc = LinearSVC(C=10)
    # svc.fit(data_train,target_train.ravel())
    # pred_train = cross_val_predict(svc, data_train, target_train.ravel(), cv=5)
    # score = cross_val_score(svc,data_train,target_train.ravel(),cv=5)
    # cmat = confusion_matrix(target_train,pred_train)
    # print(cmat)
    # print(score)

    # 测试集结果展示
    pred_test = svf.predict(data_test)
    accu = np.count_nonzero(target_test.ravel() == pred_test.ravel()) / len(target_test)
    print("accuracy of test data:", accu)
    test_cmat = confusion_matrix(target_test.ravel(), pred_test.ravel())
    print('confusion matrix of test data:')
    print(test_cmat)
    io.imshow(test_cmat, cmap=plt.cm.gray)
    plt.title("Confusion Matrix of Test Data")
    plt.show()
