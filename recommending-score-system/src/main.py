import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from process import read_relation, read_trainset, read_testset
import torch
import random
import datetime
from model_based import train, predict
from memory_based import user_sim, item_sim, predict_by_user, predict_by_item

col_names_train = ['user', 'item', 'rate', 'timestamp', 'tag']
col_names_test = ['user', 'item', 'timestamp', 'tag']

train_dpath = '../data/training.dat'
test_dpath = '../data/testing.dat'
relation_dpath = '../data/relation.txt'

idx_rand = [5, 50, 100, 200, 250, 30]

warnings.filterwarnings('ignore')


def time_score_plotplot(idxs, title=None):
    plt.figure(figsize=(18, 15))
    for i, idx in enumerate(idxs):
        score_dst = list(train_df[train_df['user'] == idx].groupby('timestamp')['rate'])
        score_dst = list(zip(*score_dst))
        # xdata = score_dst[0]
        ydata = score_dst[1]
        plt.subplot(len(idxs), 1, i + 1)
        plt.boxplot(ydata)
        # plt.plot(list(map(np.mean, ydata)), label='avg-score -- user-'+str(idx))
        # plt.legend()
        plt.xticks([])
        if i == 0:
            plt.title('Temporal features of score')
        if i == len(idxs) - 1:
            plt.xlabel('Timestamp')
            if title == None:
                plt.show()
            else:
                plt.savefig(title + '.png', dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Douban movie score prediction.')
    parser.add_argument('-s', '--seed', type=int, default=115,
                        help='random seed for initializing matrix elements')
    parser.add_argument('-n', '--normal', action='store_true',
                        help='if randomly initialize elements based on normal distribution')
    parser.add_argument('-old', action='store_true',
                        help='if use the old parameters matrix that have existed')
    parser.add_argument('-plot', action='store_true',
                        help='plot temporal phenomenon of scores')
    parser.add_argument('-svd', action='store_true',
                        help='SVD matrix factorization (if "none", use memory-based prediction)')
    parser.add_argument('-k', type=int, default=10, help='how many neighbors for prediction collaborative filtering')
    parser.add_argument('-d', '--dpath', type=str, default='../output/',
                        help='folder path of output file')
    parser.add_argument('-e', '--epoch', type=int, default=3,
                        help='number of epochs for optimization')
    parser.add_argument('-r', '--ratio', type=float, default=0.3,
                        help='ratio of the origin train set splitted for valid set')
    parser.add_argument('-lr', type=float, default=0.005,
                        help='learning rate of SGD optimization')
    parser.add_argument('-lam', type=float, default=0.0001,
                        help='lambda: the regularization parameter')
    parser.add_argument('-f', type=int, default=10,
                        help='dim of hidden factors')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if 'train_df.csv' in os.listdir('../output'):
        train_df = pd.read_csv(os.path.join(args.dpath, 'train_df.csv'),
                               names=col_names_train + ['user-idx', 'item-idx'])
        test_df = pd.read_csv(os.path.join(args.dpath, 'test_df.csv'), names=col_names_test + ['user-idx', 'item-idx'])
        relation = read_relation(relation_dpath)
        train_df['rate'] = train_df['rate'].astype(np.float32)
        train_df['timestamp'] = train_df['timestamp'].astype(np.datetime64)
        test_df['timestamp'] = test_df['timestamp'].astype(np.datetime64)

        Xtrain = train_df.loc[:, ['user-idx', 'item-idx']].to_numpy().copy()
        Ytrain = train_df['rate'].to_numpy().copy()
        Xtest = test_df.loc[:, ['user-idx', 'item-idx']].to_numpy()

        user_idx = sorted(train_df['user'].unique())  # 数值下标对应的字符下标
        item_idx = sorted(train_df['item'].unique())  # list type
    else:
        train_df = read_trainset(train_dpath, col_names_train)
        test_df = read_testset(test_dpath, col_names_test)

        train_df['user'] = train_df['user'].astype(np.int64)
        train_df['item'] = train_df['item'].astype(np.int64)
        train_df['timestamp'] = train_df['timestamp'].astype(np.datetime64)
        train_df['rate'] = train_df['rate'].astype(np.float32)
        test_df['user'] = test_df['user'].astype(np.int64)
        test_df['item'] = test_df['item'].astype(np.int64)
        test_df['timestamp'] = test_df['timestamp'].astype(np.datetime64)
        train_df = train_df.sort_values('user')
        test_df = test_df.sort_values('user')
        user_idx = sorted(train_df['user'].unique())
        item_idx = sorted(train_df['item'].unique())  # list

        # 转化为ndarray再计算下标，速度更快
        Xtrain = train_df.loc[:, ['user', 'item']].to_numpy().copy()
        Xtest = test_df.loc[:, ['user', 'item']].to_numpy().copy()
        Ytrain = train_df['rate'].to_numpy().copy()

        Xtrain[:, 0] = [user_idx.index(u) for u in Xtrain[:, 0]]
        Xtrain[:, 1] = [item_idx.index(i) for i in Xtrain[:, 1]]
        Xtest[:, 0] = [user_idx.index(u) for u in Xtest[:, 0]]
        Xtest[:, 1] = [item_idx.index(i) for i in Xtest[:, 1]]  # 至此Xtrain等数组中全为数值下标

        train_df['user-idx'] = Xtrain[:, 0]
        train_df['item-idx'] = Xtrain[:, 1]
        test_df['user-idx'] = Xtest[:, 0]
        test_df['item-idx'] = Xtest[:, 1]
        train_df.to_csv(os.path.join(args.dpath, 'train_df.csv'), index=None, header=None)
        test_df.to_csv(os.path.join(args.dpath, 'test_df.csv'), index=None, header=None)

    M = len(user_idx)
    N = len(item_idx)
    print('训练数据\t用户个数：{}，商品个数：{}'.format(M, N))
    print('测试数据\t用户个数：{}，商品个数：{}'.format(len(test_df['user'].unique()), len(test_df['item'].unique())))

    if set(test_df['item'].unique()) - set(item_idx) != set():
        raise Exception('测试集合商品编号不为训练集中商品编号集合的子集!')
    if set(test_df['user'].unique()) - set(user_idx) != set():
        raise Exception('测试集合用户编号不为训练集中商品编号集合的子集!')

    if args.plot:
        time_score_plotplot(idx_rand, os.path.join(args.dpath, 'temporal'))

    # 先不用日期信息
    # 参数初始化

    if args.svd:
        print('Currently using model-based prediction (Matrix SVD)')
        # 使用基于模型 SVD 方法
        # 张量数据
        Xtrain = torch.from_numpy(Xtrain)
        Ytrain = torch.from_numpy(Ytrain)
        Xtest = torch.from_numpy(Xtest)
        if ('best_paras-P.txt' in os.listdir('../output')) and args.old:
            P = torch.from_numpy(np.loadtxt(os.path.join(args.dpath, 'best_paras-P.txt'))).float()
            Q = torch.from_numpy(np.loadtxt(os.path.join(args.dpath, 'best_paras-Q.txt'))).float()
            b = torch.from_numpy(np.loadtxt(os.path.join(args.dpath, 'best_paras-b.txt'))).float()
            c = torch.from_numpy(np.loadtxt(os.path.join(args.dpath, 'best_paras-c.txt'))).float()
        elif args.normal:
            P = torch.randn(M, args.f) / np.sqrt(args.f)  # user factor matrix, [m, f]
            Q = torch.randn(N, args.f) / np.sqrt(args.f)  # item factor matrix, [n, f]
            b = torch.tensor(train_df.groupby('user').agg('mean')['rate'].tolist()) / 2
            c = torch.tensor(train_df.groupby('item').agg('mean')['rate'].tolist()) / 2
        else:
            P = torch.rand(M, args.f) / np.sqrt(args.f)  # user factor matrix, [m, f]
            Q = torch.rand(N, args.f) / np.sqrt(args.f)  # item factor matrix, [n, f]
            b = torch.tensor(train_df.groupby('user').agg('mean')['rate'].tolist()) / 2
            c = torch.tensor(train_df.groupby('item').agg('mean')['rate'].tolist()) / 2

        f = open('../output/running.log', 'a')
        f.write('=' * 10 + str(datetime.datetime.now()) + '=' * 10)
        f.write('\nlr: {}, lam: {}, normal: {}, F: {}\n'.format(args.lr, args.lam, args.normal, args.f))
        f.close()

        # 接着训练，Adam
        P, Q, b, c = train(Xtrain, Ytrain, P, Q, b, c, args.epoch, args.ratio, lr=args.lr, lam=args.lam)
        np.savetxt(os.path.join(args.dpath, 'best_paras-P.txt'), P.numpy())
        np.savetxt(os.path.join(args.dpath, 'best_paras-Q.txt'), Q.numpy())
        np.savetxt(os.path.join(args.dpath, 'best_paras-b.txt'), b.numpy())
        np.savetxt(os.path.join(args.dpath, 'best_paras-c.txt'), c.numpy())

        with torch.no_grad():
            ypred = predict(Xtest, P, Q, b, c)  # float32
        np.savetxt(os.path.join(args.dpath, 'test-label.txt'), ypred.long().numpy(), fmt='%d')

    else:
        print('Currently using memory-based prediction (item-based & user-based)')
        # 使用基于内存的矩阵补全方法, shape: [2173, 58431]
        R = np.zeros([M, N])
        R[Xtrain[:, 0].tolist(), Xtrain[:, 1].tolist()] = Ytrain  # row: user, col: item
        print('矩阵稀疏度:', len(Xtrain) / M / N)
        # R = sparse.coo_matrix((Ytrain, (Xtrain[:, 0], Xtrain[:, 1]))).toarray()
        # R_csc = sparse.csc_matrix(R)  # 方便列索引
        # R_csr = sparse.csr_matrix(R)  # 方便行索引
        R_user_count = train_df.groupby('user-idx').agg('count')['item-idx'].to_numpy().copy()  # 每个user多少数据项
        R_item_count = train_df.groupby('item-idx').agg('count')['user-idx'].to_numpy().copy()  # count by item
        R_user_avg = R.sum(1) / R_user_count  # [M,]
        R_item_avg = R.sum(0) / R_item_count  # [N,]
        if 'similarities.npz' in os.listdir('../output'):
            print('开始读取相似度矩阵...')
            Similarities = np.load(os.path.join(args.dpath, 'similarities.npz'))
            Sim_user = Similarities['arr_0']
            Sim_item = Similarities['arr_1']
        else:
            print('开始计算相似度...')
            start = time.time()
            Sim_user = np.zeros([R.shape[0], R.shape[0]])
            Sim_user = user_sim(R, R_user_avg, Xtrain, Sim_user)
            Sim_item = np.zeros([R.shape[1], R.shape[1]])
            Sim_item = item_sim(R, R_item_avg, Xtrain, Sim_item)
            print('Running time: {:.2f}'.format(time.time() - start))
            np.savez_compressed(os.path.join(args.dpath, 'similarities.npz'), Sim_user, Sim_item)  # 压缩存储
            # np.savetxt(os.path.join(args.path, 'similarity-user.txt'), Sim_user) # 矩阵太大不能用txt文件存储
            # np.savetxt(os.path.join(args.path, 'similarity-item.txt'), Sim_item)

        # print('shape of user-similarity:', Sim_user.shape, Sim_user.max(), Sim_user.min())
        # print('shape of item-similarity:', Sim_item.shape, Sim_item.max(), Sim_item.min())

        print("执行预测函数...")
        start = time.time()
        ypred_user = predict_by_user(R, Sim_user, Xtrain, args.k, Xtest, R_user_avg)  # 函数中已经取整，返回float型
        ypred_item = predict_by_item(R, Sim_item, Xtrain, args.k, R_item_avg)

        print('Running time: {:.2f}'.format(time.time() - start))

        np.savetxt(os.path.join(args.dpath, 'test-label-user_based.txt'), ypred_user, fmt='%d')
        np.savetxt(os.path.join(args.dpath, 'test-label-item_based.txt'), ypred_item, fmt='%d')

    print('预测结果已经输出至' + args.dpath)
