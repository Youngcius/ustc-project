import numpy as np
import torch


def train(xdata: torch.Tensor, ydata: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
          num_epochs: int, ratio_split=0.3, lr=0.005, lam=0.001) -> tuple:
    '''
    训练：num_epochs轮次，带验证过程
    批量梯度下降：按照user id批次进行(0-2173)
    '''

    # 初始化为子节点，方便计算梯度
    P.requires_grad_(True)
    Q.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)

    M, N, F = P.size(0), Q.size(0), P.size(1)
    length = len(xdata)

    losses_train = []
    rmse_valid = []
    losses_train_batch = []
    rmse_valid_batch = []
    rmse_min = np.inf  # minimum rmse of valid set to decide best parameters
    best_paras = (P.detach(), Q.detach(), b.detach(), c.detach())

    with  open('../output/running.log', 'a') as f:
        # f.write('=' * 10 + str(datetime.datetime.now()) + '=' * 10)
        f.write('epoch\ttrain-loss\tvalid-rmse\n')

    optimizer = torch.optim.Adam([P, Q, b, c], lr=lr)

    for e in range(num_epochs):
        # 划分训练集、验证集（0.7:0.3）
        # 每个epoch都随机划分一次，训练效果更好；相当于批量随机梯度下降
        idx_perm = np.random.permutation(range(length))
        idx_train = idx_perm[int(ratio_split * length):]
        idx_valid = idx_perm[:int(ratio_split * length)]
        xtrain = xdata[idx_train]
        ytrain = ydata[idx_train]
        xvalid = xdata[idx_valid]
        yvalid = ydata[idx_valid]

        losses_train_batch.append([])
        rmse_valid_batch.append([])
        ##################
        # training process
        for m in range(M):
            n_s = xtrain[xtrain[:, 0] == m][:, 1]  # item idx

            yhat = P[m].view(1, -1).mm(Q.t()).flatten() + b[m] + c  # Ru_hat: [1, N]
            yhat = 5 * torch.sigmoid(yhat)
            loss = (yhat[n_s] - ytrain[xtrain[:, 0] == m]).pow(2).mean() + lam * (
                    P.norm(dim=1).sum() + Q.norm(dim=1).sum()) / len(n_s)
            loss.backward()
            losses_train_batch[e].append(loss.item())
            ##############
            # 按照学习率SGD优化
            # with torch.no_grad():
            #     P -= lr * P.grad
            #     Q -= lr * Q.grad
            #     b -= lr * b.grad
            #     c -= lr * c.grad
            #     P.grad.zero_()
            #     Q.grad.zero_()
            #     b.grad.zero_()
            #     c.grad.zero_()
            ##############
            # 使用torch内置Adam优化器代替SGD
            optimizer.step()
            optimizer.zero_grad()

            ####################
            # validation process
            with torch.no_grad():

                n_s = xvalid[xvalid[:, 0] == m][:, 1]  # item idx

                yhat = P[m].view(1, -1).mm(Q.t()).flatten() + b[m] + c  # Ru_hat: [1, N]
                yhat = 5 * torch.sigmoid(yhat)
                # loss = (yhat[n_s] - ytrain[xtrain[:, 0] == m]).pow(2).mean() + lam * (P.norm(dim=1).sum() + Q.norm(dim=1).sum()) / len(n_s)
                # losses_valid_batch[e].append(loss.item())
                rmse = (yhat[n_s].round() - yvalid[xvalid[:, 0] == m]).pow(2).mean().sqrt().item()
                rmse_valid_batch[e].append(rmse)

            if (m + 1) % 400 == 0:
                print('epoch: {}, batch [{}/{}], train loss: {:.4f}, valid rmse: {:.4f}'.format(e, m + 1, M, np.mean(
                    losses_train_batch[e]), np.mean(rmse_valid_batch[e])))

        losses_train.append(np.mean(losses_train_batch[e]))
        rmse_valid.append(np.mean(rmse_valid_batch[e]))
        print('epoch: {}, loss on training set: {:.4f}, RMSE on validation set: {:.4f}'.format(e, losses_train[e],
                                                                                               rmse_valid[e]))

        f.write('{}\t{:.4f}\t{:.4f}\n'.format(e, losses_train[e], rmse_valid[e]))

        if rmse_valid[e] < rmse_min:
            rmse_min = rmse_valid[e]
            best_paras = (P.detach(), Q.detach(), b.detach(), c.detach())

    f.write('\n')
    f.close()

    return best_paras


def predict(xdata: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    '''
    预测，输出张量数据（一维, float32）
    不能直接矩阵计算，内存不足

    '''
    # idx_m_n = [(user_idx.index(u), item_idx.index(i)) for u,i in xdata]
    pred = [P[m].view(1, -1).mm(Q.t()).flatten()[n] + b[m] + c[n] for m, n in xdata]
    pred = torch.Tensor(pred)
    return (5 * torch.sigmoid(pred)).round()
