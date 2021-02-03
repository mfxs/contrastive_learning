# 导入库
import random
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim

from cl_packages import *
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# 下游回归器
def downstream_task(X_train, y_train, X_test, y_test, encoder=None, gpu=torch.device('cuda:0'), title=None):
    if encoder is not None:
        encoder.eval()
        h_train = encoder.encode(torch.tensor(X_train, dtype=torch.float32, device=gpu)).cpu().numpy()
        h_test = encoder.encode(torch.tensor(X_test, dtype=torch.float32, device=gpu)).cpu().numpy()
    else:
        h_train = X_train
        h_test = X_test

    regressor = LinearRegression().fit(h_train, y_train)
    y_fit = regressor.predict(h_train)
    y_pred = regressor.predict(h_test)

    print('Fit R2: {}'.format(100 * r2_score(y_train, y_fit, multioutput='raw_values')))
    print('Predict R2: {}\n'.format(100 * r2_score(y_test, y_pred, multioutput='raw_values')))

    plt.plot(y_test, '.-', lw=1, ms=5)
    plt.plot(y_pred, '.-', lw=1, ms=5)
    plt.legend(['Test', 'Prediction'])
    plt.grid()
    if title is not None:
        plt.title(title)
    plt.show()

    return r2_score(y_test, y_pred)


# 参数解析
parser = argparse.ArgumentParser(description='MoCo for soft sensor')

# 实验参数
parser.add_argument('--num_y', type=int, default=12)
parser.add_argument('--no_y', type=int, default=-8)
parser.add_argument('--n_exp', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--unsup_train', type=int, default=2000)

# 训练参数
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--step_size', type=int, default=50)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--topk', type=int, default=3)


# 主函数
def main():
    args = parser.parse_args()

    # 转炉炼钢数据
    data = pd.read_csv('data_preprocess.csv', index_col=0)
    data = data.drop(['记录时间', '熔炼号', '钢种', '实际值-低碳锰铁', 'sb_record_time', 'sb_record_time_x', 'sb_record_time_y'],
                     axis=1).values

    # 参数设定
    sup_train = [10, 20, 50, 100, 200, 500, 1000, 2000]
    sigma = [0.1, 0.2, 0.3, 0.4, 0.5]
    encoder = (512,)
    projector = (128,)
    gpu = torch.device('cuda:0')
    r2 = np.zeros((args.n_exp, len(sigma) + 1, len(sup_train)))

    # 种子设定
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 多次实验
    for exp in range(args.n_exp):
        print('=====Experiment {}/{}====='.format(exp + 1, args.n_exp))

        # 数据划分
        index = list(range(data.shape[0]))
        random.shuffle(index)
        X_unsup = data[index[:args.unsup_train], :-args.num_y]
        y_unsup = data[index[:args.unsup_train], args.no_y]
        X_test = data[index[args.unsup_train:], :-args.num_y]
        y_test = data[index[args.unsup_train:], args.no_y]

        # 数据归一化
        scaler_X = MinMaxScaler().fit(X_unsup)
        X_unsup = scaler_X.transform(X_unsup)
        X_test = scaler_X.transform(X_test)

        # t-SNE降维可视化
        t_sne(X_unsup, y_unsup, title='Without contrastive learning (exp={})'.format(exp + 1))

        # 无对比学习
        print('=====Without Contrastive Learning=====')
        for j in sup_train:
            print('=====Labeled samples are {}====='.format(j))
            X_sup = X_unsup[:j]
            y_sup = y_unsup[:j]
            r2[exp, -1, sup_train.index(j)] = downstream_task(X_sup, y_sup, X_test, y_test,
                                                              title='Without contrastive learning (exp={}, samples={})'.format(
                                                                  exp + 1, j))

        # 有对比学习
        print('=====With Contrastive Learning=====')

        # 不同噪声水平
        for i in sigma:
            print('=====Noise is {}x====='.format(i))

            # 模型生成
            dataset = CLDataset(torch.tensor(X_unsup, dtype=torch.float32, device=gpu))
            model = MoCo(X_unsup.shape[1], encoder, projector, gpu=gpu).to(gpu)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
            criterion = nn.CrossEntropyLoss()

            # 模型训练
            model.train()
            loss_hist = []
            correct = []
            for epoch in range(args.n_epoch):
                loss_hist.append(0)
                correct.append(0)
                data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

                # Mini-Batch
                for batch_X in data_loader:
                    optimizer.zero_grad()
                    q = batch_X
                    k = q + i * torch.randn(q.shape, dtype=torch.float32, device=gpu)

                    # 前向计算
                    l, label, keys = model(q, k)
                    loss = criterion(l, label)
                    loss_hist[-1] += loss.item()
                    _, pos = torch.topk(l, args.topk, 1)
                    correct[-1] += (pos == 0).sum().item()

                    # 反向传播
                    loss.backward()
                    optimizer.step()

                    # 更新队列
                    model.enqueue_dequeue(keys)

                # 打印结果
                scheduler.step()
                print('Epoch:{} Loss:{}'.format(epoch + 1, loss_hist[-1]))
            print('Optimization finished!')

            # Loss曲线
            plt.plot(loss_hist)
            plt.title('Loss curve (exp={}, sigma={})'.format(exp + 1, i))
            plt.xlabel('Epoch')
            plt.ylabel('Loss function value')
            plt.grid()
            plt.show()
            plt.plot(correct)
            plt.title('Correctly predicted (top-{}) (exp={}, sigma={})'.format(args.topk, exp + 1, i))
            plt.xlabel('Epoch')
            plt.ylabel('Number of correctly predicted samples')
            plt.grid()
            plt.show()

            # 数据编码
            model.eval()
            h = model.encode(torch.tensor(X_unsup, dtype=torch.float32, device=gpu)).cpu().numpy()

            # t-SNE降维可视化
            t_sne(h, y_unsup, title='With contrastive learning (exp={}, sigma={})'.format(exp + 1, i))

            # 下游软测量应用
            for j in sup_train:
                print('=====Labeled samples are {}====='.format(j))
                X_sup = X_unsup[:j]
                y_sup = y_unsup[:j]
                r2[exp, sigma.index(i), sup_train.index(j)] = downstream_task(X_sup, y_sup, X_test, y_test, model,
                                                                              title='With Contrastive Learning (exp={}, sigma={}, samples={})'.format(
                                                                                  exp + 1, i, j))

    # 展示结果
    print('=====Averaged results=====')
    mean = r2.mean(axis=0)
    std = r2.std(axis=0)
    print('Mean:', mean)
    print('Standard Deviation:', std)
    for i in range(mean.shape[0]):
        plt.errorbar([str(j) for j in sup_train], mean[i, :], marker='H', ms=3)
    plt.grid()
    plt.ylim(0, 1)
    plt.legend(['0.1x', '0.2x', '0.3x', '0.4x', '0.5x', 'NCL'])
    plt.show()

    # 存储结果
    np.save('r2.npy', r2)


if __name__ == '__main__':
    main()
