# 导入库
import random
import argparse
import pandas as pd
from cl_packages import *

# 参数解析
parser = argparse.ArgumentParser(description='Semi-supervised learning soft sensor')

# 实验参数
parser.add_argument('--dataset', type=int, default=1)
parser.add_argument('--start_point', type=int, default=0)
parser.add_argument('--length', type=int, default=10000)
parser.add_argument('--num_exp', type=int, default=3)
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--sup_ratio', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)

# 训练参数
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--period', type=int, default=1)
parser.add_argument('--n_epoch_1', type=int, default=200)
parser.add_argument('--n_epoch_2', type=int, default=200)
parser.add_argument('--batch_size_1', type=int, default=64)
parser.add_argument('--batch_size_2', type=int, default=64)
parser.add_argument('--lr_1', type=float, default=0.0001)
parser.add_argument('--lr_2', type=float, default=0.0001)
parser.add_argument('--weight_decay_1', type=float, default=0.1)
parser.add_argument('--weight_decay_2', type=float, default=0.001)
parser.add_argument('--step_size_1', type=int, default=50)
parser.add_argument('--step_size_2', type=int, default=50)
parser.add_argument('--gamma_1', type=float, default=0.5)
parser.add_argument('--gamma_2', type=float, default=0.5)


# 主函数
def main():
    args = parser.parse_args()
    random.seed(args.seed)
    r2 = pd.DataFrame(np.zeros((args.num_exp, 4)),
                      columns=['stage1_train', 'stage1_test', 'stage2_train', 'stage2_test'])
    rmse = pd.DataFrame(np.zeros((args.num_exp, 4)),
                        columns=['stage1_train', 'stage1_test', 'stage2_train', 'stage2_test'])

    # 转炉炼钢数据
    if args.dataset == 1:
        data = pd.read_csv('data_preprocess_1.csv', index_col=0)
        data.drop(['记录时间', '熔炼号', '钢种', '实际值-低碳锰铁', 'sb_record_time', 'sb_record_time_x', 'sb_record_time_y'], axis=1,
                  inplace=True)
        X_columns = [i for i in range(35)]
        y_columns = [-11, ]
        shuffle = True

    # 脱硝数据
    elif args.dataset == 2:
        data = pd.read_csv('data_preprocess_2.csv', index_col=0).iloc[args.start_point:args.start_point + args.length]
        data = data[
            (data.iloc[:, [2, 3, 4, 5, 6, 7, 11, 13]] == 0).all(axis=1) & (data.iloc[:, [8, 12]] == 1).all(axis=1)]
        X_columns = [10, 14, 15, 16, 17, 18, 19, 20, 21, 22, -4, -3, -1]
        y_columns = [9, ]
        shuffle = False

    # 浮选矿数据
    elif args.dataset == 3:
        data = pd.read_csv('data_preprocess_3.csv')
        data['date'] = pd.to_datetime(data['date'])
        g = data.groupby(by='date')
        data = g.mean()
        X_columns = [i for i in range(21)]
        y_columns = [-1, ]
        shuffle = False

    X = data.iloc[:, X_columns].values
    y = data.iloc[:, y_columns].values

    # 过程变量和质量变量可视化
    # plot(X, data.columns[X_columns])
    # plot(y, data.columns[y_columns])

    # 多次实验
    for exp in range(args.num_exp):
        print('=====Experiment({}/{})====='.format(exp + 1, args.num_exp))

        # 数据划分
        train_size = int(data.shape[0] * args.train_ratio)
        sup_size = int(train_size * args.sup_ratio)
        if shuffle:
            index = list(range(data.shape[0]))
            random.shuffle(index)
            X_sup, X_unsup, X_test = X[index[:sup_size]], X[index[sup_size:train_size]], X[index[train_size:]]
            y_sup, y_unsup, y_test = y[index[:sup_size]], y[index[sup_size:train_size]], y[index[train_size:]]
        else:
            index = list(range(train_size))
            random.shuffle(index)
            X_sup, X_unsup, X_test = X[index[:sup_size]], X[index[sup_size:]], X[train_size:]
            y_sup, y_unsup, y_test = y[index[:sup_size]], y[index[sup_size:]], y[train_size:]

        # 原始特征可视化
        t_sne(X[index[:train_size]], y[index[:train_size]], 'Original Feature')

        # 软测量模型构建
        net = (1024, 512, 256)
        regressor = SSLSoftSensor(X_sup.shape[1], net, args.alpha, args.period, args.n_epoch_1, args.n_epoch_2,
                                  args.batch_size_1, args.batch_size_2, args.lr_1, args.lr_2, args.weight_decay_1,
                                  args.weight_decay_2, args.step_size_1, args.step_size_2, args.gamma_1, args.gamma_2,
                                  args.gpu, args.seed)

        # 训练第一阶段
        regressor.fit_stage1(X_sup, y_sup)

        # 第一阶段后特征可视化
        t_sne(regressor.encode(X[index[:train_size]]), y[index[:train_size]], 'Feature After Stage One')

        # 绘制Loss
        plot_loss(regressor.loss_hist, 'Total Loss')

        # 性能评估
        r2.iloc[exp, 0], rmse.iloc[exp, 0] = regressor.eval(X_sup, y_sup, 'Performance on train set')
        r2.iloc[exp, 1], rmse.iloc[exp, 1] = regressor.eval(X_test, y_test, 'Performance on test set')

        # 训练第二阶段
        regressor.fit_stage2(X_unsup)

        # 第二阶段后特征可视化
        t_sne(regressor.encode(X[index[:train_size]]), y[index[:train_size]], 'Feature After Stage Two')

        # 绘制Loss
        plot_loss(regressor.loss_hist, 'Total Loss')
        plot_loss(regressor.loss_hist_feat, 'Feature Loss')
        plot_loss(regressor.loss_hist_y, 'Label Loss')

        # 性能评估
        r2.iloc[exp, 2], rmse.iloc[exp, 2] = regressor.eval(X_sup, y_sup, 'Performance on train set')
        r2.iloc[exp, 3], rmse.iloc[exp, 3] = regressor.eval(X_test, y_test, 'Performance on test set')

        print()


if __name__ == '__main__':
    main()
