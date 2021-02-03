# 引入异常数据和对比学习的软测量模型

# 导入库
import math
import argparse
import pandas as pd
import seaborn as sns
from cl_packages import *
import matplotlib.pyplot as plt

# 参数解析
parser = argparse.ArgumentParser(description='Contrastive leaning soft sensor with anomalies')

# 实验参数
parser.add_argument('-dataset', type=str, default='denitrate_1')
parser.add_argument('-train_ratio', type=float, default=0.8)
parser.add_argument('-sup_ratio', type=float, default=0.1)
parser.add_argument('-seed', type=int, default=1)

# 网络参数
parser.add_argument('-encoder', type=tuple, default=(1024,))
parser.add_argument('-soft_sensor', type=tuple, default=(128,))
parser.add_argument('-projector', type=tuple, default=(256, 128))
parser.add_argument('-seq_len', type=int, default=20)
parser.add_argument('-period', type=int, default=50)
parser.add_argument('-n_epoch', type=int, default=200)
parser.add_argument('-batch_size', type=tuple, default=(64, 512, 11))
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-weight_decay', type=float, default=0.005)
parser.add_argument('-step_size', type=int, default=50)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-gpu', type=int, default=1)

# 自定义超参数
parser.add_argument('-k_near', type=int, default=3, help='number of neighbors for temporal consistency')
parser.add_argument('-threshold', type=float, default=0.1, help='threshold for temporal consistency')
parser.add_argument('-alpha', type=tuple, default=(0, 0, 0), help='coefficients for loss functions')
parser.add_argument('-num_pos', type=int, default=0, help='number of positive samples for contrastive learning')
parser.add_argument('-momentum', type=float, default=0.999, help='coefficient for momentum update')
parser.add_argument('-t', type=float, default=0.1, help='temperature hyper-parameter')
parser.add_argument('-sigma', type=float, default=0.01, help='the intensity of noise')
parser.add_argument('-p', type=float, default=0.8, help='probability for dropout')


# 划分数据
def split(data, args):
    time = pd.to_datetime(data.index)
    normal = data.iloc[:, -1].values
    data = data.iloc[:, :-1].values

    # 训练集大小
    train_size = int(args.train_ratio * data.shape[0])
    sup_size = int(args.sup_ratio * train_size)

    # 划分训练集和测试集
    X_train, y_train, normal_train = data[:train_size, :-1], data[:train_size, [-1]], normal[:train_size]
    X_test, y_test, normal_test = data[train_size:, :-1], data[train_size:, [-1]], normal[train_size:]

    # 判断有标签样本、无标签样本和异常样本的批次数是否一致
    print('Batch check:')
    unsup_size = sum(normal_train) - sup_size
    ab_size = train_size - sum(normal_train)
    if (math.ceil(sup_size / args.batch_size[0]) != math.ceil(unsup_size / args.batch_size[1])) | (
            math.ceil(sup_size / args.batch_size[0]) != math.ceil(ab_size / args.batch_size[2])):
        raise Exception('Batch does not match.')
    else:
        print('Batch matches!')

    # 相关系数矩阵
    # sns.heatmap(np.corrcoef(data[:train_size].T), -1, 1, annot=True, fmt='.2f', annot_kws={'fontsize': 6})
    # plt.show()

    # 数据标准化
    scaler = MinMaxScaler()
    X_train[normal_train] = scaler.fit_transform(X_train[normal_train])
    X_train[~normal_train] = scaler.transform(X_train[~normal_train])
    X_test = scaler.transform(X_test)

    # 转化为序列输入
    X_train = transform_3d(X_train, args.seq_len)
    X_test = transform_3d(X_test, args.seq_len)

    # 修正标签
    y_train = y_train[args.seq_len - 1:]
    y_test = y_test[args.seq_len - 1:]
    normal_train = normal_train[args.seq_len - 1:]
    normal_test = normal_test[args.seq_len - 1:]

    # 获得有标签、无标签、故障索引
    index_normal = np.where(normal_train == True)[0]
    np.random.shuffle(index_normal)
    index_sup = index_normal[:sup_size]
    index_unsup = index_normal[sup_size:]

    # 生成数据
    X_train = {'all': X_train, 'sup': X_train[index_sup], 'unsup': X_train[index_unsup],
               'normal': X_train[normal_train], 'abnormal': X_train[~normal_train]}
    y_train = {'all': y_train, 'sup': y_train[index_sup], 'unsup': y_train[index_unsup],
               'normal': y_train[normal_train], 'abnormal': y_train[~normal_train]}
    X_test = {'all': X_test, 'normal': X_test[normal_test], 'abnormal': X_test[~normal_test]}
    y_test = {'all': y_test, 'normal': y_test[normal_test], 'abnormal': y_test[~normal_test]}
    normal = {'all': np.concatenate((normal_train, normal_test)), 'train': normal_train, 'test': normal_test}

    # 可视化数据组成
    time_train = time[args.seq_len - 1:train_size]
    time_test = time[train_size + args.seq_len - 1:]
    time = np.concatenate((time_train, time_test))
    plt.plot(time_train, y_train['all'], zorder=0, label='unlabeled data', color='b')
    plt.plot(time_test, y_test['all'], zorder=0, label='test data', color='k')
    plt.scatter(time_train[index_sup], y_train['sup'], 10, 'g', zorder=1, label='labeled data')
    plt.scatter(time[~normal['all']], np.concatenate((y_train['abnormal'], y_test['abnormal'])), 10, 'r', zorder=1,
                label='abnormal data')
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('NOx in clean gas')
    plt.title('Data composition')
    plt.show()

    # 寻找每一个无标签样本是否存在有标签近邻
    u_near = -np.ones((index_unsup.shape[0]))
    for i in range(index_unsup.shape[0]):
        if ((index_sup <= index_unsup[i] + args.k_near) & (index_sup >= index_unsup[i] - args.k_near)).any():
            u_near[i] = \
                np.where((index_sup <= index_unsup[i] + args.k_near) & (index_sup >= index_unsup[i] - args.k_near))[0][
                    0]

    return X_train, y_train, X_test, y_test, normal, u_near


# 将数据转化为序列输入
def transform_3d(X, seq_len):
    X_3d = []
    for i in range(X.shape[0] - seq_len + 1):
        X_3d.append(X[i:i + seq_len])
    X_3d = np.stack(X_3d)

    return X_3d


# 主函数
def main():
    args = parser.parse_args()
    np.random.seed(args.seed)

    # 导入数据及数据划分
    data = pd.read_csv('data_' + args.dataset + '.csv', index_col=0, header=0, encoding='gbk')
    X_train, y_train, X_test, y_test, normal, u_near = split(data.iloc[:, 1:], args)
    dim_X = X_train['sup'].shape[-1]
    dim_y = y_train['sup'].shape[-1]

    # 对比使用全部有标签数据
    # model = arsenal.LstmModel(dim_X, dim_y, (1024,), weight_decay=0.01).fit(X_train['normal'][:, -1], y_train['normal'])
    # y_fit = model.predict(X_train['normal'][:, -1])
    # y_pred = model.predict(X_test['normal'][:, -1])
    # r2_fit = r2_score(y_train['normal'], y_fit)
    # r2_pred = r2_score(y_test['normal'], y_pred)
    # rmse_fit = np.sqrt(mean_squared_error(y_train['normal'], y_fit))
    # rmse_pred = np.sqrt(mean_squared_error(y_test['normal'], y_pred))
    # print(r2_fit, rmse_fit, r2_pred, rmse_pred)

    # 模型建立与训练
    regressor = CLSoftSensor(dim_X, args)
    regressor.fit(X_train['sup'], X_train['unsup'], X_train['abnormal'], X_test['normal'], y_train['sup'],
                  y_test['normal'], u_near)

    # 损失函数与中间特征可视化
    plot_loss([regressor.loss_hist, regressor.loss_hist_1, regressor.loss_hist_2, regressor.loss_hist_3,
               regressor.loss_hist_4])
    # t_sne(X_train['all'][:, -1], normal['train'], 'Original features')
    # t_sne(regressor.encode(X_train['all'], 'feature_lstm'), normal['train'], 'Hidden features')
    # t_sne(regressor.encode(X_train['all'], 'feature_cl'), normal['train'], 'Hidden features')

    # 模型性能评估
    print(regressor.eval(X_train['sup'], y_train['sup']))
    print(regressor.eval(X_train['normal'], y_train['normal']))
    print(regressor.eval(X_test['normal'], y_test['normal']))

    pass


if __name__ == '__main__':
    main()
