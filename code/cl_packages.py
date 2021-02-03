# 导入库
import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.cm as cm
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append('..')
sys.path.append('../arsenal')
import arsenal


# =====相关函数=====

# 各变量可视化
def plot(data, columns):
    for i in range(data.shape[1]):
        plt.figure()
        plt.plot(data[:, i])
        plt.title(columns[i])
        plt.show()


# 绘制Loss曲线
def plot_loss(loss_hist, title='Loss curve'):
    plt.figure()
    for i in range(len(loss_hist)):
        plt.plot(loss_hist[i], label='Loss_{}'.format(i))
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()


# t-SNE降维可视化
def t_sne(x, y, title=None):
    x_transform = TSNE(n_components=2).fit_transform(x)
    plt.scatter(x_transform[y == True, 0], x_transform[y == True, 1], 3.0)
    plt.scatter(x_transform[y == False, 0], x_transform[y == False, 1], 3.0)
    # plt.scatter(x_transform[:, 0], x_transform[:, 1], 3.0, y, cmap=cm.get_cmap('viridis'))
    # plt.colorbar()
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    if title is not None:
        plt.title(title)
    plt.show()


# 数据类定义
class CLDataset(Dataset):

    def __init__(self, data, u_near=None):
        self.data = data
        self.u_near = u_near

    def __getitem__(self, index):
        if self.u_near is not None:
            return self.data[index], self.u_near[index]
        else:
            return self.data[index]

    def __len__(self):
        return self.data.shape[0]


# =====利用MoCo作为软测量的上游任务（缺少针对软测量任务的导向）=====

# 模型类定义
class MoCo(nn.Module):

    # 初始化
    def __init__(self, dim_X, encoder=(1024, 512), projector=(128,), queue_length=1024, momentum=0.999, t=0.07,
                 gpu=torch.device('cuda:0')):

        # 初始化父类
        super(MoCo, self).__init__()

        # 参数赋值
        self.dim_X = dim_X
        self.encoder = encoder
        self.projector = projector
        self.q = queue_length
        self.m = momentum
        self.t = t
        self.gpu = gpu

        # 初始化编码网络
        self.encoder_q = nn.ModuleList()
        self.encoder_k = nn.ModuleList()
        net_encoder = [dim_X, ] + list(encoder)

        # 使用全连接网络作为encoder
        for i in range(len(encoder)):
            self.encoder_q.append(nn.Sequential(nn.Linear(net_encoder[i], net_encoder[i + 1]), nn.ReLU()))
            self.encoder_k.append(nn.Sequential(nn.Linear(net_encoder[i], net_encoder[i + 1]), nn.ReLU()))

        # encoder_q和encoder_k参数相同初始化
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 判断是否在编码网络后构建投影网络
        if self.projector is not None:

            # 初始化投影网络
            self.projector_q = nn.ModuleList()
            self.projector_k = nn.ModuleList()
            net_projector = [encoder[-1], ] + list(projector)

            # 使用全连接网络作为projector
            for i in range(len(projector)):
                self.projector_q.append(nn.Sequential(nn.Linear(net_projector[i], net_projector[i + 1]), nn.ReLU()))
                self.projector_k.append(nn.Sequential(nn.Linear(net_projector[i], net_projector[i + 1]), nn.ReLU()))

            # projector_q和projector_k参数相同初始化
            for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        # 初始化队列
        self.queue = torch.randn(self.q, projector[-1] if self.projector is not None else encoder[-1],
                                 dtype=torch.float32, device=self.gpu)
        self.queue = nn.functional.normalize(self.queue, dim=1)

    # 更新encoder_k和projector_k的参数
    @torch.no_grad()
    def update_k(self):

        # 更新encoder_k的参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        # 判断是否构建了投影网络
        if self.projector is not None:

            # 更新projector_k的参数
            for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    # 更新队列
    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        length = keys.shape[0]
        self.queue = torch.cat((self.queue[length:, :], keys), dim=0)

    # 数据编码
    @torch.no_grad()
    def encode(self, x):
        h = x
        for i in self.encoder_q:
            h = i(h)
        h = nn.functional.normalize(h, dim=1)

        return h

    # 前向传播
    def forward(self, x_q, x_k):

        # 计算query
        h_q = x_q

        # 编码
        for i in self.encoder_q:
            h_q = i(h_q)
        h_q = nn.functional.normalize(h_q, dim=1)

        # 判断是否构建了投影网络
        if self.projector is not None:

            # 投影
            for i in self.projector_q:
                h_q = i(h_q)
            h_q = nn.functional.normalize(h_q, dim=1)

        # 计算key
        with torch.no_grad():
            self.update_k()
            h_k = x_k

            # 编码
            for i in self.encoder_k:
                h_k = i(h_k)
            h_k = nn.functional.normalize(h_k, dim=1)

            # 判断是否构建了投影网络
            if self.projector is not None:

                # 投影
                for i in self.projector_k:
                    h_k = i(h_k)
                h_k = nn.functional.normalize(h_k, dim=1)

        # 计算query和key的相似程度
        l_pos = torch.diagonal(torch.matmul(h_q, h_k.t())).unsqueeze(-1)
        l_neg = torch.matmul(h_q, self.queue.t())

        # 生成相似程度矩阵和标签
        l = torch.cat([l_pos, l_neg], dim=1) / self.t
        label = torch.zeros(l.shape[0], dtype=torch.long, device=self.gpu)

        return l, label, h_k


# 封装类定义
class MoCoSoftSensor(BaseEstimator, RegressorMixin):

    # 初始化
    def __init__(self, dim_X, encoder=(1024, 512), projector=(128,), queue_length=1024, momentum=0.999, t=0.07,
                 noise=0.1, n_epoch=200, batch_size=64, lr=0.0001, weight_decay=0.01, step_size=50, gamma=0.5, topk=3,
                 gpu=torch.device('cuda:0'), seed=1):

        # 初始化父类
        super(MoCoSoftSensor, self).__init__()

        # 设置种子
        torch.manual_seed(seed)

        # 参数赋值
        self.dim_X = dim_X
        self.encoder = encoder
        self.projector = projector
        self.queue_length = queue_length
        self.momentum = momentum
        self.t = t
        self.noise = noise
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.topk = topk
        self.gpu = gpu
        self.seed = seed

        # 初始化标准器
        self.scaler_X = MinMaxScaler()

        # 模型生成
        self.loss_hist = []
        self.correct = []
        self.model = MoCo(dim_X, encoder, projector, queue_length, momentum, t, gpu).to(gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.CrossEntropyLoss()
        self.regressor = LinearRegression()

    # 上游训练
    def fit_pretext(self, X):
        X = self.scaler_X.fit_transform(X)
        dataset = CLDataset(torch.tensor(X, dtype=torch.float32, device=self.gpu))
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            self.correct.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X in data_loader:
                self.optimizer.zero_grad()
                q = batch_X
                k = q + self.noise * torch.randn(q.shape, dtype=torch.float32, device=self.gpu)
                l, label, keys = self.model(q, k)
                loss = self.criterion(l, label)
                self.loss_hist[-1] += loss.item()
                _, pos = torch.topk(l, self.topk, 1)
                self.correct[-1] += (pos == 0).sum().item()
                loss.backward()
                self.optimizer.step()
                self.model.enqueue_dequeue(keys)
            self.scheduler.step()
            print('Epoch:{} Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # 下游训练
    def fit_downstream(self, X, y, encode=True):
        if encode:
            h = self.encode(X)
        else:
            h = self.scaler_X.fit_transform(X)
        self.regressor.fit(h, y)

        return self

    # 编码
    def encode(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu)
        self.model.eval()
        feat = self.model.encode(X).cpu().numpy()

        return feat

    # 预测
    def predict(self, X, encode=True):
        if encode:
            h = self.encode(X)
        else:
            h = self.scaler_X.transform(X)
        y = self.regressor.predict(h)

        return y

    # 性能评估
    def eval(self, X, y, title, encode=True):
        y_hat = self.predict(X, encode)
        r2 = 100 * r2_score(y, y_hat)
        rmse = np.sqrt(mean_squared_error(y, y_hat))
        print('Performance(R2 MSE): {:.2f}% {:.3f}'.format(r2, rmse))
        plt.subplot(211)
        plt.plot(y)
        plt.plot(y_hat)
        plt.grid()
        plt.legend(['Ground Truth', 'Prediction'])
        plt.title(title)
        plt.subplot(212)
        plt.plot(y - y_hat)
        plt.grid()
        plt.title('Error')
        plt.show()

        return r2, rmse


# =====通过标签的线性组合关系约束特征的线性组合关系建立半监督软测量模型=====

# 模型类定义
class SemiSupervisedLearning(nn.Module):

    # 初始化
    def __init__(self, dim_X, hidden_layers=(512, 128)):

        # 初始化父类
        super(SemiSupervisedLearning, self).__init__()

        # 参数赋值
        self.dim_X = dim_X
        self.hidden_layers = hidden_layers
        self.net_structure = [dim_X, ] + list(hidden_layers) + [1, ]

        # 初始化网络
        self.net = nn.ModuleList()
        for i in range(len(hidden_layers)):
            self.net.append(nn.Sequential(nn.Linear(self.net_structure[i], self.net_structure[i + 1]), nn.ReLU()))
        self.net.append(nn.Linear(self.net_structure[-2], self.net_structure[-1]))

    # 前向传播
    def forward(self, X_l, y_l, X_u, y_u):
        feat_l = X_l
        feat_u = X_u

        for i in self.net[:-1]:
            feat_l = i(feat_l)
            feat_u = i(feat_u)

        y_hat = self.net[-1](feat_l)

        size_l = X_l.shape[0]
        size_u = X_u.shape[0]
        feat_comb = torch.zeros(feat_u.shape)
        for i in range(size_u):
            flag = 1
            while flag == 1:
                index = torch.randperm(size_l)
                index_a, index_b = index[0], index[1]
                if y_l[index_a] != y_l[index_b]:
                    flag = 0
            k = (y_u[i] - y_l[index_b]) / (y_l[index_a] - y_l[index_b])
            feat_comb[i] = k * feat_l[index_a] + (1 - k) * feat_l[index_b]

        return feat_u, feat_comb, y_hat

    # 预测结果
    def predict(self, X):
        feat = X
        for i in self.net:
            feat = i(feat)
        return feat

    # 编码
    @torch.no_grad()
    def encode(self, X):
        feat = X
        for i in self.net[:-1]:
            feat = i(feat)
        return feat


# 损失函数类定义
class SSLLoss(nn.Module):

    # 初始化
    def __init__(self):
        super(SSLLoss, self).__init__()

    # 前向传播
    def forward(self, feat_u, feat_comb, y_l, y_hat, alpha):
        term1 = torch.norm(feat_u - feat_comb) ** 2 / (feat_u.shape[0] * feat_u.shape[1])
        term2 = torch.norm(y_l - y_hat) ** 2 / y_l.shape[0]
        return term1 + alpha * term2, term1, term2


# 封装类定义
class SSLSoftSensor(BaseEstimator, RegressorMixin):

    # 初始化
    def __init__(self, dim_X, hidden_layers=(512, 128), alpha=0.01, period=10, n_epoch_1=200, n_epoch_2=50,
                 batch_size_1=64, batch_size_2=64, lr_1=0.0001, lr_2=0.0001, weight_decay_1=0.01, weight_decay_2=0.01,
                 step_size_1=50, step_size_2=50, gamma_1=0.5, gamma_2=0.5, gpu=torch.device('cuda:0'), seed=1):

        # 初始化父类
        super(SSLSoftSensor, self).__init__()

        # 设置种子
        torch.manual_seed(seed)

        # 参数赋值
        self.dim_X = dim_X
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.period = period
        self.n_epoch_1 = n_epoch_1
        self.n_epoch_2 = n_epoch_2
        self.batch_size_1 = batch_size_1
        self.batch_size_2 = batch_size_2
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.weight_decay_1 = weight_decay_1
        self.weight_decay_2 = weight_decay_2
        self.step_size_1 = step_size_1
        self.step_size_2 = step_size_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gpu = gpu
        self.seed = seed

        # 初始化标准器
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # 模型生成
        self.loss_hist = []
        self.loss_hist_feat = []
        self.loss_hist_y = []
        self.model = SemiSupervisedLearning(dim_X, hidden_layers).to(gpu)

    # 用有标签数据训练软测量模型
    def fit_stage1(self, X_l, y_l):
        print('=====Pre-training with labeled samples=====')
        self.scaler_X.fit(X_l)
        self.X_l = torch.tensor(self.scaler_X.transform(X_l), dtype=torch.float32, device=self.gpu)
        self.scaler_y.fit(y_l)
        self.y_l = torch.tensor(self.scaler_y.transform(y_l), dtype=torch.float32, device=self.gpu)
        dataset = arsenal.packages.MyDataset(self.X_l, self.y_l)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr_1, weight_decay=self.weight_decay_1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.step_size_1, self.gamma_1)
        criterion = nn.MSELoss(reduction='sum')
        for i in range(self.n_epoch_1):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size_1, shuffle=True)
            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()
                y_hat = self.model.predict(batch_X)
                loss = criterion(y_hat, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()
            print('Epoch: {}, Loss: {:.3f}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # 用有标签数据和无标签数据训练软测量模型
    def fit_stage2(self, X_u):
        print('=====Training with labeled and unlabeled samples together=====')
        X_u = torch.tensor(self.scaler_X.transform(X_u), dtype=torch.float32, device=self.gpu)
        self.loss_hist = []
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr_2, weight_decay=self.weight_decay_2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.step_size_2, self.gamma_2)
        criterion = SSLLoss()
        for i in range(self.n_epoch_2):
            if i % self.period == 0:
                y_u = torch.tensor(self.scaler_y.transform(self.model.predict(X_u).detach().cpu().numpy()),
                                   dtype=torch.float32, device=self.gpu)
                dataset = arsenal.packages.MyDataset(X_u, y_u)
            self.loss_hist.append(0)
            self.loss_hist_feat.append(0)
            self.loss_hist_y.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size_2, shuffle=True)
            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()
                feat_u, feat_comb, y_hat = self.model(self.X_l, self.y_l, batch_X, batch_y)
                loss, loss_feat, loss_y = criterion(feat_u, feat_comb.to(self.gpu), self.y_l, y_hat, self.alpha)
                self.loss_hist[-1] += loss.item()
                self.loss_hist_feat[-1] += loss_feat.item()
                self.loss_hist_y[-1] += loss_y.item()
                loss.backward()
                optimizer.step()
            scheduler.step()
            print('Epoch: {}, Loss: {:.3f}, Loss_feat: {:.3f}, Loss_y: {:.3f}'.format(i + 1, self.loss_hist[-1],
                                                                                      self.loss_hist_feat[-1],
                                                                                      self.loss_hist_y[-1]))
        print('Optimization finished!')

        return self

    # 编码
    def encode(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu)
        self.model.eval()
        feat = self.model.encode(X).cpu().numpy()

        return feat

    # 测试
    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu)
        self.model.eval()
        y = self.scaler_y.inverse_transform(self.model.predict(X).detach().cpu().numpy())

        return y

    # 性能评估
    def eval(self, X, y, title):
        y_hat = self.predict(X)
        r2 = 100 * r2_score(y, y_hat)
        rmse = np.sqrt(mean_squared_error(y, y_hat))
        print('Performance(R2 MSE): {:.2f}% {:.3f}'.format(r2, rmse))
        plt.subplot(211)
        plt.plot(y)
        plt.plot(y_hat)
        plt.grid()
        plt.legend(['Ground Truth', 'Prediction'])
        plt.title(title)
        plt.subplot(212)
        plt.plot(y - y_hat)
        plt.grid()
        plt.title('Error')
        plt.show()

        return r2, rmse


# =====利用故障数据建立软测量模型=====

# 模型类定义
class ContrastiveLearning(nn.Module):

    # 初始化
    def __init__(self, dim_X, encoder=(512,), soft_sensor=(256, 128), projector=(128,), momentum=0.999):

        # 初始化父类
        super(ContrastiveLearning, self).__init__()

        # 参数赋值
        self.dim_X = dim_X
        self.momentum = momentum
        self.net_encoder = [dim_X, ] + list(encoder)
        self.net_soft_sensor = [encoder[-1], ] + list(soft_sensor) + [1, ]
        self.net_projector = [encoder[-1], ] + list(projector)

        # 初始化网络
        self.encoder = nn.ModuleList()
        self.encoder_ab = nn.ModuleList()
        self.soft_sensor = nn.ModuleList()
        self.projector = nn.ModuleList()
        self.projector_ab = nn.ModuleList()
        for i in range(len(encoder)):
            self.encoder.append(nn.LSTM(self.net_encoder[i], self.net_encoder[i + 1], batch_first=True))
            self.encoder_ab.append(nn.LSTM(self.net_encoder[i], self.net_encoder[i + 1], batch_first=True))
        for i in range(len(soft_sensor)):
            self.soft_sensor.append(nn.Sequential(nn.Linear(self.net_soft_sensor[i], self.net_soft_sensor[i + 1]),
                                                  nn.BatchNorm1d(self.net_soft_sensor[i + 1]), nn.ReLU()))
        self.soft_sensor.append(nn.Linear(self.net_soft_sensor[-2], self.net_soft_sensor[-1]))
        for i in range(len(projector)):
            self.projector.append(nn.Sequential(nn.Linear(self.net_projector[i], self.net_projector[i + 1]), nn.ReLU()))
            self.projector_ab.append(
                nn.Sequential(nn.Linear(self.net_projector[i], self.net_projector[i + 1]), nn.ReLU()))

        # 参数同步初始化
        for param, param_ab in zip(self.encoder.parameters(), self.encoder_ab.parameters()):
            param_ab.data.copy_(param)
            param_ab.requires_grad = False
        for param, param_ab in zip(self.projector.parameters(), self.projector_ab.parameters()):
            param_ab.data.copy_(param)
            param_ab.requires_grad = False

    # 参数动量更新
    @torch.no_grad()
    def update(self):

        # 更新encoder
        for param, param_ab in zip(self.encoder.parameters(), self.encoder_ab.parameters()):
            param_ab.data = param_ab.data * self.momentum + param.data * (1 - self.momentum)

        # 更新projector
        for param, param_ab in zip(self.projector.parameters(), self.projector_ab.parameters()):
            param_ab.data = param_ab.data * self.momentum + param.data * (1 - self.momentum)

    # 前向传播
    def forward(self, X, mode='both', data='normal'):
        h = X

        # LSTM运算
        if data == 'normal':
            for i in self.encoder:
                h, _ = i(h)
        elif data == 'abnormal':
            self.update()
            for i in self.encoder_ab:
                h, _ = i(h)
        else:
            raise Exception('Wrong data selection.')
        feature = _[0].squeeze()
        y = feature

        # 只输出时序特征
        if mode == 'feature_lstm':
            return feature

        # 只输出产生对比学习损失的特征
        elif mode == 'feature_cl':
            if data == 'normal':
                for i in self.projector:
                    feature = i(feature)
            elif data == 'abnormal':
                for i in self.projector_ab:
                    feature = i(feature)
            else:
                raise Exception('Wrong data selection.')
            feature = nn.functional.normalize(feature, dim=1)
            return feature

        # 只输出标签
        elif mode == 'label':
            for i in self.soft_sensor:
                y = i(y)
            return y

        # 同时输出特征和标签
        elif mode == 'both':
            if data == 'normal':
                for i in self.projector:
                    feature = i(feature)
            elif data == 'abnormal':
                for i in self.projector_ab:
                    feature = i(feature)
            else:
                raise Exception('Wrong data selection.')
            feature = nn.functional.normalize(feature, dim=1)
            for i in self.soft_sensor:
                y = i(y)
            return feature, y

        # 异常
        else:
            raise Exception('Wrong mode selection.')


# 损失函数类定义
class CLLoss(nn.Module):

    # 初始化
    def __init__(self, threshold, alpha, num_pos, t):

        # 初始化父类
        super(CLLoss, self).__init__()

        # 参数赋值
        self.threshold = threshold
        self.alpha = alpha
        self.num_pos = num_pos
        self.t = t

    # 前向传播
    def forward(self, f_l, f_l_noise, f_u, f_u_noise, f_ab, y, batch_y, y_l, y_u, y_u_noise, u_near):

        # 有标签损失
        term0 = torch.norm(batch_y - y_l) ** 2 / y_l.shape[0]

        # 一致性损失
        if self.alpha[0] != 0:
            term1 = torch.norm(y_u - y_u_noise) ** 2 / y_u.shape[0]
        else:
            term1 = torch.tensor(0)

        # 时序一致性损失
        if self.alpha[1] != 0:
            temp = torch.abs(y_u[u_near != -1] - y[u_near[u_near != -1]])
            temp[temp < self.threshold] = 0
            term2 = torch.norm(temp) ** 2 / temp.shape[0]
        else:
            term2 = torch.tensor(0)

        # 故障数据特征层面损失
        if self.alpha[2] != 0:
            f_n = torch.cat((f_l, f_u))
            f_n_noise = torch.cat((f_l_noise, f_u_noise))
            l_pos = torch.zeros((f_n.shape[0], self.num_pos), device=f_n.device)
            if self.num_pos != 0:
                for i in range(f_n.shape[0]):
                    index = [j for j in range(f_n.shape[0])]
                    index.remove(i)
                    np.random.shuffle(index)
                    l_pos[i] = torch.matmul(f_n[i], f_n[index[:self.num_pos]].T)
            l_pos = torch.cat((l_pos, torch.matmul(f_n, f_n_noise.T).diagonal().reshape(-1, 1)), dim=1)
            l_pos = torch.exp(l_pos / self.t)
            l_neg = torch.exp(torch.matmul(f_n, f_ab.T) / self.t)
            l = torch.cat((l_pos, l_neg), dim=1)
            term3 = -torch.log(l_pos.sum(dim=1) / l.sum(dim=1)).mean(dim=0)
        else:
            term3 = torch.tensor(0)

        # 总损失
        total_loss = term0 + self.alpha[0] * term1 + self.alpha[1] * term2 + self.alpha[2] * term3

        return total_loss, term0, self.alpha[0] * term1, self.alpha[1] * term2, self.alpha[2] * term3


# 封装类定义
class CLSoftSensor(BaseEstimator, RegressorMixin):

    # 初始化
    def __init__(self, dim_X, args):

        # 初始化父类
        super(CLSoftSensor, self).__init__()

        # 设置种子
        torch.manual_seed(args.seed)

        # 参数赋值
        self.dim_X = dim_X
        self.encoder = args.encoder
        self.soft_sensor = args.soft_sensor
        self.projector = args.projector
        self.period = args.period
        self.n_epoch = args.n_epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.gpu = args.gpu
        self.seed = args.seed
        self.threshold = args.threshold
        self.alpha = args.alpha
        self.num_pos = args.num_pos
        self.momentum = args.momentum
        self.t = args.t
        self.sigma = args.sigma
        self.p = args.p

        # 初始化标准器
        self.scaler_y = MinMaxScaler()

        # 模型生成
        self.loss_hist = []
        self.loss_hist_1 = []
        self.loss_hist_2 = []
        self.loss_hist_3 = []
        self.loss_hist_4 = []
        self.model = ContrastiveLearning(dim_X, args.encoder, args.soft_sensor, args.projector, args.momentum).to(
            args.gpu)
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, args.step_size, args.gamma)
        self.criterion = CLLoss(args.threshold, args.alpha, args.num_pos, args.t)

    # 训练
    def fit(self, X_l, X_u, X_ab, X_test, y_true, y_test, u_near):
        X_l = torch.tensor(X_l, dtype=torch.float32, device=self.gpu)
        X_u = torch.tensor(X_u, dtype=torch.float32, device=self.gpu)
        X_ab = torch.tensor(X_ab, dtype=torch.float32, device=self.gpu)
        y_true = torch.tensor(self.scaler_y.fit_transform(y_true), dtype=torch.float32, device=self.gpu)
        dataset_l = CLDataset(X_l, y_true)
        dataset_u = CLDataset(X_u, u_near)
        dataset_ab = CLDataset(X_ab)
        dataloader_l = DataLoader(dataset_l, self.batch_size[0], shuffle=False)
        dataloader_u = DataLoader(dataset_u, self.batch_size[1], shuffle=False)
        dataloader_ab = DataLoader(dataset_ab, self.batch_size[2], shuffle=False)
        self.model.train()
        for i in range(self.n_epoch):
            if i % self.period == 0:
                print(self.eval(X_test, y_test))
                self.model.train()
            self.loss_hist.append(0)
            self.loss_hist_1.append(0)
            self.loss_hist_2.append(0)
            self.loss_hist_3.append(0)
            self.loss_hist_4.append(0)
            for batch_l, batch_u, batch_ab in zip(dataloader_l, dataloader_u, dataloader_ab):
                self.optimizer.zero_grad()
                if self.alpha[2] != 0:
                    f_l, y_l = self.model(batch_l[0], 'both')
                    X_l_noise = batch_l[0].clone()
                    X_l_noise[:, -1] = X_l_noise[:, -1] + self.sigma * torch.nn.functional.dropout(
                        torch.randn(X_l_noise[:, -1].shape).to(self.gpu), p=self.p)
                    f_l_noise = self.model(X_l_noise, 'feature_cl')
                    f_ab = self.model(batch_ab, 'feature_cl', 'abnormal')
                else:
                    y_l = self.model(batch_l[0], 'label')
                    f_l, f_l_noise, f_ab = None, None, None
                if ((self.alpha[0] != 0) | (self.alpha[1] != 0)) & (self.alpha[2] != 0):
                    f_u, y_u = self.model(batch_u[0], 'both')
                elif ((self.alpha[0] != 0) | (self.alpha[1] != 0)) & (self.alpha[2] == 0):
                    y_u = self.model(batch_u[0], 'label')
                    f_u = None
                elif ((self.alpha[0] == 0) & (self.alpha[1] == 0)) & (self.alpha[2] != 0):
                    f_u = self.model(batch_u[0], 'feature_cl')
                    y_u = None
                else:
                    f_u, y_u = None, None
                if (self.alpha[0] != 0) | (self.alpha[2] != 0):
                    batch_X_u_noise = batch_u[0].clone()
                    batch_X_u_noise[:, -1] = batch_X_u_noise[:, -1] + self.sigma * torch.nn.functional.dropout(
                        torch.randn(batch_X_u_noise[:, -1].shape).to(self.gpu), p=self.p)
                    if (self.alpha[0] != 0) & (self.alpha[2] != 0):
                        f_u_noise, y_u_noise = self.model(batch_X_u_noise, 'both')
                    elif self.alpha[0] == 0:
                        f_u_noise = self.model(batch_X_u_noise, 'feature_cl')
                        y_u_noise = None
                    else:
                        y_u_noise = self.model(batch_X_u_noise, 'label')
                        f_u_noise = None
                else:
                    f_u_noise, y_u_noise = None, None
                loss, loss_1, loss_2, loss_3, loss_4 = self.criterion(f_l, f_l_noise, f_u, f_u_noise, f_ab, y_true,
                                                                      batch_l[1], y_l, y_u, y_u_noise,
                                                                      batch_u[1].long())
                self.loss_hist[-1] += loss.item()
                self.loss_hist_1[-1] += loss_1.item()
                self.loss_hist_2[-1] += loss_2.item()
                self.loss_hist_3[-1] += loss_3.item()
                self.loss_hist_4[-1] += loss_4.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print('Epoch: {}, Loss: {:.3f}, Loss_1: {:.3f}, Loss_2: {:.3f}, Loss_3: {:.3f}, Loss_4: {:.3f}'. \
                  format(i + 1, self.loss_hist[i], self.loss_hist_1[i], self.loss_hist_2[i], self.loss_hist_3[i],
                         self.loss_hist_4[i]))
        print('Optimization finished!')

        return self

    # 编码
    def encode(self, X, mode='feature_cl'):
        X = torch.tensor(X, dtype=torch.float32, device=self.gpu)
        self.model.eval()
        f = self.model(X, mode).detach().cpu().numpy()

        return f

    # 测试
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.gpu)
        self.model.eval()
        y = self.scaler_y.inverse_transform(self.model(X, 'label').detach().cpu().numpy())

        return y

    # 性能评估
    def eval(self, X, y):

        # 预测结果
        y_hat = self.predict(X)

        # 计算性能指标
        r2 = 100 * r2_score(y, y_hat)
        rmse = np.sqrt(mean_squared_error(y, y_hat))

        # 画图
        plt.subplot(211)
        plt.plot(y, label='Ground Truth')
        plt.plot(y_hat, label='Prediction')
        plt.grid()
        plt.legend()
        plt.xlabel('Sample')
        plt.ylabel('NOx in clean gas')
        plt.title('Performance (R2 RMSE): {:.2f}% {:.3f}'.format(r2, rmse))
        plt.subplot(212)
        plt.plot(y - y_hat, label='Error')
        plt.grid()
        plt.legend()
        plt.xlabel('Sample')
        plt.ylabel('Error')
        plt.title('Error')
        plt.show()

        return r2, rmse
