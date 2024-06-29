import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import matplotlib.pyplot as plt

"""def batch_norm(X, gamma, beta, moving_mean, moving_variance, eps, momentum):
    # 通过 is_grad_enabled 来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_variance + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = torch.mean(X, dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis = 1) 的均值和方差
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_variance = momentum * moving_variance + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_variance.data"""

"""class BatchNorm(nn.Module):
    # num_features ：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2 表示全连接层，4 表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化为1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_variance = torch.ones(shape)
    def forward(self, X):
        # 如果X不在内存上，将moving_mean 和 moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = (self.moving_mean.to(X.device))
            self.moving_variance = (self.moving_variance.to(X.device))
        # 保存更新过的moving_mean 和 moving_var
        Y, self.moving_mean, self.moving_variance = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_variance, eps=1e-5, momentum=0.9
        )
        return Y"""

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84),nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    train_loss_history = []   # 创建损失历史记录表
    train_acc_history = []   # 创建精确度历史记录表
    test_acc_history = []

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = d2l. evaluate_accuracy_gpu(net, test_iter)
        train_loss_history.append(train_l)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        print(f'epoch {epoch/num_epochs}', f'train_l={train_l}', f'train_acc={train_acc}', f'test_acc={test_acc}')

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

    return train_loss_history, train_acc_history, test_acc_history

if __name__ == '__main__':
    lr, num_epoch, batch_size = 0.9, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    train_loss_history, train_acc_history, test_acc_history = train_ch6(net, train_iter, test_iter, num_epoch, lr, d2l.try_gpu())

    # print("第一个批量规范化层中学到的拉伸参数gamma：")
    # print(net[1].gamma.reshape((-1,)))
    # print("第一个批量规范化层中学到的便移参数beta：")
    # print(net[1].beta.reshape((-1,)))

    plt.plot(train_loss_history, label='train loss')
    plt.plot(train_acc_history, label='train acc')
    plt.plot(test_acc_history, label='test acc')
    plt.legend()
    plt.grid(True)
    plt.show()

