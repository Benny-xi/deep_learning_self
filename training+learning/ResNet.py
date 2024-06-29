import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
from matplotlib import pyplot as plt

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y = Y + x
        return F.relu(Y)

# 测试
"""blk = Residual(3, 3)
X = torch.randn(4, 3, 6, 6)
y = blk(X)
print(y.shape)

blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)"""

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

# 测试
"""X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)"""

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
    lr, num_epoch, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    train_loss_history, train_acc_history, test_acc_history = train_ch6(net, train_iter, test_iter, num_epoch, lr, d2l.try_gpu())


    plt.xlabel(xlabel='epoch')

    plt.plot(train_loss_history, label='train loss',)
    plt.plot(train_acc_history, label='train acc')
    plt.plot(test_acc_history, label='test acc')
    plt.legend()
    plt.grid(True)
    plt.show()