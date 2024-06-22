import torch
import torch.nn as nn
from d2l import torch as d2l
from d2l.torch import evaluate_accuracy_gpu

"""创建nin块"""
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),  # 创建 ReLU 的实例
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),  # 创建 ReLU 的实例
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()   # 创建 ReLU 的实例
    )

"""创建网络"""
net = nn.Sequential(
    nin_block(1, 96, 11, 4, 0),
    nn.MaxPool2d(3, 2),
    nin_block(96, 256, 5, 1, 2),
    nn.MaxPool2d(3, 2),
    nin_block(256, 384, 3, 1, 1),
    nn.MaxPool2d(3, 2),
    nn.Dropout(0.5),
    # 标签类别是10
    nin_block(384, 10, 3, 1, 1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为（批量大小，10）
    nn.Flatten()
)

"""测试"""
# X = torch.rand(size = (1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):

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
        # Sum of training loss, sum of training accuracy, no. of examples
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
        test_acc = evaluate_accuracy_gpu(net, test_iter)

        print(f'epoch {epoch/num_epochs}', f'train_l={train_l}', f'train_acc={train_acc}', f'test_acc={test_acc}')

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    train_ch6(net,train_iter, test_iter, num_epochs, lr, d2l.try_gpu())