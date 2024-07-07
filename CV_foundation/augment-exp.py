import os

import numpy as np
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt

# all_images = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)

"""展示图片"""
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
# d2l.plt.show()

"""  定义训练和测试时使用的数据增强操作。"""
train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                             torchvision.transforms.ToTensor()])
test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def load_cifar10(is_train, augs, batch_size):
    """
    加载CIFAR-10数据集，并返回一个DataLoader对象。

    Args:
        is_train (bool): 是否为训练集，True表示加载训练集，False表示加载测试集。
        augs (callable, optional): 对图像进行预处理或增强的函数或方法，默认为None。
        batch_size (int): DataLoader对象每次返回的样本数。

    Returns:
        torch.utils.data.DataLoader: 加载CIFAR-10数据集后的DataLoader对象，用于批量读取数据。

    """
    dataset = torchvision.datasets.CIFAR10(root='../data', train=is_train, transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,num_workers=d2l.get_dataloader_workers())
    return dataloader

def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):

    train_loss_history = []
    train_acc_history = []
    test_acc_history = []

    # 指定文件路径在checkpoints文件夹下
    checkpoints_dir = 'checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)  # 如果文件夹不存在，则创建它
    filename = os.path.join(checkpoints_dir, 'aug_Adam.txt')  # 指定文件名和路径

    timer, num_batches = d2l.Timer(), len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)  # 更改了Accumulator的参数以匹配add方法的调用
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())  # 确保metric的参数匹配
            timer.stop()
            # print(f'epoch {epoch + 1}, batch {i + 1}/{num_batches}, train loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, ')
            print(f'Epoch {epoch + 1}/{num_epochs}, train loss{metric[0] / metric[2]:.3f}, '
                  f'train acc {metric[1] / metric[3]:.3f}, {timer.avg():.3f} examples/sec on {str(devices)}')
            # 写入当前批次的训练损失和准确度到文件
            with open(filename, 'a') as f:
                f.write(f'Epoch {epoch + 1}/{num_epochs}, train loss{metric[0] / metric[2]:.3f}, '
                      f'train acc {metric[1] / metric[3]:.3f}, {timer.avg():.3f} examples/sec on {str(devices)}\n')

        loss_epoch = metric[0] / metric[2]  # 更改了分母为metric[3]
        train_acc_epoch = metric[1] / metric[3]  # 修复了逻辑bug，并更改了分母为metric[3]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        train_loss_history.append(loss_epoch)
        train_acc_history.append(train_acc_epoch)
        test_acc_history.append(test_acc)

    print(f'loss {loss_epoch:.3f}, train acc {train_acc_epoch:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

    # 写入最终epoch的结果（与之前相同）
    with open(filename, 'a') as f:
        f.write(
            f'Epoch {num_epochs}, Final Train Loss {loss_epoch:.3f}, Final Train Acc {train_acc_epoch:.3f}, Test Acc {test_acc:.3f}\n')

    return train_loss_history, train_acc_history, test_acc_history



if __name__ == '__main__':
    batch_size = 256
    devices = d2l.try_all_gpus()
    net = d2l.resnet18(10, 3)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    def train_with_data_aug(train_augs, test_augs, net, lr = 0.001):
        train_iter = load_cifar10(True, train_augs, batch_size)
        test_iter = load_cifar10(False, test_augs, batch_size)
        loss = nn.CrossEntropyLoss(reduction='none')
        trainer = torch.optim.Adam(net.parameters(), lr=lr)
        train_loss_history, train_acc_history, test_acc_history = train(net, train_iter, test_iter, loss, trainer, 10, devices)

        plt.xlabel(xlabel='epoch')

        plt.plot(train_loss_history, label='train loss', )
        plt.plot(train_acc_history, label='train acc')
        plt.plot(test_acc_history, label='test acc')
        plt.title('augmentation + adam')
        plt.legend()
        plt.grid(True)
        plt.show()

    train_with_data_aug(train_augs, test_augs, net)




