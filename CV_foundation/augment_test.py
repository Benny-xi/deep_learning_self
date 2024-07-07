
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)



def train_batch(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):
        # 微调BERT中所需
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

#@save
def train(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

            print(f'Epoch {epoch + 1}/{num_epochs}, train loss{metric[0] / metric[2]:.3f}, '
                  f'train acc {metric[1] / metric[3]:.3f}, {timer.avg():.3f} examples/sec on {str(devices)}')

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        print("***************************************************************")
        print(f'epoch {epoch + 1}, train loss{metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print("***************************************************************")


    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')



if __name__ == '__main__':
    batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

    net.apply(init_weights)

    def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
        train_iter = load_cifar10(True, train_augs, batch_size)
        test_iter = load_cifar10(False, test_augs, batch_size)
        loss = nn.CrossEntropyLoss(reduction="none")
        trainer = torch.optim.Adam(net.parameters(), lr=lr)
        train(net, train_iter, test_iter, loss, trainer, 10, devices)


    train_with_data_aug(train_augs, test_augs, net)