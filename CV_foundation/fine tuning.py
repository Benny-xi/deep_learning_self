import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 获取数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

"创建两个实例来分别读取训练和测试数据集中的所有图像"
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
# 打印训练集和测试集中的样本数量
print('train:', len(train_imgs), 'test:', len(test_imgs))

# 使用RGB通道的均值和标准差，标准化每个通道
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
                                             torchvision.transforms.RandomHorizontalFlip(),
                                             torchvision.transforms.ToTensor(),
                                             normalize])
test_augs = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(224),
                                            torchvision.transforms.ToTensor(),
                                            normalize])

# 定义和初始化模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.kaiming_normal_(finetune_net.fc.weight)


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


    timer, num_batches = d2l.Timer(), len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)  # 更改了Accumulator的参数以匹配add方法的调用
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())  # 确保metric的参数匹配
            timer.stop()
            print(f'Epoch {epoch + 1}/{num_epochs}, train loss{metric[0] / metric[2]:.3f}, '
                  f'train acc {metric[1] / metric[3]:.3f}, {timer.avg():.3f} examples/sec on {str(devices)}')


        loss_epoch = metric[0] / metric[2]  # 更改了分母为metric[3]
        train_acc_epoch = metric[1] / metric[3]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        print("************************************************")
        print(f'Epoch {epoch + 1}/{num_epochs}, train loss{loss_epoch:.3f}, '
              f'train acc {train_acc_epoch:.3f}, test_acc{test_acc:.3f}')
        print("************************************************")

        train_loss_history.append(loss_epoch)
        train_acc_history.append(train_acc_epoch)
        test_acc_history.append(test_acc)

    print(f'loss {loss_epoch:.3f}, train acc {train_acc_epoch:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

    # 写入最终epoch的结果（与之前相同）
    # with open(filename, 'a') as f:
    #     f.write(
    #         f'Epoch {num_epochs}, Final Train Loss {loss_epoch:.3f}, Final Train Acc {train_acc_epoch:.3f}, Test Acc {test_acc:.3f}\n')

    return train_loss_history, train_acc_history, test_acc_history


if __name__ == '__main__':
    # 定义训练函数，微调模型
    def train_fine_tuning(net, lr, batch_size, num_epochs, param_group=True):
        loss = nn.CrossEntropyLoss(reduction='none')
        train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform=train_augs), batch_size=batch_size, shuffle=True)
        test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'test'), transform=test_augs), batch_size=batch_size)
        devices = d2l.try_all_gpus()
        if param_group:
            params_1x = [param for name, param in net.named_parameters()
                         if name not in ['fc.weight', 'fc.bias']]
            trainer = torch.optim.SGD([{'params': params_1x},
                                       {'params': net.fc.parameters(),
                                        'lr': lr * 10}],
                                      lr=lr, weight_decay=0.001)
        else:
            trainer = torch.optim.SGD([{'params': net.parameters()}], lr=lr, weight_decay=0.001)

        train(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


    train_fine_tuning(finetune_net, lr=5e-5, batch_size=128, num_epochs=3)


