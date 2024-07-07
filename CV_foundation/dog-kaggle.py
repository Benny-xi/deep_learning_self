import os
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l



def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224x224的新图像
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 添加随机噪声
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])





def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

def plot_result(train_loss_history,valid_loss_history):
    plt.xlabel(xlabel='epoch')

    plt.plot(train_loss_history, label='train loss', )
    plt.plot(valid_loss_history, label='valid_loss')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    data_dir = os.path.join('..', 'data', 'dog')
    batch_size = 128
    valid_ratio = 0.1
    reorg_dog_data(data_dir, valid_ratio)

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]


    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)]

    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                             drop_last=True)

    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                            drop_last=False)

    loss = nn.CrossEntropyLoss(reduction='none')


    def evaluate_loss(data_iter, net, devices):
        l_sum, n = 0.0, 0
        for features, labels in data_iter:
            features, labels = features.to(devices[0]), labels.to(devices[0])
            outputs = net(features)
            l = loss(outputs, labels)
            l_sum += l.sum()
            n += labels.numel()
        return (l_sum / n).to('cpu')


    def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
              lr_decay):
        # 只训练小型自定义输出网络

        train_loss_history = []  # 创建损失历史记录表
        valid_loss_history = []
        test_acc_history = []

        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
        trainer = torch.optim.SGD((param for param in net.parameters()
                                   if param.requires_grad), lr=lr,
                                  momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
        num_batches, timer = len(train_iter), d2l.Timer()
        legend = ['train loss']
        if valid_iter is not None:
            legend.append('valid loss')

        for epoch in range(num_epochs):
            metric = d2l.Accumulator(2)
            for i, (features, labels) in enumerate(train_iter):
                timer.start()
                features, labels = features.to(devices[0]), labels.to(devices[0])
                trainer.zero_grad()
                output = net(features)
                l = loss(output, labels).sum()
                l.backward()
                trainer.step()
                metric.add(l, labels.shape[0])
                timer.stop()
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    print(f'Epoch {epoch + 1}/{num_epochs}, train loss {metric[0] / metric[1]:.3f}, ')

            measures = f'train loss {metric[0] / metric[1]:.3f}'
            train_loss = metric[0] / metric[1]
            train_loss_history.append(train_loss)
            if valid_iter is not None:
                valid_loss = evaluate_loss(valid_iter, net, devices)
                print(f'Epoch {epoch + 1}/{num_epochs}, valid loss {valid_loss}')

            valid_loss_history.append(valid_loss)
            scheduler.step()
        if valid_iter is not None:
            measures += f', valid loss {valid_loss:.3f}'
        print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
                         f' examples/sec on {str(devices)}')

        return train_loss_history, valid_loss_history


    devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
    lr_period, lr_decay, net = 2, 0.9, get_net(devices)
    train_loss_history, valid_loss_history = train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay)

    plot_result(train_loss_history, valid_loss_history)




