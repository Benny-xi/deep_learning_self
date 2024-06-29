import os
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils import data

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

train_path = '../data/classify-leaves/train.csv'
img_dir = '../data/classify-leaves'

train_csv = pd.read_csv(train_path) # pandas 库提供了许多方法来查看和操作 DataFrame 数据

"""使用各种方法查看和操作 DataFrame"""
"""
print(train_csv.info())     # 显示 DataFrame 的简要摘要，包括索引类型、列数、非空值数以及每列的内存使用情况。
print(train_csv.describe())     # 生成描述性统计信息，包括计数、均值、标准差、最小值、四分位数和最大值。
print(train_csv.tail(5))        # 返回 DataFrame 的最后 n 行。默认返回最后五行。
print(train_csv.shape)       # 返回 DataFrame 的维度（行数和列数）。
print(train_csv.columns)       # 返回 DataFrame 的列标签。
print(train_csv.dtypes)      # 返回每列的数据类型。
print(train_csv.head(10))       # 返回 DataFrame 的前 n 行。默认返回前五行。
print(train_csv.sample(5))    # 随机抽样 DataFrame 的 n 行。
print(train_csv.isnull())     # 返回一个布尔值的 DataFrame，指示每个单元格是否包含缺失值。
print(train_csv.sum())      # 对 DataFrame 或某一列进行求和。
print(train_csv.mean())
print(train_csv.median())   # 计算 DataFrame 或某一列的中位数。
print(train_csv.min())
print(train_csv.max())
print(train_csv['label'].value_counts())    # 计算某一列中每个唯一值的出现次数。
print(len(train_csv))
"""

" 预览前面八张图片"
# for i in range (8):
#     image = Image.open(os.path.join(img_dir, train_csv.iloc[i, 0]))
#     plt.subplot(2, 4, i+1)
#     plt.imshow(image)
#     plt.xticks([]), plt.yticks([])    # 隐藏坐标轴
#     plt.title(train_csv.iloc[i, 1], fontsize='small')
# plt.show()

" 查看图像尺寸"
# image = Image.open(os.path.join(img_dir, train_csv.iloc[0, 0]))
# print("图像尺寸：", image.size)

"生成 label 和 类别索引 的映射"
labels = sorted(list(set(train_csv['label'])))  # 获取唯一标签并排序
len_label = len(labels)     # 计算标签数量
labels_cls_map = dict(zip(labels, range(len_label)))    # 创建 label 到类别索引的映射
cls_labels_map = {v : k for k, v in labels_cls_map.items()}     # 创建 类别索引 到 标签 的映射

"定义获取dataloader的使用进程数的方法"
def get_dataloader_workers():
    """使用两个进程来读取数据"""
    return 2

"""定义获取GPU的方法"""
def try_gpu(i=0):
    """如果存在，则返回GPU(i),否则返回CPU()"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

"""自定义类Dataset"""
class LeavesImageDataset(Dataset):
    def __init__(self, X, y, root_dir: str, transform=None):
        self.X = X
        self.y = y
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.X[idx])
        image = Image.open(img_path)
        label = labels_cls_map[self.y[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

class LeavesImageTestDataset(Dataset):
    def __init__(self, annotations_file:str, root_dir: str, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

"""定义网络"""
def get_net(num_cls, train = True):
    if train:
        # net = torchvision.models.resnet50(weights= torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net = torchvision.models.resnet18(weights= torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        net = torchvision.models.resnet18()
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_cls)
    return net

"定义训练方法"
def train_model(net, dataset_sizes, dataloaders, num_epochs, lr, device, weight_decay):
    print("training on ", device)
    print("train_nums:",dataset_sizes['train'])
    print("val_nums:",dataset_sizes['val'])

    train_loss_history = []   # 创建损失历史记录表
    train_acc_history = []   # 创建精确度历史记录表
    val_acc_history = []
    val_loss_history = []

    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_corrects = 0

            start_time = time.time()
            for X, y in tqdm(dataloaders[phase]):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                with torch.set_grad_enabled(phase == 'train'):  # 根据当前的执行阶段（训练或评估）有条件地启用或禁用自动求导机制
                    y_hat = net(X)
                    _, preds = torch.max(y_hat, 1)
                    l = loss(y_hat, y)

                    if phase == 'train':
                        l.backward()
                        optimizer.step()

                running_loss += l.item() * X.size(0)
                running_corrects += torch.sum(preds == y.data)
                print(f'running_loss:{running_loss}, running_corrects:{running_corrects}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            elapsed_time = time.time() - start_time
            print(f'{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} elapsed_time: '
                  f'{elapsed_time:.4f}秒', end='\n')
        # 每五轮保存一次模型
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, 'resnet_18_' + str(epoch) + '.pth')
            torch.save(net.state_dict(), checkpoint_path)
        print('-' * 10)
    return net, train_loss_history, train_acc_history, val_loss_history, val_acc_history

"""定义评估方法"""
def val(model, device):
    test_dataloader = dataloaders['train']
    corrects = 0
    total = len(data_set['train'])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            _,preds = torch.max(outputs, dim=1)
            corrects += torch.sum(preds == labels.data)
    print("Test ACC : {:.4f}".format(corrects.double() / total))

"定义预测方法"
def pred(net, test_path, img_dir, batch_size, device, output):
    predict = torch.tensor([]).to(device)
    sub_dataset = LeavesImageTestDataset(test_path, img_dir, data_transforms['val'])
    sub_iter = data.DataLoader(dataset=sub_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=get_dataloader_workers(),pin_memory=False)

    net.eval()
    net = net.to(device)
    for X in tqdm(sub_iter):
        with torch.no_grad():
            X = X.to(device)
            y_hat = net(X)
            y_hat = F.softmax(y_hat, dim=1)
            values, preds = torch.max(y_hat, dim=1)
            predict = torch.cat((predict, preds), dim=0)
    predict = predict.detach().cpu().numpy()
    predict_label = []
    for i in range(predict.shape[0]):
        predict_label.append(cls_labels_map[int(predict[i])])
    submission = pd.read_csv(test_path)
    submission['label'] = pd.Series(predict_label)
    sub_path = os.path.join(output, 'submission.csv')
    submission.to_csv(sub_path, index=False)

"划分训练集和验证集"
X = train_csv.iloc[:, 0]
Y = train_csv.iloc[:, 1]
X_train, X_val, Y_train, Y_val = train_test_split(X.values, Y.values, test_size=0.25, random_state=123)

"定义训练数据"
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_set = {
    'train': LeavesImageDataset(X_train, Y_train, img_dir, data_transforms['train']),
    'val': LeavesImageDataset(X_val, Y_val, img_dir, data_transforms['val']),
}
dataset_sizes = {
    'train': len(data_set['train']),
    'val': len(data_set['val'])
}

"""run"""
if __name__ == '__main__':
    lr, num_epochs, batch_size, weight_decay = 0.01, 30, 64, 1e-3

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataloaders = {
        'train': data.DataLoader(dataset = data_set['train'], batch_size=batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers(), pin_memory=True),
        'val': data.DataLoader(dataset = data_set['val'], batch_size=batch_size, shuffle=False,
                               num_workers=get_dataloader_workers(), pin_memory=False)
    }
# 训练模型
    model, train_loss_history, train_acc_history, val_loss_history, val_acc_history = (
        train_model(get_net(len(labels_cls_map)), dataset_sizes, dataloaders, num_epochs, lr,try_gpu(), weight_decay))
# 评估模型
    net = get_net(len_label, train = False)
    checkpoint_path = os.path.join(checkpoint_dir, 'resnet_18_20.pth')
    state_dict = torch.load(checkpoint_path)
    net.load_state_dict(state_dict)
    val(net, try_gpu())

# 画图
    plt.xlabel(xlabel='epoch')

    train_loss_history_cpu = [train_loss.cpu().numpy() if torch.is_tensor(train_loss) else train_loss for train_loss in train_loss_history]
    train_acc_history_cpu = [train_acc.cpu().numpy() if torch.is_tensor(train_acc) else train_acc for train_acc in train_acc_history]
    val_loss_history_cpu = [val_loss.cpu().numpy() if torch.is_tensor(val_loss) else val_loss for val_loss in val_loss_history]
    val_acc_history_cpu = [val_acc.cpu().numpy() if torch.is_tensor(val_acc) else val_acc for val_acc in val_acc_history]

    plt.plot(train_loss_history_cpu, label='train loss', linestyle='-')
    plt.plot(train_acc_history_cpu, label='train acc', linestyle='--')
    plt.plot(val_loss_history_cpu, label='val loss', linestyle='-')
    plt.plot(val_acc_history_cpu, label='val acc', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()