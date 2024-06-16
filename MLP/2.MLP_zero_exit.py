import torch
from torch import nn
from d2l import torch as d2l
from PIL import Image
import random

# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 模型
def net(X):
    """自定义模型"""
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)   # @ 表示矩阵乘法
    return (H @ W2 + b2)

"""简洁实现，通过nn.Sequential"""
# net = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(786, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )
#
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)
# net.apply(init_weights)



# 准确率计算
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 评估准确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数， 预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X),y), y.numel())   # accuracy(net(X), y)的形状是什么？？？
    return metric[0] / metric[1]

class Accumulator:
    """在n个变量上叠加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data) # 列表乘法操作符 * 可以用来重复列表中的元素
    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:   # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch 内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        train_loss, train_acc = train_metrics
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch:{epoch}',f'train_loss:{train_loss:.6f}',f'train_acc:{train_acc:.6f}',f'test_acc:{test_acc:.6f}')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if isinstance(img, torch.Tensor):
            # Tensor Image
            ax.imshow(img.numpy())
        elif isinstance(img, Image.Image):
            # PIL Image
            ax.imshow(img)
        else:
            raise TypeError("Unsupported image type. Must be a torch.Tensor or PIL.Image.Image.")

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.tight_layout()
    d2l.plt.show()

# 预测一下
def predict_ch3(net, test_iter, n=6):
    # for X, y in test_iter:
    #     break
    # trues = d2l.get_fashion_mnist_labels(y)
    # preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    # titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n,
    #                 titles=titles[0:n])

    """从 test_iter 中随机选择 n 个样本的数据和标签"""
    samples = random.sample(list(test_iter), n)
    for X, y in samples:
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == '__main__':
    num_epochs = 15
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    """
    torch.randn 是 PyTorch 中用于生成服从标准正态分布（均值为0，标准差为1）
    的随机张量的函数。这个函数在深度学习中经常用于初始化权重，因为标准正态分布的随机数可以帮助权重在训练开始时保持适当的规模。
    """
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    """
    reduction='none'：返回一个张量，每个元素表示对应样本的损失值。
    reduction='mean'：返回一个标量，表示所有样本损失值的平均值。
    reduction='sum'：返回一个标量，表示所有样本损失值的总和。
    """

    # 优化器
    updater = torch.optim.SGD(params, lr=0.1)

    train_ch3(net, train_iter,test_iter,loss,num_epochs, updater)

    predict_ch3(net, test_iter)

