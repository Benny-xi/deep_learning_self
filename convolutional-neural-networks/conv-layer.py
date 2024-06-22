import torch
from torch import nn
from d2l import torch as d2l


"""计算二维互相关运算"""
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print("二维互相关运算输出：", corr2d(X, K))

"""定义卷积层"""
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones((6,8))
print("给定图像X的形状：", X.shape)
X[:, 2:6] = 0
print("给定图像X是：", X)

"""构造一个高度为1，宽度为2的卷积核k"""
K = torch.tensor([[1.0, -1.0]])
print("卷积核K的形状：", K.shape)
print("卷积核K为：", K)

"""将输入图像X和卷积核K进行互相关运算"""
Y = corr2d(X, K)
print("卷积操作后的输出：", Y)

# 构造一个二维卷积层，它具有一个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小，通道，高度，宽度）
# 其中批量大小和通道数都是1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    print(f"epoch {i+1}, loss {l.sum():.5f}")


# 学习到的权重张量
print("学习到的权重：", conv2d.weight.data.reshape((1, 2)))