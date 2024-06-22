import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i+p_h, j: j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
pool2x2 = pool2d(X,(2, 2))
print("2*2最大池化：")
print(pool2x2)
print("2*2平均池化：")
print(pool2d(X, (2, 2), 'avg'))
print("*填充和步幅*" * 10)

X = torch.arange(16, dtype=torch.float32).reshape((1,1,4,4))
print("单通道单样本数的4*4样本X：")
print(X)

pool2d_2 = nn.MaxPool2d(3)
print(pool2d_2(X))
