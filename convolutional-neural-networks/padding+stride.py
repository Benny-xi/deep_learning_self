import torch
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1)
X = torch.rand(size=(8, 8))
print("初始X是：")
print(X)
P_conv = comp_conv2d(conv2d, X).shape
# print("padding后经过卷积的尺寸：")
# print(P_conv)
