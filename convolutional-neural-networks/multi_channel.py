import torch
from d2l import torch as d2l
import numpy as np

def corr2d_multi_in(X, K):
    # 先遍历X和K的第0个维度（通道数），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print("X的维度：",X.shape)

print("两个输入通道：")
print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    # 迭代K的第0个维度，每次都对输入X执行互相关运算
    # 最后将所有的结果都叠加起来
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# print("叠加前K的样子：")
# print(K)
# print("叠加前K的维度：", K.shape)
K = torch.stack((K, K+1, K+2), 0)
# print("叠加后K'的样子：")
# print(K)
print("叠加后K的维度：", K.shape)
corr2d_mio = corr2d_multi_in_out(X, K)
print("两个输入通道和三输出通道：")
print(corr2d_mio)
print("*" * 100)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X_io1 = torch.normal(0,1,(3,3,3))
K_io1 = torch.normal(0,1,(2,3,1,1))

print("X_io1.shape:", X_io1.shape)
Y1 = corr2d_multi_in_out_1x1(X_io1, K_io1)
Y2 = corr2d_multi_in_out(X_io1, K_io1)

print("Y1:", Y1, "Y1.shape:", Y1.shape)
print("Y2:", Y2, "Y2.shape:", Y2.shape)

