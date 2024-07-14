import torch
from torch import nn
from d2l import torch as d2l

def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h -1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
Y = trans_conv(X, K)
print(Y)

X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, padding=1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))
print(tconv(X).shape)


print("***************************************************9")

X = torch.arange(9.0).reshape(3,3)
print("X:\n", X)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("K:\n", K)
Y = d2l.corr2d(X, K)
print("Y:\n", Y)

print("***************************************************9")

def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
print("W:\n", W)

Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2)
print("Y:\n", Y)
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))

Z = trans_conv(Y, K)
print("Z:\n", Z)
Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))