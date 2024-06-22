import torch
from torch import nn

"""模仿书本"""
# # 创建一个对角线边缘的矩阵 X
# X = torch.diag(torch.ones(5))
# print("对角线边缘图像：", X)
#
# # 创建卷积核
# K = torch.tensor([[1.0, -1.0]])
#
# # 定义卷积操作
# def corr2d(X, K):
#     h, w = K.shape
#     Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
#     return Y
#
# # 创建卷积层
# class Conv2D(nn.Module):
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.weight = nn.Parameter(torch.rand(kernel_size))
#         self.bias = nn.Parameter(torch.zeros(1))
#     def forward(self, x):
#         return corr2d(x, self.weight) + self.bias
#
# Y = corr2d(X, K)
# print("卷积后输出：", Y)

import torch
import torch.nn.functional as F

# 创建具有对角线边缘的图像 X
# X = torch.tensor([
#     [1.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 1.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 1.0]
# ])

# 给定的卷积核 K

X = torch.diag(torch.ones(5))
print("创建对角线边缘图像：")
print(X)

# K = torch.tensor([
#     [0.0, 1.0],
#     [2.0, 3.0]
# ])

# Reshape X and K to fit the Conv2d input format

K = torch.tensor([[1.0, -1.0]])
print("卷积核K的形状：", K.shape)
print("卷积核K为：", K)

X = X.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 5, 5)
K = K.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 2, 2)

# Apply convolution
output = F.conv2d(X, K, stride=1, padding=0)
output = output.squeeze(0).squeeze(0)  # shape: (4, 4)
print("原始的卷积输出:")
print(output)

# 转置图像 X
X_transposed = X.squeeze(0).squeeze(0).T  # 转置图像
X_transposed = X_transposed.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 5, 5)

# 转置X后应用卷积
output_X_transposed = F.conv2d(X_transposed, K, stride=1, padding=0)
output_X_transposed = output_X_transposed.squeeze(0).squeeze(0)  # shape: (4, 4)
print("转置X后的输出：")
print(output_X_transposed)

# 转置卷积核 K
K_transposed = K.squeeze(0).squeeze(0).T  # 转置卷积核
K_transposed = K_transposed.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 2, 2)

# 转置K后进行卷积
output_K_transposed = F.conv2d(X, K_transposed, stride=1, padding=0)
output_K_transposed = output_K_transposed.squeeze(0).squeeze(0)  # shape: (4, 4)
print("转置K后的输出：")
print(output_K_transposed)
