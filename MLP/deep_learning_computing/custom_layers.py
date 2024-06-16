import torch
import torch.nn.functional as F
from torch import nn

"""不带参数的层"""
class Centeredlayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer_param = Centeredlayer()
print("不带参数的层：",layer_param(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), Centeredlayer())

Y = net(torch.rand(4, 8))
print("net是什么：", net)
print("Y的形状：", Y.shape)
print("Y的均值：", Y.mean())

"""带参数的层"""
class MyLinear(nn.Module):
    def __init__(self, in_unit, out_unit):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_unit, out_unit))
        self.bias = nn.Parameter(torch.randn(out_unit,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print("linear 的权重：", linear.weight)

""" 使用自定义层进行前向传播"""
print("自定义带参数层进行前向传播：", linear(torch.rand(2,5)))