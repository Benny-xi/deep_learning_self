import torch
from torch import nn
from torch.nn import functional as F


X = torch.rand(2,20)

""" 层与块 """
# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# print("net", net)
# print("net(X):", net(X))

"""自定义块"""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
print("自定义快")
print("net(X):", net(X))


"""顺序快"""
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):

        for block in self._modules.values():
            X = block(X)
        return X

net_shunxu = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print("顺序块")
print("net_顺序：", net_shunxu(X))

"""前向传播中执行代码"""
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
net_qianxiang = FixedHiddenMLP()
print("前向传播中执行代码")
print("net_前向: ", net_qianxiang(X))

