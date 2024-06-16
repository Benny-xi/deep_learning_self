import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print("net(X):", net(X))
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

print("net是什么：", net)
print("参数访问：", net[2].state_dict())

print("net模型第三层的偏置类型：",type(net[2].bias))
print("net模型第三层的偏置项的值：",net[2].bias)
print("net模型第三层的偏置项的数据：",net[2].bias.data)

print("一次性查看 net模型第三层的所有参数",*[(name, param.shape) for name, param in net[0].named_parameters()])
print("一次性查看 net模型的所有参数", *[(name, param.shape) for name, param in net.named_parameters()])