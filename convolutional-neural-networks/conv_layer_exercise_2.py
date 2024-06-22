import torch
from torch import nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 定义模型
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
        # return torch.nn.Conv2d(x, self.weight.unsqueeze(0).unsqueeze(0)) + self.bias

# 创建 模型实例
model = Conv2D(kernel_size=(2, 2))

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# 随机生成输入和目标数据
inputs = torch.randn(4,4)
targets = torch.randn(3,3)

# 开始训练循环
epochs = 100
for epoch in range(epochs):
    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 计算梯度
    optimizer.step() # 更新梯度

    if (epoch+1) % 10 == 0:
        print(f"Epoch[{epoch+1}/100], Loss:{loss.item():.4f}")