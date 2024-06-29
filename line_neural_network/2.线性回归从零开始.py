import torch
import random
from d2l import torch as d2l
import matplotlib.pyplot as plt

def synthetic_data(w,b, num_examples):
    """生成 y = Xw+b +噪声 """
    X = torch.normal(0,1,(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
print("true_w:", true_w)
print(f'true_w.shape:{true_w.shape}')
print("len(w):", len(true_w))
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:',labels[0])

# # d2l.set_figsize()
# # d2l.plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(), 1)
#
# # 设置图形大小
# plt.figure(figsize=(8,6))
# # 绘制散点图
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=10)
#
# plt.show()

# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本随机读取，没有特定顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size=10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 初始化模型参数
w = torch.normal(0,0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print("w=", w)
print("b=", b)

# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# train
lr = 0.03
num_epochs = 5
model = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l= loss(model(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(model(features, w, b), labels)
        print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b-b}')