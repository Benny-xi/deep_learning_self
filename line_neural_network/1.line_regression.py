import math
import time
import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

"""
矢量化加速，利用先行代码库，而不使用for循环（代价高）
"""
# n = 10000
# a = torch.ones([n])
# b = torch.ones([n])
#
# print("a:", a)
# print("a.shape=", a.shape)
# print("+++++++++++++++++++++++++++++++")
# print("b:", b)
# print("+++++++++++++++++++++++++++++++")
#
# class Timer:
#     """记录多次运行时间"""
#     def __init__(self):     # 用于初始化对象的属性
#         self.times = []
#         self.start()
#
#     def start(self):
#         """启动计时器"""
#         self.tik = time.time()
#
#     def stop(self):
#         """停止计时器并将时间记录在列表中"""
#         self.times.append(time.time() - self.tik)
#         return self.times[-1]
#
#     def avg(self):
#         """返回平均时间"""
#         return sum(self.times)/len(self.times)
#
#     def sum(self):
#         """返回时间总和"""
#         return sum(self.times)
#
#     def cumsum(self):
#         """返回累计时间"""
#         return np.array(self.times).cumsum().tolist()
#
# c = torch.zeros(n)
# timer = Timer()
# for i in range(n):
#     c[i] = a[i] +b[i]
# print(f'{timer.stop():.5f} sec')
#
#
# timer.start()
# d = a+b
# print(f'{timer.stop():.5f} sec')

"""正态分布和平方损失"""
def normal(x, mu, sigma):
    p = 1/math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x-mu)**2)

x = np.arange(-7, 7, 0.01)
# 均值和标准差对
params = [(0,1),(0,2),(3,1)]
# d2l.plot(x,[normal(x,mu,sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5,2.5),
#          legend = [f'mean {mu}, std {sigma}'for mu, sigma in params])

# 开始绘图
plt.figure(figsize=(4.5,2.5))
# 绘制每条曲线
for mu,sigma in params:
    y=normal(x,mu,sigma)
    plt.plot(x,y,label=f'mean{mu}, std{sigma}')
# 设置图例
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
# 添加网格线
plt.grid(True)
plt.grid(which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)  # 设置网格线样式

plt.show()