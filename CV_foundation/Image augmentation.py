import numpy as np
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt

"""使用matplotlib显示图片"""
# plt.plot()
# img = Image.open('..//img//Kobe.jpg')
#
# # 将PIL图片转换为NumPy数组
# img_array = np.array(img)
#
# # 显示图片
# plt.imshow(img_array)
# plt.axis('off')  # 不显示坐标轴
# plt.show()

"""使用d2l显示图片"""
d2l.set_figsize()
img = Image.open('..//img//cute.jpg')
d2l.plt.imshow(img)  # 用d2l中的函数显示图片
d2l.plt.show()

"""图片增广"""
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    # 生成一个元素为图像的列表
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

# 对图像应用水平随机翻转
apply(img, aug=torchvision.transforms.RandomHorizontalFlip())

# 对图像应用垂直随机翻转
apply(img, aug=torchvision.transforms.RandomVerticalFlip())

# 创建一个随机大小裁剪并缩放的变换，裁剪后的图像大小为(200, 200)，缩放比例在0.1到1之间，长宽比在0.5到2之间
shape_aug = torchvision.transforms.RandomResizedCrop((150, 150), scale=(0.1, 1), ratio=(0.5, 2))

# 对图像应用颜色抖动变换，亮度抖动0.5，对比度、饱和度和色调不进行抖动
apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))

# 对图像应用颜色抖动变换，亮度不进行抖动，色调抖动0.5，对比度和饱和度不进行抖动
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))

# 创建一个颜色抖动变换对象，亮度、对比度、饱和度和色调分别抖动0.5
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=.5, saturation=0.5, hue=0.5)

# 对图像应用颜色抖动变换对象
apply(img, color_aug)

augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
