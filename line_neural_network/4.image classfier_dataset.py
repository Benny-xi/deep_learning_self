import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

##  基础讲解
# """读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换程32位浮点数格式，并除以255使得所有像素的数值均在0-1
# """
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(
#     root='./data', train=True, transform=trans, download=True
# )
# mnist_test = torchvision.datasets.FashionMNIST(
#     root='./data', train=False, transform=trans, download=True
# )
#
# print("len(mnist_train):", len(mnist_train))
# print("len(mnist_test):", len(mnist_test))
# print("================================================")
#
# def get_fashion_mnist_labels(labels):
#     """返回Fashion-MNIST数据集的文本标签"""
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
# """可视化样本"""
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
#     """绘制图像列表"""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             ax.imshow(img.numpy())
#         else:
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     d2l.plt.show()
#     return axes
#
# train_loader = data.DataLoader(mnist_train, batch_size=18, shuffle=True)
# X, y = next(iter(train_loader))
# show_images(X.reshape(18,28,28), 2, 9, titles=get_fashion_mnist_labels(y))



batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break