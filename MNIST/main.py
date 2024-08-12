from operator import itruediv

import matplotlib.pyplot as plt
import torch
import torchvision
from IPython.core.pylabtools import figsize
from bokeh.plotting import figure
from pandas import pivot
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 👇 使用 MNIST 来进行测试：下载训练用的数据集并给它配置一个数据加载器，然后把内容绘制出来，最后检查数据加载器中样本的形状与标签是否正确

dataset = torchvision.datasets.MNIST(root="mnist/",
                                     train=True,
                                     download=True,
                                     transform=torchvision.transforms.ToTensor()
                                     )
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
plt.show()

# 👆 MNIST 数据集中的每张图都是一个阿拉伯数字的28x28像素的灰度图，每个像素的取值区间是[0,1]

'''
3.2 扩散模型之退化过程
    如何为内容加入噪声呢？引入一个参数来控制输入的噪声量，例如：
        noise = torch.rand_like(x)
        noisy_x = (1-amount)*x + amount*noise
    如果amount=0,则返回输入，不做任何更改；如果amount=1，我们会得到一个纯粹的噪声。所以可以通过这种方式将内容和噪声进行混合，并将结果保持在相同
    的范围内（0-1）。但是要注意张量的形状，以免受到广播机制的不正确影响，下面写一个方法表示
'''


def corrupt(x, amount):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    根据amount为输入x加入噪声，这就是退化过程
    :param x: 输入 Torch.tensor
    :param amount:噪声量
    :return: 退化结果
    """
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1) # 整理形状以保证广播机制不出错
    return x*(1-amount) + noise*amount

# 开始测试这个方法
# 绘制输入数据
fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')


# 加入噪声
amount = torch.linspace(0, 1, x.shape[0]) # 从0到1退化更强烈了
noise_x = corrupt(x, amount)

# 绘制加噪版本的图像
axs[1].set_title('Corrupt data (-- amount increases -->)')
axs[1].imshow(torchvision.utils.make_grid(noise_x)[0], cmap='Greys')
plt.show()

