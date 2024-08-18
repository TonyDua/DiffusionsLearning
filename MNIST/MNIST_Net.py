"""
教程来源： 《从零设计并训练一个神经网络，你就能真正理解它了》 https://www.bilibili.com/video/BV134421U77t/
如何设计、实现并训练一个标准的前馈神经网络，用于手写数字图像的分类，重点讲解了神经网络的设计和实现、数据的准备和处理、模型的训练和测试流程

图像大小：28 * 28
数据点 ：28*28=784
输入层： 1x783

三层网络
输入层：784个神经元
隐藏层：256个神经元
输入层和隐藏层连接的线性层：784*256
输出层：10个神经元
隐藏层和输出层连接的线性层：256*10

"""

import torch
import torchvision

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms


# 定义神经网络Network
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        # 线性层1 输入层和隐藏层之间连接的线性层
        self.layer1 = nn.Linear(784, 256)
        # 线性层2 隐藏层和输出层之间连接的线性层
        self.layer2 = nn.Linear(256, 10)

    # 向前传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        网络的前向传播函数
        Note:forward方法 必须写类型注解！
        :param x: torch.Tensor
        :return: torch.Tensor
        """

        x = x.view(-1, 28 * 28)  # 使用view函数 将x展平
        x = self.layer1(x)  # 将x输入到layer1层
        x = torch.relu(x)  # 使用relu进行激活函数 通常写为 F.relu
        return self.layer2(x)  # 输入layer2层返回结果


def print_data_loader(loader: DataLoader):
    for batch_idx, (data, label) in enumerate(loader):
        if batch_idx == 3:
            print(f'batch_idx:{batch_idx}')
            print(f'data.shape:{data.shape}')
            print(f'label:{label.shape}')
            print(label)


def show_mnist_images(loader: DataLoader):
    # 实现单张图片可视化

    images, labels = next(iter(loader))
    img = torchvision.utils.make_grid(images)

    img = img.numpy().transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    print(labels)
    plt.imshow(img)
    plt.show()


def train(loader: DataLoader,epochs: int = 10):

    model = Network()  # 1.模型实例
    optimizer = optim.Adam(model.parameters())  # 2.优化器：优化模型中的参数
    criterion = nn.CrossEntropyLoss()  # 3.损失函数：分类问题，使用交叉熵损失误差

    # 迭代循环训练
    for epoch in range(epochs): # 整个训练集的循环训练次数

        # 内层每循环一次，执行一次梯度下降算法（5步骤）
        for batch_idx, (data, label) in enumerate(loader):

            output = model(data) # 1.计算神经网络前向传播结果
            loss = criterion(output, label) # 2. 计算output和label之间的损失Loss
            loss.backward() # 3.使用backward计算梯度
            optimizer.step() # 4.使用optimizer.step更新参数
            optimizer.zero_grad() # 5.将梯度清零

            # 每迭代100次，输出一次模型的损失，观察训练结果
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(loader)} | Loss:{loss.item():.4f}")

    torch.save(model.state_dict(), f'mnist_{epochs}.pth')


def eval_model(loader: DataLoader,epochs: int = 10):

    model = Network()
    model_file = f'mnist_{epochs}.pth'
    model.load_state_dict(torch.load(model_file))
    model.eval()

    right = 0
    total = 0

    for i, (x, y) in enumerate(loader):
        output = model(x)
        predict = output.argmax(1) # 选择概率最大的标签计算预测结果

        '''
        对于 batch_size=1，可以直接使用 predict = output.argmax(1).item()
        total += 1
        if predict == y.item():  # 对于 batch_size=1，y 也是一个标量
            right += 1
        '''

        for p, t in zip(predict, y):
            total += 1
            if p.item() == t.item():
                right += 1
            else:
                print(f'wrong case: predict = {p.item()} | y = {t.item()} ')
    # 计算测试效果
    acc = right / total
    print(f'{model_file} test accuracy: {right} / {total} ={acc:.3f}')

    f'''
    对于 batch_size=1，计算测试效果:
        sample_num = len(loader) * 64
        acc = right * 1.0 / sample_num
        print(f' {model_file} test accuracy:  %d / %d = %.3lf' % (right, sample_num, acc))
    '''


if __name__ == '__main__':

    # 图像预处理
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()]
    )

    # 加载训练数据集
    train_dataset = torchvision.datasets.MNIST(root="mnist/",
                                               train=True,
                                               download=True,
                                               transform=transform
                                               )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 加载测试数据集
    test_dataset = torchvision.datasets.MNIST(root="mnist/",
                                              train=False,
                                              download=True,
                                              transform=transform
                                              )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # print(len(train_dataloader),len(test_dataloader))
    # print_data_loader(train_dataloader)
    # show_mnist_images(train_dataloader)
    # train(train_dataloader,epochs=20)
    eval_model(test_dataloader,epochs=10)
