'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
txue4133@uni.sydney.edu.au, weiyu.ju@sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt

from network import * # the network you used

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()

# training process. 
def train_net(net, trainloader, valloader):
########## ToDo: Your codes goes below #######
    # 设置模型为训练模式，这将启用某些层（如 Dropout）的训练特定行为
    net.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learningrate = 0.01
    epoch = 80

    # 定义优化器和学习率调度器
    optimizer = optim.SGD(net.parameters(), lr=learningrate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=18, gamma=0.6)
    # 初始化训练和测试结果列表
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    val_acc = 0

    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(1,epoch + 1):
        correct = 0  # 用于记录正确预测的样本数
        train_loss = 0  # 用于累计每个批次的损失
        # 遍历每个批次的数据和标签
        for batch_idx, (data, target) in enumerate(trainloader):
            # 将数据和标签移动到指定的设备上（例如 GPU）
            data, target = data.to(device), target.to(device)
            # 将优化器的梯度清零，以避免之前迭代的梯度累加
            optimizer.zero_grad()
            # 前向传播：通过模型传递数据并获得输出
            output = net(data)
            # 计算损失
            loss = loss_func(output, target)
            # 反向传播：计算相对于损失的梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 获取模型预测，找到每个样本的预测类别
            pred = output.max(1, keepdim=True)[1]
            # 比较预测类别与真实类别，统计正确预测的样本数
            correct += pred.eq(target.view_as(pred)).sum().item()
            # 累加每个批次的损失
            train_loss += loss.item()

        # 计算整个训练集的准确率和平均损失
        train_acc = correct / len(trainloader.dataset)
        train_loss /= len(trainloader.dataset)

        # 打印当前 epoch 的训练准确率和损失
        print('Train Epoch = {}, acc.={}, loss={}'.format(epoch, train_acc, train_loss))
        # 保存模型的当前状态
        torch.save(net.state_dict(), 'modified{}.pth'.format(learningrate))

        # 设置模型为评估模式（不计算梯度）
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in valloader:
                data, target = data.to(device), target.to(device)
                output = net(data)
                test_loss += loss_func(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(valloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(valloader.dataset),
            100. * correct / len(valloader.dataset)))
        test_acc = correct / len(valloader.dataset)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if test_acc>val_acc:
            val_acc = test_acc

        # 调整学习率
        scheduler.step()



    # 绘制训练和测试结果的曲线图
    plt.subplot(2, 1, 1)
    plt.plot(range(0, epoch), train_loss_list, label='train loss')
    plt.plot(range(0, epoch), test_loss_list, label='test loss')
    plt.legend()
    plt.title('loss vs. epoches')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(0, epoch), train_acc_list, label='train accuracy')
    plt.plot(range(0, epoch), test_acc_list, label='test accuracy')
    plt.legend()
    plt.xlabel('accuracy vs. epoches')
    plt.ylabel('Accuracy')

    plt.savefig("./accuracy_loss-max{}-lr{}.jpg".format(val_acc,learningrate))
    plt.show()

    return val_acc



##############################################
if __name__ == "__main__":
    ############################################
    # Transformation definition
    # NOTE:
    # Write the train_transform here. We recommend you use
    # Normalization, RandomCrop and any other transform you think is useful.

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(20), # 随机旋转图像，在[-10, 10]度范围内旋转
        transforms.CenterCrop(200),  # 中心裁剪
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3), # 随机改变图像的亮度、对比度、饱和度和色相
        transforms.Resize(256),
        # transforms.RandomCrop(224),
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ####################################

    ####################################
    # Define the training dataset and dataloader.
    # You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

    train_image_path = './train/'
    validation_image_path = './test/'

    trainset = ImageFolder(train_image_path, train_transform)
    valset = ImageFolder(validation_image_path, train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                             shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                             shuffle=True, num_workers=2)
    ####################################

    # ==================================
    # use cuda if called with '--cuda'.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = AlexNet().to(device)

    #network.load_state_dict(torch.load('modified.pth'))
    if args.cuda:
        network = network.cuda()

    # train and eval your trained network
    # you have to define your own
    val_acc = train_net(network, trainloader, valloader)

    print("final validation accuracy:", val_acc)

    # ==================================
