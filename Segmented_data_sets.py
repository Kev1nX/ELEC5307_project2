# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
import os
import shutil

# 定义压缩文件和解压目录
zip_file_path = "./2023_ELEC5307_P2Train.zip"
extracted_folder = "./"

# 如果解压目录不存在，则创建
os.makedirs(extracted_folder, exist_ok=True)

# 解压缩文件
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)


# 定义数据集目录
dataset_dir = "./"
dataset_dir1 = "./2023_ELEC5307_P2Train"

# 获取数据集目录下的所有类别（子文件夹）
categories = [f for f in os.listdir(dataset_dir1) if os.path.isdir(os.path.join(dataset_dir1, f))]


# 创建训练和测试数据集目录
os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "test"), exist_ok=True)


# 遍历所有类别
for category in categories:
    category_dir = os.path.join(dataset_dir1, category)
    image_files = os.listdir(category_dir)

    # 划分训练和测试数据集
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # 创建类别对应的训练目录，并移动训练文件到该目录
    train_category_dir = os.path.join(dataset_dir, "train", category)
    os.makedirs(train_category_dir, exist_ok=True)
    for file in train_files:
        src = os.path.join(category_dir, file)
        dst = os.path.join(train_category_dir, file)
        if not os.path.exists(dst):  # 检查目标路径是否已存在文件
            os.rename(src, dst)

    # 创建类别对应的测试目录，并移动测试文件到该目录
    test_category_dir = os.path.join(dataset_dir, "test", category)
    os.makedirs(test_category_dir, exist_ok=True)
    for file in test_files:
        src = os.path.join(category_dir, file)
        dst = os.path.join(test_category_dir, file)
        if not os.path.exists(dst):  # 检查目标路径是否已存在文件
            os.rename(src, dst)

    # 删除多余文件夹
    # for category in categories:
    #     category_dir = os.path.join(dataset_dir, category)
    #     if os.path.isdir(category_dir):
    #         shutil.rmtree(category_dir)

# 删除多余文件夹
shutil.rmtree("./2023_ELEC5307_P2Train")

# 定义图像预处理操作：将图像转换为 PyTorch 张量
train_transform = transforms.Compose([transforms.ToTensor()])
val_transform = transforms.Compose([transforms.ToTensor()])

# 使用 ImageFolder 加载训练和测试数据集，并应用预处理操作
TrainSet = ImageFolder('./train', transform=train_transform)
ValSet = ImageFolder('./test', transform=val_transform)
