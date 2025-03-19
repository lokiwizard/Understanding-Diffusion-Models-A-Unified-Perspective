# 载入minist数据集

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据预处理方法
transform = transforms.Compose([
    transforms.Resize((32, 32)),            # 将图像缩放到 32x32
    transforms.ToTensor(),          # 将图像转换为 Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

def load_data():
    # 下载并加载训练集和测试集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    trainX, testX = load_data()
    for data, target in trainX:
        print(data.shape, target.shape)
        break






