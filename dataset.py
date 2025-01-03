import os
import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler

# 数据集路径
train_dir = r'C:\Users\19330556651\Desktop\final task\train'
test_dir = r'C:\Users\19330556651\Desktop\final task\test'
val_dir = r'C:\Users\19330556651\Desktop\final task\val'

# 定义图像转换器
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.RandomRotation(10),  # 随机旋转图像（±10度）
    transforms.ToTensor(),  # 将图像转换为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化图像像素值
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224像素
    transforms.ToTensor(),  # 将图像转换为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化图像像素值
])

# 加载数据集并处理类别不平衡
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_test)

# 计算每个类别的样本数量
train_class_counts = torch.tensor([0] * len(train_dataset.classes))
for _, label in train_dataset:
    train_class_counts[label] += 1

test_class_counts = torch.tensor([0] * len(test_dataset.classes))
for _, label in test_dataset:
    test_class_counts[label] += 1

val_class_counts = torch.tensor([0] * len(val_dataset.classes))
for _, label in val_dataset:
    val_class_counts[label] += 1

# 计算每个类别的权重
train_class_weights = 1.0 / train_class_counts.double()
train_sample_weights = [train_class_weights[label] for _, label in train_dataset]

# 创建加权随机采样器
sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)

# 创建数据加载器，使用加权随机采样器
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 输出训练集信息
if __name__ == '__main__':
    print("Training dataset:")
    print(f"Total images: {len(train_dataset)}")
    for i, class_name in enumerate(train_dataset.classes):
        print(f"    {class_name}: {train_class_counts[i]} images")

    # 输出测试集信息
    print("\nTesting dataset:")
    print(f"Total images: {len(test_dataset)}")
    for i, class_name in enumerate(test_dataset.classes):
        print(f"    {class_name}: {test_class_counts[i]} images")

    # 输出验证集信息
    print("\nValidation dataset:")
    print(f"Total images: {len(val_dataset)}")
    for i, class_name in enumerate(val_dataset.classes):
        print(f"    {class_name}: {val_class_counts[i]} images")


    # 绘制数据分布柱状图
    def plot_dataset_distribution(class_names, class_counts, dataset_name, color):
        plt.bar(class_names, class_counts, color=color, alpha=0.6, label=dataset_name)


    # 设置柱状图参数
    plt.figure(figsize=(10, 6))

    # 绘制训练集数据分布
    plot_dataset_distribution(train_dataset.classes, train_class_counts, "Training dataset", color='blue')

    # 绘制测试集数据分布
    plot_dataset_distribution(test_dataset.classes, test_class_counts, "Testing dataset", color='green')

    # 绘制验证集数据分布
    plot_dataset_distribution(val_dataset.classes, val_class_counts, "Validation dataset", color='orange')

    # 添加图例、标签和标题
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Dataset Distribution')
    plt.xticks(rotation=45)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()
