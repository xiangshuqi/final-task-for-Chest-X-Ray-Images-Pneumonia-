import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from model import SimplifiedCNNModel
from dataset import test_dir

# 图像转换器（与训练保持一致）
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载测试数据集
test_dataset = ImageFolder(root=test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimplifiedCNNModel()
model.load_state_dict(torch.load('best_model.pth'))  # 加载训练好的权重
model = model.to(device)
model.eval()  # 设置为评估模式

# 评估模型性能
all_labels = []
all_preds = []
all_probs = []  # 保存预测概率，用于计算 AUC 和绘制 ROC 曲线

with torch.no_grad():  # 禁用梯度计算
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        probs = torch.sigmoid(outputs).squeeze()  # 使用 sigmoid 获取概率
        preds = probs > 0.5  # 使用 0.5 阈值进行分类

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 将预测值和真实标签转为 NumPy 数组
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 计算评估指标
test_accuracy = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, pos_label=1)
test_recall = recall_score(all_labels, all_preds, pos_label=1)
test_f1 = f1_score(all_labels, all_preds, pos_label=1)
test_auc = roc_auc_score(all_labels, all_probs)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
class_names = test_dataset.classes  # 获取类别名 ['NORMAL', 'PNEUMONIA']

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# ROC 曲线
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {test_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# 可视化部分测试集图片及预测结果
def imshow(img, title):
    """显示图像的辅助函数"""
    img = img.numpy().transpose((1, 2, 0))  # 转换为 HWC 格式
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 去标准化
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


# 选择部分测试集图片
inputs, labels = next(iter(test_loader))
inputs, labels = inputs[:8], labels[:8]  # 选择前 8 张图片
inputs = inputs.to(device)
labels = labels.numpy()

with torch.no_grad():
    outputs = model(inputs)
    probs = torch.sigmoid(outputs).squeeze()
    preds = (probs > 0.5).cpu().numpy()

# 显示图像及结果
plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    imshow(inputs[i].cpu(), title=f"True: {class_names[labels[i]]}\nPred: {class_names[int(preds[i])]}")
plt.tight_layout()
plt.show()
