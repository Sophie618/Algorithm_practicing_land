"""
实战项目：MNIST手写数字识别
这是最经典的深度学习入门项目，面试时可能让你现场实现！
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

print("=" * 70)
print("实战：MNIST手写数字识别 - 完整项目")
print("=" * 70)


# ============= 1. 数据准备 =============
print("\n步骤1: 数据加载和预处理")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST的均值和标准差）
])

# 下载并加载数据（第一次会下载）
try:
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 查看一批数据
    images, labels = next(iter(train_loader))
    print(f"图像形状: {images.shape}")  # [batch_size, 1, 28, 28]
    print(f"标签形状: {labels.shape}")  # [batch_size]
    
    DATA_LOADED = True
except Exception as e:
    print(f"数据加载失败: {e}")
    print("将使用假数据进行演示")
    DATA_LOADED = False


# ============= 2. 模型定义 =============
print("\n步骤2: 定义CNN模型")

class MNISTNet(nn.Module):
    """MNIST分类的CNN模型"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 尺寸减半
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个类别
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 输入: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))      # (batch, 32, 28, 28)
        x = self.pool(x)               # (batch, 32, 14, 14)
        
        x = F.relu(self.conv2(x))      # (batch, 64, 14, 14)
        x = self.pool(x)               # (batch, 64, 7, 7)
        
        x = x.view(-1, 64 * 7 * 7)     # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                # (batch, 10)
        
        return x


# 方法2：更简洁的实现
class MNISTNetSimple(nn.Module):
    """简洁版MNIST网络"""
    def __init__(self):
        super(MNISTNetSimple, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = MNISTNet().to(device)
print(model)

# 统计参数
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n可训练参数数量: {total_params:,}")


# ============= 3. 训练和评估函数 =============
print("\n步骤3: 定义训练和评估函数")

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ============= 4. 训练模型 =============
if DATA_LOADED:
    print("\n步骤4: 训练模型")
    
    # 超参数
    num_epochs = 5
    learning_rate = 0.001
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 打印结果
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_mnist_model.pth')
            print(f"  ✓ 保存最佳模型 (准确率: {best_acc:.2f}%)")
    
    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")


# ============= 5. 模型推理 =============
if DATA_LOADED:
    print("\n步骤5: 模型推理")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_mnist_model.pth'))
    model.eval()
    
    # 随机选择几个测试样本
    import random
    indices = random.sample(range(len(test_dataset)), 5)
    
    print("\n预测结果:")
    with torch.no_grad():
        for idx in indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)  # 增加batch维度
            
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted].item()
            
            print(f"  真实标签: {label}, 预测: {predicted}, 置信度: {confidence:.2%}")
    
    # 清理模型文件
    import os
    if os.path.exists('best_mnist_model.pth'):
        os.remove('best_mnist_model.pth')


# ============= 6. 混淆矩阵（进阶） =============
if DATA_LOADED:
    print("\n步骤6: 生成混淆矩阵")
    
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵（部分）:")
    print(cm[:5, :5])  # 只显示5x5
    
    # 分类报告
    print("\n分类报告:")
    report = classification_report(all_labels, all_preds, 
                                   target_names=[str(i) for i in range(10)])
    print(report)


# ============= 7. 面试要点总结 =============
print("\n" + "=" * 70)
print("面试要点总结")
print("=" * 70)

print("""
这个完整项目展示了以下关键点：

1. 数据预处理:
   - transforms.Compose() 组合多个变换
   - ToTensor() + Normalize() 标准化
   - DataLoader 批量加载

2. 模型设计:
   - 卷积层提取特征
   - 池化层降维
   - 全连接层分类
   - Dropout 防止过拟合

3. 训练流程:
   - model.train() / model.eval() 切换模式
   - optimizer.zero_grad() 清空梯度
   - loss.backward() 反向传播
   - optimizer.step() 更新参数

4. 评估指标:
   - 准确率 (Accuracy)
   - 混淆矩阵 (Confusion Matrix)
   - 分类报告 (Precision, Recall, F1-Score)

5. 技巧:
   - 学习率调度器
   - 模型保存/加载
   - GPU加速
   - 置信度计算

面试时如果被要求现场实现，按照这个模板来即可！
""")

print("\n重要提醒：")
print("1. 训练时记得 model.train()，测试时记得 model.eval()")
print("2. 每个batch之前记得 optimizer.zero_grad()")
print("3. 交叉熵损失会自动应用Softmax，输出层不需要Softmax")
print("4. 推理时用 with torch.no_grad() 节省内存")

