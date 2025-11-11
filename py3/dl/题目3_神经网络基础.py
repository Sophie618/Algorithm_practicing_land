"""
PyTorch 神经网络基础 - 最常见的面试题
掌握这个文件，80%的PyTorch面试题你都能应对
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 70)
print("神经网络基础 - 这是面试的核心内容！")
print("=" * 70)


# ============= 1. 最简单的神经网络：线性回归 =============
print("\n" + "=" * 50)
print("题目1：使用 nn.Module 实现线性回归")

class LinearRegression(nn.Module):
    """线性回归模型：y = wx + b"""
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# 生成训练数据
torch.manual_seed(42)
X_train = torch.randn(100, 1)
y_train = 2 * X_train + 3 + 0.1 * torch.randn(100, 1)

# 创建模型
model = LinearRegression(input_dim=1, output_dim=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环（必须背下来！）
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 查看学到的参数
print(f"学到的参数: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")
print(f"真实参数: w=2.0000, b=3.0000")


# ============= 2. 两层全连接神经网络（最经典！） =============
print("\n" + "=" * 50)
print("题目2：实现一个两层全连接神经网络（面试必考！）")

class TwoLayerNet(nn.Module):
    """两层全连接神经网络"""
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        # 第一层：输入层到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 激活函数
        self.relu = nn.ReLU()
        # 第二层：隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 前向传播过程
        x = self.fc1(x)      # 线性变换
        x = self.relu(x)     # 非线性激活
        x = self.fc2(x)      # 输出层
        return x

# 创建模型
model = TwoLayerNet(input_size=10, hidden_size=20, output_size=2)
print(model)

# 测试前向传播
x = torch.randn(5, 10)  # batch_size=5, input_size=10
output = model(x)
print(f"输入形状: {x.shape}, 输出形状: {output.shape}")


# ============= 3. 完整的分类任务：二分类 =============
print("\n" + "=" * 50)
print("题目3：完整的二分类训练流程")

class BinaryClassifier(nn.Module):
    """二分类模型"""
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 生成二分类数据
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# 创建模型
model = BinaryClassifier(input_size=20)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 50
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        # 评估模式
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}")


# ============= 4. 多分类问题 =============
print("\n" + "=" * 50)
print("题目4：多分类神经网络")

class MultiClassifier(nn.Module):
    """多分类模型"""
    def __init__(self, input_size, num_classes):
        super(MultiClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# 生成多分类数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                          n_informative=15, n_redundant=5, random_state=42)
X_train = torch.FloatTensor(X)
y_train = torch.LongTensor(y)

# 创建模型
model = MultiClassifier(input_size=20, num_classes=3)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵（会自动应用Softmax）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(50):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_train).float().mean()
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")


# ============= 5. 激活函数对比 =============
print("\n" + "=" * 50)
print("题目5：常用激活函数（面试常问）")

x = torch.linspace(-5, 5, 100)

# ReLU: max(0, x)
relu = nn.ReLU()
print(f"ReLU: 负数变0，正数不变")

# Sigmoid: 1 / (1 + e^(-x))
sigmoid = nn.Sigmoid()
print(f"Sigmoid: 输出范围(0,1)，用于二分类")

# Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
tanh = nn.Tanh()
print(f"Tanh: 输出范围(-1,1)")

# LeakyReLU: max(0.01x, x)
leaky_relu = nn.LeakyReLU(0.01)
print(f"LeakyReLU: 解决ReLU的死神经元问题")


# ============= 6. 模型保存和加载（重要！） =============
print("\n" + "=" * 50)
print("题目6：模型保存和加载")

# 创建一个简单模型
model = TwoLayerNet(10, 20, 2)

# 方法1：保存整个模型
torch.save(model, 'model_complete.pth')
loaded_model = torch.load('model_complete.pth')

# 方法2：只保存参数（推荐）
torch.save(model.state_dict(), 'model_params.pth')
new_model = TwoLayerNet(10, 20, 2)
new_model.load_state_dict(torch.load('model_params.pth'))

print("模型保存和加载成功！")

# 清理文件
import os
os.remove('model_complete.pth')
os.remove('model_params.pth')


# ============= 7. 使用 DataLoader（重要！） =============
print("\n" + "=" * 50)
print("题目7：使用 DataLoader 批量加载数据")

from torch.utils.data import TensorDataset, DataLoader

# 创建数据集
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 遍历数据
model = TwoLayerNet(10, 20, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for batch_idx, (data, target) in enumerate(dataloader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} 完成")


# ============= 8. 自定义损失函数 =============
print("\n" + "=" * 50)
print("题目8：自定义损失函数")

class CustomLoss(nn.Module):
    """自定义损失函数示例"""
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        # 例：均方误差 + L1正则化
        mse = torch.mean((predictions - targets) ** 2)
        l1_reg = torch.mean(torch.abs(predictions))
        return mse + 0.01 * l1_reg

custom_loss = CustomLoss()
pred = torch.randn(10, 1)
target = torch.randn(10, 1)
loss = custom_loss(pred, target)
print(f"自定义损失: {loss.item():.4f}")


print("\n" + "=" * 70)
print("神经网络基础完成！")
print("=" * 70)
print("\n必须记住的要点：")
print("1. nn.Module 的标准写法：__init__ 和 forward")
print("2. 训练三步曲：optimizer.zero_grad() -> loss.backward() -> optimizer.step()")
print("3. 二分类用 BCELoss，多分类用 CrossEntropyLoss")
print("4. model.train() 和 model.eval() 的区别")
print("5. torch.save() 和 torch.load() 保存加载模型")

