"""
卷积神经网络 (CNN) - 深度学习的核心
这是面试中最常被问到的进阶内容
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("=" * 70)
print("卷积神经网络 (CNN) - 图像处理的核心")
print("=" * 70)


# ============= 1. 理解卷积层 =============
print("\n" + "=" * 50)
print("1. 卷积层的基本概念")

# 创建一个简单的卷积层
conv = nn.Conv2d(
    in_channels=3,      # 输入通道数（如RGB图像为3）
    out_channels=16,    # 输出通道数（卷积核数量）
    kernel_size=3,      # 卷积核大小 3x3
    stride=1,           # 步长
    padding=1           # 填充
)

# 输入：batch_size=1, channels=3, height=32, width=32
x = torch.randn(1, 3, 32, 32)
output = conv(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")


# ============= 2. 经典CNN结构：LeNet风格 =============
print("\n" + "=" * 50)
print("2. 实现一个简单的CNN分类器（面试常考！）")

class SimpleCNN(nn.Module):
    """简单的CNN分类器 - 适用于MNIST/CIFAR等"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层1：3通道 -> 32通道
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化
        
        # 卷积层2：32通道 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层：减小特征图尺寸
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 输入: (batch, 3, 32, 32)
        
        # 第一个卷积块
        x = self.conv1(x)           # (batch, 32, 32, 32)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 32, 16, 16)
        
        # 第二个卷积块
        x = self.conv2(x)           # (batch, 64, 16, 16)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 64, 8, 8)
        
        # 展平
        x = x.view(x.size(0), -1)   # (batch, 64*8*8)
        
        # 全连接层
        x = self.fc1(x)             # (batch, 512)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)             # (batch, num_classes)
        
        return x

# 创建模型并测试
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 32, 32)  # batch_size=4
output = model(x)
print(f"模型输出形状: {output.shape}")  # (4, 10)

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数量: {total_params:,}")


# ============= 3. 使用 Sequential 简化代码 =============
print("\n" + "=" * 50)
print("3. 使用 nn.Sequential 简化CNN定义")

class CNNSequential(nn.Module):
    """使用Sequential定义CNN"""
    def __init__(self, num_classes=10):
        super(CNNSequential, self).__init__()
        
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二层卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

model_seq = CNNSequential()
print("Sequential版本的CNN创建成功")


# ============= 4. 完整训练示例（假数据） =============
print("\n" + "=" * 50)
print("4. 完整的CNN训练流程")

# 生成假数据（模拟CIFAR-10）
def generate_fake_data(num_samples=100, num_classes=10):
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

X_train, y_train = generate_fake_data(100, 10)
X_test, y_test = generate_fake_data(20, 10)

# 创建DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 创建模型
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数（标准模板！）
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 评估函数（标准模板！）
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")


# ============= 5. 常用CNN层详解 =============
print("\n" + "=" * 50)
print("5. CNN常用层的作用（面试常问）")

print("""
1. Conv2d (卷积层):
   - 作用: 提取局部特征（边缘、纹理等）
   - 参数: in_channels, out_channels, kernel_size, stride, padding

2. MaxPool2d (最大池化):
   - 作用: 降低特征图尺寸，减少参数，增加感受野
   - 参数: kernel_size, stride

3. BatchNorm2d (批归一化):
   - 作用: 加速训练，稳定梯度，轻微正则化
   - 通常放在卷积和激活之间

4. Dropout:
   - 作用: 防止过拟合，随机丢弃神经元
   - 训练时启用，测试时自动关闭

5. AdaptiveAvgPool2d (自适应平均池化):
   - 作用: 将任意尺寸特征图转为固定尺寸
""")


# ============= 6. 残差块 (ResNet风格) =============
print("\n" + "=" * 50)
print("6. 残差块（ResNet的核心，面试加分项）")

class ResidualBlock(nn.Module):
    """残差块 - 解决深层网络梯度消失问题"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 短路连接（跳跃连接）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)  # 保存输入
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # 残差连接！
        out = F.relu(out)
        
        return out

# 测试残差块
resblock = ResidualBlock(64, 64)
x = torch.randn(1, 64, 32, 32)
output = resblock(x)
print(f"残差块输入: {x.shape}, 输出: {output.shape}")


# ============= 7. 1x1卷积的妙用 =============
print("\n" + "=" * 50)
print("7. 1x1卷积（面试常问）")

print("""
1x1卷积的作用：
1. 降维/升维: 改变通道数
2. 增加非线性: 后接ReLU增加表达能力
3. 跨通道信息融合

示例：
""")

# 降维示例
conv_1x1 = nn.Conv2d(256, 64, kernel_size=1)  # 256维降到64维
x = torch.randn(1, 256, 14, 14)
output = conv_1x1(x)
print(f"1x1卷积降维: {x.shape} -> {output.shape}")


print("\n" + "=" * 70)
print("CNN卷积神经网络完成！")
print("=" * 70)
print("\n必须记住的要点：")
print("1. CNN结构: 卷积层 -> 激活 -> 池化 -> ... -> 全连接")
print("2. BatchNorm 加速训练，Dropout 防止过拟合")
print("3. 残差连接: out = F.relu(F(x) + x)")
print("4. 训练时 model.train()，测试时 model.eval()")
print("5. 特征提取 + 分类器的两段式结构")

