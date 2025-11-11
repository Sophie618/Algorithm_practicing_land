"""
PyTorch 常见面试问答题
理论 + 代码实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 70)
print("PyTorch 面试常见问答（带代码示例）")
print("=" * 70)


# ============= 问题1：过拟合与正则化 =============
print("\n" + "=" * 50)
print("Q1: 什么是过拟合？如何防止？")
print("=" * 50)

print("""
过拟合：模型在训练集上表现好，但在测试集上表现差。

防止过拟合的方法：
1. Dropout
2. L2正则化（权重衰减）
3. 数据增强
4. Early Stopping
5. Batch Normalization
6. 减少模型复杂度
7. 增加训练数据
""")

# 代码示例：Dropout
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(0.5)  # 随机丢弃50%的神经元
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 训练时启用，测试时自动关闭
        x = self.fc2(x)
        return x

# 代码示例：L2正则化（权重衰减）
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2正则化
print("✓ Dropout和L2正则化示例")


# ============= 问题2：梯度消失和梯度爆炸 =============
print("\n" + "=" * 50)
print("Q2: 什么是梯度消失/爆炸？如何解决？")
print("=" * 50)

print("""
梯度消失：反向传播时梯度越来越小，导致底层参数无法更新
梯度爆炸：反向传播时梯度越来越大，导致参数更新过大

解决方法：
1. 使用ReLU等激活函数（避免Sigmoid/Tanh）
2. Batch Normalization
3. 残差连接（ResNet）
4. 梯度裁剪
5. 合适的权重初始化
""")

# 代码示例：梯度裁剪
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

output = model(x)
loss = ((output - y) ** 2).mean()
loss.backward()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
print("✓ 梯度裁剪示例")


# ============= 问题3：优化器对比 =============
print("\n" + "=" * 50)
print("Q3: SGD、Momentum、Adam有什么区别？")
print("=" * 50)

print("""
SGD (随机梯度下降):
  - 最基础的优化器
  - w = w - lr * grad
  - 简单但可能震荡

SGD with Momentum:
  - 加入动量，考虑历史梯度
  - 加速收敛，减少震荡

Adam (Adaptive Moment Estimation):
  - 自适应学习率
  - 结合Momentum和RMSProp
  - 最常用，收敛快
  - 适合大多数场景

推荐：优先使用Adam，特殊情况用SGD+Momentum
""")

# 代码对比
model = nn.Linear(10, 1)

optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

print("✓ 三种优化器创建成功")


# ============= 问题4：Batch Normalization =============
print("\n" + "=" * 50)
print("Q4: Batch Normalization的作用是什么？")
print("=" * 50)

print("""
Batch Normalization (BN) 的作用：
1. 加速训练（可以用更大的学习率）
2. 减少对初始化的敏感性
3. 有轻微的正则化效果
4. 缓解梯度消失/爆炸

原理：对每个batch的数据进行归一化
  x_norm = (x - mean) / sqrt(var + eps)
  output = gamma * x_norm + beta  # gamma和beta是可学习参数

位置：通常放在线性层和激活函数之间
  Conv/Linear -> BN -> Activation
""")

# 代码示例
class ModelWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # 对卷积层使用BN
        
        self.fc1 = nn.Linear(100, 50)
        self.bn2 = nn.BatchNorm1d(50)  # 对全连接层使用BN
    
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # 全连接层
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x

print("✓ Batch Normalization示例")


# ============= 问题5：损失函数选择 =============
print("\n" + "=" * 50)
print("Q5: 不同任务应该选择什么损失函数？")
print("=" * 50)

print("""
任务类型              损失函数                  PyTorch实现
------------------------------------------------------------
回归                  均方误差 MSE             nn.MSELoss()
                     平均绝对误差 MAE          nn.L1Loss()

二分类                二元交叉熵               nn.BCELoss()
                     (带Sigmoid)              nn.BCEWithLogitsLoss()

多分类                交叉熵                   nn.CrossEntropyLoss()
                     (自动Softmax)

目标检测              Smooth L1               nn.SmoothL1Loss()

生成对抗网络          对抗损失                自定义
""")

# 代码示例
# 回归
mse_loss = nn.MSELoss()
pred = torch.randn(10, 1)
target = torch.randn(10, 1)
loss = mse_loss(pred, target)
print(f"MSE Loss: {loss.item():.4f}")

# 二分类
bce_loss = nn.BCEWithLogitsLoss()
pred = torch.randn(10, 1)  # logits（未经sigmoid）
target = torch.randint(0, 2, (10, 1)).float()
loss = bce_loss(pred, target)
print(f"BCE Loss: {loss.item():.4f}")

# 多分类
ce_loss = nn.CrossEntropyLoss()
pred = torch.randn(10, 5)  # 10个样本，5个类别
target = torch.randint(0, 5, (10,))
loss = ce_loss(pred, target)
print(f"CE Loss: {loss.item():.4f}")


# ============= 问题6：学习率调度 =============
print("\n" + "=" * 50)
print("Q6: 如何调整学习率？")
print("=" * 50)

print("""
学习率调度策略：

1. StepLR: 每N个epoch降低学习率
2. MultiStepLR: 在指定的epoch降低学习率
3. ExponentialLR: 指数衰减
4. ReduceLROnPlateau: 当指标不再改善时降低学习率（最常用！）
5. CosineAnnealingLR: 余弦退火
""")

# 代码示例
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 方法1: 每3个epoch学习率乘以0.1
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 方法2: 在epoch 30和80时降低学习率
scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# 方法3: 当loss不再下降时降低学习率（推荐！）
scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.1, patience=10)

# 使用示例
for epoch in range(5):
    # 训练代码...
    train_loss = 0.5  # 假设的loss
    
    # StepLR和MultiStepLR的使用
    scheduler1.step()
    
    # ReduceLROnPlateau需要传入监控的指标
    # scheduler3.step(train_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")


# ============= 问题7：模型保存和加载 =============
print("\n" + "=" * 50)
print("Q7: 如何保存和加载模型？")
print("=" * 50)

print("""
两种方法：

方法1: 保存整个模型（不推荐）
  torch.save(model, 'model.pth')
  model = torch.load('model.pth')

方法2: 只保存参数（推荐！）
  torch.save(model.state_dict(), 'model_params.pth')
  model = MyModel()
  model.load_state_dict(torch.load('model_params.pth'))

保存完整训练状态（包括优化器）：
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
  }, 'checkpoint.pth')
""")

# 代码示例
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters())

# 保存检查点
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.5,
}
torch.save(checkpoint, 'checkpoint_demo.pth')

# 加载检查点
checkpoint = torch.load('checkpoint_demo.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"✓ 加载成功：epoch={epoch}, loss={loss}")

# 清理
import os
os.remove('checkpoint_demo.pth')


# ============= 问题8：迁移学习 =============
print("\n" + "=" * 50)
print("Q8: 什么是迁移学习？如何实现？")
print("=" * 50)

print("""
迁移学习：使用预训练模型的知识来解决新任务

常见做法：
1. 使用预训练模型（如ResNet、VGG）作为特征提取器
2. 冻结前面的层，只训练最后几层
3. Fine-tuning：用小学习率微调所有层

优点：
- 训练更快
- 需要更少的数据
- 通常效果更好
""")

# 代码示例（伪代码）
print("""
# 加载预训练模型
import torchvision.models as models
resnet = models.resnet18(pretrained=True)

# 冻结所有层
for param in resnet.parameters():
    param.requires_grad = False

# 替换最后一层
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10个类别

# 只训练最后一层
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
""")


# ============= 问题9：数据增强 =============
print("\n" + "=" * 50)
print("Q9: 什么是数据增强？常用方法有哪些？")
print("=" * 50)

print("""
数据增强：通过对训练数据进行变换，生成更多样本

常用方法：
1. 图像：
   - 随机裁剪 RandomCrop
   - 随机翻转 RandomHorizontalFlip
   - 随机旋转 RandomRotation
   - 颜色抖动 ColorJitter
   - 随机擦除 RandomErasing

2. 文本：
   - 同义词替换
   - 回译
   - 随机插入/删除

代码示例：
""")

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("✓ 数据增强pipeline创建成功")


# ============= 问题10：自定义Dataset =============
print("\n" + "=" * 50)
print("Q10: 如何创建自定义Dataset？")
print("=" * 50)

print("""
需要继承 torch.utils.data.Dataset 并实现：
1. __init__: 初始化
2. __len__: 返回数据集大小
3. __getitem__: 返回单个样本
""")

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """自定义数据集示例"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# 使用示例
data = torch.randn(100, 3, 32, 32)
labels = torch.randint(0, 10, (100,))

dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

print(f"✓ 自定义Dataset创建成功，样本数: {len(dataset)}")


print("\n" + "=" * 70)
print("面试问答完成！")
print("=" * 70)
print("\n最后的建议：")
print("1. 理解原理比记住代码更重要")
print("2. 能用自己的话解释清楚概念")
print("3. 多练习，形成肌肉记忆")
print("4. 不懂就诚实说不懂，但要表现出学习意愿")
print("5. 面试时边写边讲解你的思路")

