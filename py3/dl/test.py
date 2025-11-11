import torch
import torch.nn as nn
import torch.optim as optim

# 创建简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNN()

# 数据：100个样本，每个样本3个特征
X = torch.randn(100, 3)
y = torch.randn(100, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

# 训练模型
for epoch in range(100):  # 训练100个epoch
    optimizer.zero_grad()  # 清空梯度
    output = model(X)  # 前向传播
    loss = criterion(output, y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
