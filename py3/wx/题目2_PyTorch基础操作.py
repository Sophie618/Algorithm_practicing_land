"""
PyTorch 基础操作 - 必须掌握的核心内容
这些是面试中最常被问到的基础操作
"""

import torch
import numpy as np

print("PyTorch 版本:", torch.__version__)

# ============= 1. Tensor 创建 =============
print("\n" + "=" * 50)
print("1. Tensor 创建方法")

# 从Python列表创建
t1 = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表创建: {t1}")

# 从NumPy数组创建
arr = np.array([1, 2, 3, 4, 5])
t2 = torch.from_numpy(arr)
print(f"从NumPy创建: {t2}")

# 创建全零/全一矩阵
zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)
print(f"全零矩阵:\n{zeros}")
print(f"全一矩阵:\n{ones}")

# 创建随机矩阵（最常用）
rand = torch.rand(2, 3)  # 0-1之间的均匀分布
randn = torch.randn(2, 3)  # 标准正态分布
print(f"随机矩阵(均匀):\n{rand}")
print(f"随机矩阵(正态):\n{randn}")


# ============= 2. Tensor 形状操作 =============
print("\n" + "=" * 50)
print("2. Tensor 形状操作（超级重要！）")

x = torch.randn(2, 3, 4)
print(f"原始形状: {x.shape}")  # torch.Size([2, 3, 4])

# reshape: 改变形状（可能复制数据）
reshaped = x.reshape(6, 4)
print(f"reshape后: {reshaped.shape}")

# view: 改变形状（不复制数据，必须连续）
viewed = x.view(-1, 4)  # -1表示自动计算
print(f"view后: {viewed.shape}")

# squeeze: 去除大小为1的维度
x_with_1 = torch.randn(2, 1, 3, 1, 4)
squeezed = x_with_1.squeeze()
print(f"squeeze前: {x_with_1.shape}, squeeze后: {squeezed.shape}")

# unsqueeze: 增加维度
unsqueezed = x.unsqueeze(0)  # 在第0维增加
print(f"unsqueeze后: {unsqueezed.shape}")


# ============= 3. Tensor 运算 =============
print("\n" + "=" * 50)
print("3. Tensor 运算")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 基本运算
print(f"加法: {a + b}")
print(f"减法: {a - b}")
print(f"乘法(逐元素): {a * b}")
print(f"除法: {a / b}")

# 矩阵乘法（超级重要！）
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C1 = torch.matmul(A, B)  # 方法1
C2 = A @ B  # 方法2（推荐）
print(f"矩阵乘法 ({A.shape} @ {B.shape}) = {C1.shape}")

# 常用数学函数
x = torch.tensor([1.0, 2.0, 3.0])
print(f"平方: {x ** 2}")
print(f"平方根: {torch.sqrt(x)}")
print(f"指数: {torch.exp(x)}")
print(f"对数: {torch.log(x)}")


# ============= 4. Tensor 索引和切片 =============
print("\n" + "=" * 50)
print("4. Tensor 索引和切片")

x = torch.arange(24).reshape(4, 6)
print(f"原始矩阵:\n{x}")

# 索引
print(f"第1行: {x[0]}")
print(f"第2列: {x[:, 1]}")
print(f"左上角2x2: \n{x[:2, :2]}")

# 条件索引（非常有用！）
mask = x > 10
print(f"大于10的元素: {x[mask]}")


# ============= 5. 自动求导（核心！） =============
print("\n" + "=" * 50)
print("5. 自动求导 - PyTorch的核心功能")

# 创建需要梯度的tensor
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 定义计算
z = x ** 2 + y ** 3  # z = x^2 + y^3
print(f"z = {z.item()}")

# 反向传播计算梯度
z.backward()

# 查看梯度
print(f"dz/dx = {x.grad.item()}")  # 2x = 4
print(f"dz/dy = {y.grad.item()}")  # 3y^2 = 27


# ============= 6. 实战：简单的线性回归 =============
print("\n" + "=" * 50)
print("6. 实战：手动实现梯度下降")

# 生成数据 y = 2x + 3 + noise
torch.manual_seed(42)
X = torch.randn(100, 1)
y_true = 2 * X + 3 + 0.1 * torch.randn(100, 1)

# 初始化参数
w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

# 训练参数
learning_rate = 0.01
num_epochs = 100

for epoch in range(num_epochs):
    # 前向传播
    y_pred = X @ w + b  # 矩阵乘法
    
    # 计算损失（均方误差）
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数（手动）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清空梯度（重要！）
        w.grad.zero_()
        b.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"最终参数: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"真实参数: w = 2.0000, b = 3.0000")


# ============= 7. GPU 操作 =============
print("\n" + "=" * 50)
print("7. GPU 操作")

# 检查是否有GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU 可用！")
    
    # 创建tensor并移到GPU
    x_gpu = torch.randn(3, 3).to(device)
    print(f"Tensor 在 GPU 上: {x_gpu.device}")
    
    # 移回CPU
    x_cpu = x_gpu.cpu()
    print(f"Tensor 在 CPU 上: {x_cpu.device}")
else:
    print("GPU 不可用，使用 CPU")
    device = torch.device("cpu")


# ============= 8. 常用技巧 =============
print("\n" + "=" * 50)
print("8. 常用技巧")

# 设置随机种子（保证可复现）
torch.manual_seed(42)

# 禁用梯度计算（推理时节省内存）
with torch.no_grad():
    x = torch.randn(2, 2)
    y = x * 2
    print(f"不计算梯度: {y.requires_grad}")

# 张量拼接
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.cat([a, b], dim=0)  # 按行拼接
print(f"拼接结果:\n{c}")

# 统计操作
x = torch.randn(3, 4)
print(f"均值: {x.mean()}")
print(f"标准差: {x.std()}")
print(f"最大值: {x.max()}")
print(f"按行求和: {x.sum(dim=1)}")


print("\n" + "=" * 50)
print("PyTorch基础操作完成！这些是面试必考内容")
print("重点记住：")
print("1. Tensor的创建和形状操作")
print("2. requires_grad=True 和 .backward()")
print("3. 梯度清零 optimizer.zero_grad()")
print("4. 矩阵乘法 @")

