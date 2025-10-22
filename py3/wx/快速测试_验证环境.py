"""
快速测试脚本 - 验证你的环境是否正常
面试前运行这个脚本，确保一切正常！
"""

import sys

print("=" * 70)
print("PyTorch 环境检查")
print("=" * 70)

# 1. 检查Python版本
print(f"\n✓ Python版本: {sys.version}")

# 2. 检查PyTorch
try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA不可用，将使用CPU（面试够用）")
    
except ImportError:
    print("❌ PyTorch未安装！")
    print("   安装命令: pip install torch torchvision")
    sys.exit(1)

# 3. 检查NumPy
try:
    import numpy as np
    print(f"✓ NumPy版本: {np.__version__}")
except ImportError:
    print("❌ NumPy未安装！")
    sys.exit(1)

# 4. 检查其他常用库
try:
    import sklearn
    print(f"✓ scikit-learn版本: {sklearn.__version__}")
except ImportError:
    print("⚠ scikit-learn未安装（可选）")

# 5. 快速功能测试
print("\n" + "=" * 70)
print("功能测试")
print("=" * 70)

# 测试Tensor操作
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = x @ y
print(f"✓ Tensor运算正常: {z.shape}")

# 测试自动求导
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(f"✓ 自动求导正常: dy/dx = {x.grad.item()}")

# 测试神经网络
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
test_input = torch.randn(5, 10)
output = model(test_input)
print(f"✓ 神经网络正常: 输入{test_input.shape} -> 输出{output.shape}")

print("\n" + "=" * 70)
print("环境检查完成！✓ 一切正常，可以开始准备了！")
print("=" * 70)

print("\n建议的学习顺序：")
print("1. 题目1_Python基础编程.py")
print("2. 题目2_PyTorch基础操作.py")
print("3. 题目3_神经网络基础.py")
print("4. 题目4_CNN卷积神经网络.py")
print("5. 题目5_实战MNIST手写数字识别.py")
print("6. 题目6_常见面试问答.py")

print("\n祝你面试顺利！💪")

