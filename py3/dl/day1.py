import torch

# 检查 PyTorch 是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 创建一个随机的 tensor
x = torch.rand(5, 5).to(device)
print("随机的 Tensor:")
print(x)

# 确认 tensor 加法
y = torch.rand(5, 5).to(device)
result = x + y
print("Tensor 加法结果:")
print(result)

# 执行一个简单的矩阵乘法
a = torch.rand(3, 3).to(device)
b = torch.rand(3, 3).to(device)
matrix_multiplication_result = torch.matmul(a, b)
print("矩阵乘法结果:")
print(matrix_multiplication_result)
x=torch.rand(5,5)